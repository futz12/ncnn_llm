# Xiaomi-OCR-0 部署案例：边缘端大模型推理的内核级调试

> 本文记录在 ncnn_llm 上部署 Xiaomi-OCR-0（Qwen3.5-VL 混合架构）的完整过程，
> 重点展示**推理内核层面的问题定位与修复**，而非简单的模型跑通。
>
> 适用读者：边缘端大模型部署工程师、推理框架开发者。

---

## 1. 项目概述

**目标**：在 CPU（i5-12600KF，无 GPU 加速）上用 ncnn_llm 跑通 Xiaomi-OCR-0，
实现 receipt/手写公式/密集公式/中文书法 4 类图像的 OCR。

**模型架构**（Qwen3.5-VL 混合）：
- 文本塔：24 层 = 6 层 SDPA（全注意力） + 18 层 linear_attn（GatedDeltaRule 线性注意力）
- 视觉塔：12 层 ViT（patch 16×16，spatial_merge 2×2）
- hidden 1024，head_dim 256，partial_rotary 0.25，mrope_interleaved
- vocab 248320，tie_word_embeddings

**最终成果**：
- int8 模型（1.2GB）和 fp32 模型（6.4GB）均跑通，4 张测试图全部正确
- 定位并修复 **5 个 bug**，其中 3 个在推理内核/导出层面

---

## 2. 修复的 5 个 Bug（核心技术亮点）

### Bug 1：int8 GEMM 内核列错位（内核级）

**现象**：decoder 输出全乱码，int8 模型无法使用。

**定位**：逐层 dump 对比发现 vision 的 `proj_0`（第一个注意力输出投影）
列 0-191 正确、列 192-767 全部错误（maxerr 2.638）。

**根因**：ncnn 的 `gemm_BT_x86` 在 **constantB（权重预打包）** 路径下，
`TILE_N` 与 `create_pipeline` 打包时不一致：
- 打包时 `M=0` → 走 `M<=1` 分支 → `TILE_N = N/(nT*2) = 192`
- 前向时 `M=5916` → 走 `M>1` 分支 → `TILE_N = N/nT = 384`

打包按 TILE_N=192 分通道，前向按 TILE_N=384 读取，导致列 192-767 读错通道。

**修复**：在 `gemm_BT_x86` 取到 tile 配置后，从预打包的 `BT_data` 反推真实 TILE_N：
```cpp
if (BT.w > 0 && TILE_K > 0 && BT.w % TILE_K == 0) {
    int bttile_n = BT.w / TILE_K;
    if (bttile_n < TILE_N) TILE_N = bttile_n;
}
```

**验证**：`proj_0` 全列 maxerr 从 2.638 降到 **7.6e-6**；vision 输出 std 从 0.666 降到 **0.1207**（HF 0.1209）。

---

### Bug 2：RMSNormGated reshape 序列维塌缩（图优化级）

**现象**：24 层 hidden state 与 HF 误差 0.43（HF bf16 仅 0.06），
经 final norm（权重 ~5）放大后 logits 完全跑偏，输出乱码。

**定位**：逐层对比发现 L0 误差 0.08，且 **channel 0** 异常偏高（0.65 vs 平均 0.08）。
追踪到 GDR 输出归一化路径，发现中间 blob 只有 2048 元素（应为 643072）。

**根因**：pnnx 导出的 reshape 表达式 `"128,*(16,0d)"` 中，`0d` 在 ncnn 3D Mat 上
求值为 **0**（depth 维度未使用），导致序列维塌缩成 1。
GDR 输出 (16,S,128) 被 reshape 成 (128,16)，与完整 silu(gate) (S·16,128) 相乘时
**广播**到所有 token，归一化用了错误的统计量。

**修复**：reshape 表达式 `0d` → `0c`（`0c` = 序列长度），保留 per-token 布局。
修复已内置到导出脚本 `export/qwen35_export_xiaomi_ocr_0.py` 的 `_fix_gdr_reshape()`。

**验证**：L0 误差 0.08 → **0.0008**（100×），24 层全部与 HF 对齐。

---

### Bug 3：image token id 内存损坏（内存级）

**现象**：图像 token 全为 `-117901064`（0xF8F8F8F8）。

**根因**：`inject_image_embeds` 用 `memset(ptr, image_pad_id, h*sizeof(int))`
填充 1479 个图像占位 token，但 `memset` **按字节填充**，
`image_pad_id=248056` 的低字节 `0xF8` 被重复 4 次 → `0xF8F8F8F8`。

**修复**：改为逐 int 循环赋值。

```cpp
for (int i = 0; i < image_embeds.h; ++i)
    token_ids_injected[placeholder_index + i] = image_pad_id;
```

---

### Bug 4：thinking template 不完整（精度级）

**现象**：序列比 HF 少 2 个 token，导致后续 token 位置全部错位。

**根因**：`--enable-thinking` 模板只输出 `\n`（2 个 token），
HF processor 默认输出 `\n\n`（4 个 token）。

**修复**：模板补全为 `\n\n`。

---

### Bug 5：last-token 取错行（逻辑级）

**现象**：首 token 有多余前缀（`ar`/`竖`/`30`），argmax 错误。

**定位**：用 ncnn 的 hidden 通过 HF lm_head 计算，`MR` logit = 48.02（正确），
但 ncnn 自己的 logits = 27.67。对比发现 lm_head 输入的 hidden 与 decode_out 最后一行**不同**。

**根因**：`total_prompt_tokens` 在 `inject_image_embeds` **之前**计算（=22 文本 token），
而 lm_head 用 `row_range(total_prompt_tokens-1, 1)` 取最后一个 token 的 hidden。
导致取的是 row 21（文本最后一个 token）而非 row 1499（序列真正的末尾）。

**修复**：把 `total_prompt_tokens` 移到图像注入之后计算（=1500）。

**验证**：首 token argmax 从 277(`ar`) → **17896(`MR`)**，与 HF 完全一致。

---

## 3. 逐层验证数据（修复后）

| 层 | ncnn vs HF mean_abs | 修复前 |
|---|---|---|
| L0 | 0.0008 | 0.08 |
| L3 | 0.0021 | 0.15 |
| L12 | 0.0037 | 0.27 |
| L23 | 0.0147 | 0.43 |

- vision 特征 vs HF：mean_abs **0.0018**
- mrope cache vs HF：mean_abs **6e-9**（完全一致）
- 图像 embedding 注入：mean_abs **0.0**（精确）

---

## 4. 调试方法论（可复用）

1. **逐层二分定位**：dump 每层输出 → 与 HF hook 对比 → 找第一个发散层
2. **从输出反推放大效应**：final norm 权重 ~5，会放大 hidden 误差 12× → hidden 必须对齐到 bf16 精度
3. **验证输入链每一环**：embed → rmsn → qkv → conv → gdr → norm → gemm，逐环节比对
4. **警惕 reshape 表达式语义**：ncnn 的 `0d/0c/0h` 在 3D Mat 上取值不同（`0d=0`），导出时易错
5. **对比隐藏状态而非只看输出**：logits 错误可能来自 hidden 取错行，而非计算错误
6. **构建可复现的参考**：用 HF hook 抓取每层参考值，是定位偏差的最快方式

---

## 5. 最终效果

4 张测试图全部输出正确内容：

| 图像 | 输出首行 | 状态 |
|---|---|---|
| receipt | `MR. D.I.Y. SDN BHD` | ✅ |
| qianziwen | `行书千文` | ✅ |
| dense-equations | `We then define the following variables:` | ✅ |
| handwrite-formula | `(b) we got Friedmann equation:` | ✅ |

---

## 6. 经验总结

### 边缘部署的核心挑战

1. **精度链式放大**：final norm 的大权重会把 hidden 的小误差放大 12 倍，
   必须确保 hidden 对齐到 bf16 精度，不能只看最终输出是否"像"对的。

2. **量化内核的隐蔽性**：int8 GEMM 的 tile 策略错误不会报错，只表现为数值偏差，
   需要逐列对比才能发现通道错位。

3. **导出工具的信任边界**：pnnx 生成的 reshape 表达式在 3D Mat 上可能有歧义
   （`0d` vs `0c`），导出后必须验证中间张量形状。

4. **多模态的 token 对齐**：图像 token 注入会改变序列长度，
   所有依赖"最后一个 token"的逻辑都必须用注入后的长度。

### 技术栈

- **推理框架**：ncnn（自定义算子 ShortConv/GatedDeltaRule）
- **量化**：block-int8（block=64，minmax 标定）+ bf16 存储
- **模型导出**：pnnx（PyTorch → ncnn 计算图）
- **调试手段**：逐层 dump + HF hook 对比 + 中间张量形状验证

---

> 完整部署教程见 [xiaomi-ocr-deploy-tutorial.md](xiaomi-ocr-deploy-tutorial.md)
> 边缘大模型部署学习路线见 [edge-llm-learning-path.md](edge-llm-learning-path.md)
