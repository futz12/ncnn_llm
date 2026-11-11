# Xiaomi-OCR-0 部署教程：从模型转换到 OCR 跑通

> 本教程记录在 Windows 11 + ncnn_llm 上完整部署 Xiaomi-OCR-0 模型的端到端流程：
> **环境准备 → 模型下载 → 模型转换（导出）→ 库编译 → 模型运行**。
>
> 结合 2026-11-11 成功跑通的实战经验，包含所有踩过的坑和修复方法。

---

## 目录

1. [环境准备](#1-环境准备)
2. [获取代码与依赖](#2-获取代码与依赖)
3. [下载模型](#3-下载模型)
4. [模型转换（导出）](#4-模型转换导出)
5. [修补已知 bug](#5-修补已知-bug)
6. [编译 ncnn_llm](#6-编译-ncnn_llm)
7. [运行 OCR](#7-运行-ocr)
8. [验证与排错](#8-验证与排错)
9. [常见问题](#9-常见问题)

---

## 1. 环境准备

| 项 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Windows 10/11 (x64) | 本教程以 Windows 11 为例 |
| CPU | 支持 AVX2 的 x86 CPU | i5-12600KF 实测；AVX-512 非必需 |
| 内存 | ≥ 16 GB | int8 模型约 1.7 GB，加上运行时缓存建议 16 GB+ |
| 编译器 | **VS 2019+ (MSVC)** | 实测 VS 18 BuildTools (MSVC 14.50) |
| CMake | ≥ 3.20 | 用于配置工程 |
| Ninja | 任意版本 | 用于编译 |
| Python | 3.10+ (conda 环境) | 用于模型导出 |
| Python 包 | torch, transformers, numpy, pnnx | 见下方安装 |

### 1.1 安装 Python 环境

```bash
# 创建 conda 环境（Python 3.12 实测可用）
conda create -n dl312 python=3.12 -y
conda activate dl312

# 安装依赖（版本号实测可用）
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers numpy pnnx
```

> **注意**：`transformers` 需要支持 `Qwen3_5ForConditionalGeneration`（≥ 5.x）。
> `flash-linear-attention` / `causal_conv1d` **不需要**安装（不装会自动回退到参考实现，只是慢一些）。

### 1.2 编译环境

确保 `vcvarsall.bat` 可用（VS BuildTools 默认路径）：

```
C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat
```

---

## 2. 获取代码与依赖

```bash
git clone --recursive https://github.com/HougeLangley/ncnn_llm.git
cd ncnn_llm
git submodule update --init --recursive
```

> `--recursive` 会同时拉取 `ncnn` 子模块（位于 `ncnn/`）。

---

## 3. 下载模型

从 ModelScope 下载 Xiaomi-OCR-0：

```bash
# 方式一：git clone
git clone https://www.modelscope.cn/SeerRay-Lab/Xiaomi-OCR-0.git models/Xiaomi-OCR-0

# 方式二：modelscope CLI
pip install modelscope
modelscope download --model SeerRay-Lab/Xiaomi-OCR-0 --local_dir models/Xiaomi-OCR-0
```

下载后目录结构：

```
models/Xiaomi-OCR-0/
├── config.json
├── model-*.safetensors     # 权重分片
├── tokenizer.json
├── vocab.txt / merges.txt
└── example_pics/           # 测试图 + ground truth
    ├── receipt.jpg / receipt.md
    ├── qianziwen.jpeg / qianziwen.md
    ├── dense-equations.jpg / dense-equations.md
    └── handwrite-formula.jpg / handwrite-formula.md
```

---

## 4. 模型转换（导出）

### 4.1 导出命令

```bash
python export/qwen35_export_xiaomi_ocr_0.py \
    -m models/Xiaomi-OCR-0 \
    -o models/Xiaomi-OCR-0_ncnn \
    --int8-dir models/Xiaomi-OCR-0_ncnn_int8 \
    --bits 8
```

**参数说明**：

| 参数 | 含义 |
|---|---|
| `-m` | HF 模型目录 |
| `-o` | BF16 输出目录（同时保留 fp32） |
| `--int8-dir` | int8 量化输出目录（**OCR 用这个**） |
| `--bits 8` | 量化位宽（block-int8，block=64） |
| `--prefix` | 输出文件名前缀（默认取输出目录名） |

导出过程会生成：

```
models/Xiaomi-OCR-0_ncnn/          # BF16 + fp32
├── model.json                     # 模型配置（type=qwen3.5, vision=qwen3.5_vl）
├── Xiaomi-OCR-0_embed_token.ncnn.param / .bin
├── Xiaomi-OCR-0_decoder.ncnn.param / .bin
├── Xiaomi-OCR-0_proj_out.ncnn.param / .bin
├── Xiaomi-OCR-0_vision_embed_patch.ncnn.param / .bin
├── Xiaomi-OCR-0_vision_embed_pos.ncnn.param / .bin
├── Xiaomi-OCR-0_vision_encoder.ncnn.param / .bin
├── vocab.txt / merges.txt
└── ...

models/Xiaomi-OCR-0_ncnn_int8/     # int8 量化（结构相同，权重为 block-int8）
```

> **导出耗时**：约 2-5 分钟（取决于机器）。日志会显示 pnnx 转换过程。
> 如果看到 `flash-linear-attention is not installed` 的警告，**忽略即可**（会回退到参考实现）。

### 4.2 导出原理（了解即可）

- **文本 decoder**：混合架构（6 层 SDPA + 18 层 linear_attn/GatedDeltaRule）
  - SDPA 层导出为 ncnn `SDPA` + KV-cache 形式
  - linear_attn 层导出为自定义 `ShortConv` + `GatedDeltaRule` 算子
- **Vision tower**：patch_embed → pos_embed → window attention encoder → merger
- **权重**：从 fp32 HF 权重出发，生成 BF16 和 block-INT8 两份拷贝

---

## 5. 修补已知 bug

> ⚠️ **重要**：导出后的模型在直接运行前，需要修补以下已知问题。
> 这些是导出脚本/ncnn 的 bug，不是配置问题。

### 5.1 修复 reshape 表达式（`0d` → `0c`）

**问题**：decoder 的 RMSNormGated 层 reshape 表达式用了 `0d`，在 3D ncnn Mat 上 `0d=0`，导致序列维塌缩，归一化被广播到所有 token。

**修复**（18 处，每个 linear_attn 层一处）：

```bash
python - <<'PY'
p='models/Xiaomi-OCR-0_ncnn_int8/Xiaomi-OCR-0_decoder.ncnn.param'
data=open(p,'rb').read()
old=b'128,*(16,0d)'
new=b'128,*(16,0c)'
cnt=data.count(old)
data=data.replace(old,new)
open(p,'wb').write(data)
print(f'replaced {cnt} occurrences')
PY
```

> **如何确认**：修复前 `128,*(16,0d)` 出现 18 次；修复后应为 `128,*(16,0c)`。

### 5.2 修复源码中的 3 个 bug

以下修复需要改 ncnn_llm 源码（**如果仓库已包含这些修复则跳过**）：

#### (a) `inject_image_embeds` 的 memset 错误

**文件**：`src/utils/rope_embed.cpp`

```cpp
// 错误写法（memset 只填 1 字节 → token id 变成 0xF8F8F8F8 = -117901064）：
memset(token_ids_injected.data() + placeholder_index, image_pad_id, image_embeds.h * sizeof(int));

// 正确写法：
for (int i = 0; i < image_embeds.h; ++i) {
    token_ids_injected[placeholder_index + i] = image_pad_id;
}
```

#### (b) thinking template 缺少闭合标签

**文件**：`src/utils/prompt.cpp`

```cpp
// 错误写法（只输出 ，token 数少 2 个）：
if (enable_thinking) {
    prompt << "\n";
}

// 正确写法（对齐 HF processor 默认）：
if (enable_thinking) {
    prompt << "\n\n\n\n";
}
```

#### (c) `total_prompt_tokens` 计算时机错误

**文件**：`src/ncnn_llm_gpt.cpp`（多模态 prefill 函数内）

```cpp
// 错误：在 inject_image_embeds 之前计算（只含文本 token，不含图像 token）
const int total_prompt_tokens = (int)token_ids.size();   // = 22
inject_image_embeds(token_ids, token_embed, ...);         // token_ids 扩展到 1500

// 正确：移到注入之后
inject_image_embeds(token_ids, token_embed, ...);
const int total_prompt_tokens = (int)token_ids.size();   // = 1500
```

> **影响**：`row_range(total_prompt_tokens-1, 1)` 决定取哪个 token 的 hidden 喂给 lm_head。
> 修复前取 row 21（文本最后一个 token），修复后取 row 1499（序列真正的最后一个 token）。

### 5.3 修复 ncnn 子模块的 GEMM bug（如需要）

**问题**：ncnn 子模块的 `gemm_wq_int8.h` / `gemm_x86.cpp` / `gemm_bf16s.h` 在 perf 提交
（`ab44b901` "32-col multi-panel unrolling"）中破坏了 block-int8 GEMM 的 descale/布局。

**修复**：回退到上游稳定提交 `dcf9478e`，然后重打 fp32 vision TILE_N 补丁：

```bash
cd ncnn
git stash                                    # 暂存本地修改（如有）
git checkout dcf9478e -- src/layer/x86/gemm_wq_int8.h src/layer/x86/gemm_x86.cpp src/layer/x86/gemm_bf16s.h
```

然后在 `gemm_x86.cpp` 的 `gemm_BT_x86` 函数中，`get_optimal_tile_mnk` 调用之后加入：

```cpp
// constantB pre-packing calls get_optimal_tile_mnk with M=0 (M<=1 branch),
// which may select a smaller TILE_N than the forward pass (M>1 branch).
// Infer the real TILE_N from the pre-packed BT_data to avoid channel misalignment.
if (BT.w > 0 && TILE_K > 0 && BT.w % TILE_K == 0)
{
    int bttile_n = BT.w / TILE_K;
    if (bttile_n < TILE_N) TILE_N = bttile_n;
}
```

> **验证**：修复后 `proj_0` 输出 maxerr ≈ 7.6e-6（修复前 2.638）。

---

## 6. 编译 ncnn_llm

### 6.1 配置 + 编译

```bat
:: 配置（在 VS 开发者命令行中）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_LLM_ENABLE_VULKAN=OFF

:: 编译
cmake --build build --config Release --target llm_ncnn_run -j
```

或使用项目提供的 `rebuild_ocr.bat`：

```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set "PATH=C:\Users\<你的用户名>\.conda\envs\dl312\Scripts;%PATH%"
cd /d "H:\work\deploy_model\ncnn_llm\build"
ninja llm_ncnn_run
echo NINJA_EXIT=%ERRORLEVEL%
```

> **ninja 不跟踪头文件**：如果修改了 `gemm_wq_int8.h` 等头文件，必须手动删除 obj 强制重编：
> ```bat
> del build\CMakeFiles\ncnn.dir\layer\x86\gemm_x86*.obj
> ```

### 6.2 编译选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `NCNN_LLM_ENABLE_VULKAN` | ON | Vulkan 加速（**当前不可用**，见 §9） |
| `NCNN_LLM_ENABLE_TOOLS` | ON | 构建 ncnn 工具 |
| `NCNN_OPENMP` | ON | OpenMP 并行（建议开启） |

---

## 7. 运行 OCR

### 7.1 运行命令

```bat
echo Extract the text in the image. | build\llm_ncnn_run.exe ^
  --model models\Xiaomi-OCR-0_ncnn_int8 ^
  --image models\Xiaomi-OCR-0\example_pics\receipt.jpg ^
  --temperature 0 --threads 4 --enable-thinking --max-new-tokens 200
```

**参数说明**：

| 参数 | 含义 |
|---|---|
| `--model` | int8 模型目录 |
| `--image` | 输入图像路径 |
| `--temperature 0` | 贪心解码（确定性输出） |
| `--threads 4` | CPU 线程数 |
| `--enable-thinking` | **必须**（生成 `\n\n` 前缀） |
| `--max-new-tokens` | 最大生成 token 数 |

> ⚠️ **关键**：用户输入必须是 **"Extract the text in the image."**（通过 stdin 传入）。
> 空输入会触发 tool-call 乱答。

### 7.2 预期输出

```
Assistant: MR. D.I.Y. SDN BHD

(CO.REG :704427-T)

LOT 1851-A & 1853-B, JALAN KPB6,
KAWASAN PERINDUSTRIAN BALAKONG,
4390 SERI KEMBANGAN, SELANGOR (GST ID No: OOLONZBBS)
(CHERAS LEISURE MALL)-TAX INVOICE

CLOTH PEG E-JAO *S WAQ - G/GO ...
```

### 7.3 4 张测试图验证

| 图像 | 预期首行 | 状态 |
|---|---|---|
| receipt.jpg | `MR. D.I.Y. SDN BHD` | ✅ |
| qianziwen.jpeg | `行书千文` | ✅ |
| dense-equations.jpg | `We then define the following variables:` | ✅ |
| handwrite-formula.jpg | `(b) we got Friedmann equation:` | ✅ |

---

## 8. 验证与排错

### 8.1 逐层验证（推荐调试方法）

如果输出不对，用逐层对比定位：

1. **dump 中间层输出**（在 `src/ncnn_llm_gpt.cpp` 中加调试代码）
2. **HF 参考**（用 `transformers` 加载同一模型，hook 每层输出）
3. **对比**：找第一个发散的层

```python
# HF 参考示例
from transformers import Qwen3_5ForConditionalGeneration
model = Qwen3_5ForConditionalGeneration.from_pretrained('models/Xiaomi-OCR-0', torch_dtype=torch.float32).eval()
cap = {}
for i in range(24):
    model.model.language_model.layers[i].register_forward_hook(
        lambda m, inp, out, k=i: cap.__setitem__(k, out[0] if isinstance(out, tuple) else out))
with torch.no_grad():
    _ = model(input_ids)
# 对比 cap[i] 与 ncnn dump 的第 i 层输出
```

### 8.2 关键检查点

| 检查项 | 期望值 | 说明 |
|---|---|---|
| vision 特征 mean/std | -0.0034 / 0.121 | 与 HF 对齐 |
| mrope cache vs HF | 误差 < 1e-6 | 完全一致 |
| 图像 embedding 注入 | 误差 = 0 | 精确 |
| L0 hidden vs HF | 误差 < 0.001 | 修复前 0.08 |
| 首 token argmax | 17896 (`MR`) | 修复前 277 (`ar`) |

---

## 9. 常见问题

### Q1: Vulkan 加速不可用

**现象**：`--use-vulkan` 崩溃或回退 CPU。

**原因**：decoder 的 `ShortConv`（自定义层）没有 Vulkan 实现，`ncnn_llm_gpt.cpp` 会强制整网回 CPU。

**解决**：当前版本请使用 **CPU 推理**（不加 `--use-vulkan`）。GPU 加速需要实现 ShortConv 的 Vulkan kernel。

### Q2: 输出乱码 / `****`

**排查顺序**：
1. 是否用了 `--enable-thinking`？
2. 用户输入是否是 "Extract the text in the image."？
3. 模型目录是否用了 `_ncnn_int8`（bf16 目录会崩）？
4. 是否打了 §5 的所有补丁？

### Q3: 首 token 有多余前缀（如 `ar `、`竖`）

**原因**：§5.2(c) 的 `total_prompt_tokens` 修复未生效。

**验证**：首 token argmax 应为 17896（`MR`），不是 277（`ar`）。

### Q4: ninja 编译不生效（改了头文件但没重编）

**解决**：
```bat
del build\CMakeFiles\ncnn.dir\layer\x86\gemm_x86*.obj
rebuild_ocr.bat
```

### Q5: 线程数导致崩溃

**现象**：`0xC0000005` 访问冲突。

**解决**：高密度权重下用 `--threads 2` 或 `--threads 12`（避免 OpenMP/线程路径问题）。

### Q6: 输出在 ~10 行后有字符偏差

**原因**：bf16 精度累积（如 `1853-B` vs `1851-B`、`JALAN KPB6` vs `JALAN KPB 6`）。

**解决**：这是精度问题，非 bug。可尝试 `--no-bf16`（fp32 存储）验证，但差异类似。

---

## 附录：完整修复清单

| # | 文件 | 修复内容 |
|---|---|---|
| 1 | `ncnn/src/layer/x86/gemm_wq_int8.h` | 回退到 `dcf9478e` |
| 2 | `ncnn/src/layer/x86/gemm_x86.cpp` | 回退 + TILE_N 补丁 |
| 3 | `ncnn/src/layer/x86/gemm_bf16s.h` | 回退到 `dcf9478e` |
| 4 | `models/*/Xiaomi-OCR-0_decoder.ncnn.param` | `128,*(16,0d)` → `128,*(16,0c)`（18 处） |
| 5 | `src/utils/rope_embed.cpp` | `memset` → 逐 int 循环 |
| 6 | `src/utils/prompt.cpp` | `\n` → `\n\n\n\n` |
| 7 | `src/ncnn_llm_gpt.cpp` | `total_prompt_tokens` 移到注入后 |

> 详细的调试历程、逐层验证数据和根因分析见 [xiaomi-ocr-0-status.md](xiaomi-ocr-0-status.md)。
