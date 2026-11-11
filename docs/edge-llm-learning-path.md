# 边缘设备大模型部署学习路线

> 以 ncnn_llm 项目（Xiaomi-OCR-0 部署）为锚点，从零到能独立部署/优化边缘大模型。
> 每个阶段都对应项目中的真实文件和技术，学完能直接上手改这个项目。

---

## 阶段 0：基础准备（1-2 周）

### 目标
能读懂模型导出脚本和推理代码，不迷路。

### 学习内容

| 主题 | 要掌握 | 项目对应 |
|---|---|---|
| **C++ 基础** | 类/模板/智能指针/RAII | `src/ncnn_llm_gpt.cpp` 全部 |
| **Python 基础** | numpy 数组操作、torch 张量 | `export/qwen35_export_xiaomi_ocr_0.py` |
| **Transformer 架构** | Attention、MLP、LayerNorm/RMSNorm、RoPE | `docs/xiaomi-ocr-0-status.md` §5 |
| **HuggingFace 生态** | `transformers` 加载模型、`AutoProcessor`、`generate()` | `temp_hf_*.py` 系列脚本 |

### 实践
- 用 `transformers` 跑通 Xiaomi-OCR-0 的 HF 推理（作为"参考答案"）
- 读懂 `export/qwen35_export_xiaomi_ocr_0.py` 的导出流程（fp32 → bf16/int8）

---

## 阶段 1：理解推理框架（2-3 周）

### 目标
理解 ncnn 的核心抽象，能读懂 `ncnn::Mat`、`Extractor`、`Net`。

### 学习内容

| 主题 | 要掌握 | 项目对应 |
|---|---|---|
| **ncnn::Mat 内存布局** | (w,h,c) 维度、elemsize、cstep、数据排列 | `ncnn/src/mat.h` |
| **Extractor 机制** | 惰性计算、blob 名字索引、中间结果提取 | `src/ncnn_llm_gpt.cpp` 的 `ex.input/ex.extract` |
| **Net 与 Layer** | 参数加载、前向传播、pipeline 创建 | `ncnn/src/net.cpp` |
| **自定义算子** | 如何注册和实现一个 Layer | `src/kernel/shortconv.h`、`gdr.h` |
| **模型格式** | `.ncnn.param`（计算图）+ `.ncnn.bin`（权重） | `models/Xiaomi-OCR-0_ncnn_int8/*.param` |

### 实践
- 用 `ncnn` 的 `ncnn2mem` 工具把模型转成 C 数组，理解二进制格式
- 写一个最小的自定义 Layer（如 `y = x * 2`）并跑通

---

## 阶段 2：模型导出与转换（2-3 周）

### 目标
能把 HF 模型转成 ncnn 格式，理解每个环节在做什么。

### 学习内容

| 主题 | 要掌握 | 项目对应 |
|---|---|---|
| **pnnx 转换** | PyTorch → pnnx → ncnn 的流程 | `export/qwen35_export_xiaomi_ocr_0.py` 中的 `pnnx.export` |
| **权重精度转换** | fp32 → bf16 的截断/舍入规则 | `export/bf16_convert.py` |
| **KV-cache 改写** | 把普通 Attention 改成带 cache 的增量解码 | `export/add_kvcache.py` |
| **混合架构导出** | SDPA 层 + linear_attn 层的不同处理 | `export/qwen35_gdr.py`（GatedDeltaRule） |
| **Vision tower 导出** | patch_embed/pos_embed/encoder/merger 拆分 | `export/qwen35_vision_export.py` |
| **Tokenizer 提取** | BPE/Unigram 词表导出 | `export/extract_tokenizer.py` |

### 实践
- 完整跑一遍 `qwen35_export_xiaomi_ocr_0.py`，观察输出结构
- 用 `netron` 可视化 `.ncnn.param` 计算图
- 对比导出前后的权重（fp32 vs bf16 vs int8）

---

## 阶段 3：量化技术（3-4 周，核心）

### 目标
理解边缘部署的核心——权重量化，能自己实现/调试量化流程。

### 学习内容

| 主题 | 要掌握 | 项目对应 |
|---|---|---|
| **量化基础** | 对称/非对称量化、scale/zero-point、per-tensor vs per-channel | `export/quantize_model.py` |
| **Block 量化** | block=64 的分组量化、minmax 标定 | `--block 64 --method minmax` |
| **int8 反量化 GEMM** | 量化权重 → 反量化 → 矩阵乘的硬件实现 | `ncnn/src/layer/x86/gemm_wq_int8.h` |
| **bf16 存储** | bf16 格式、与 fp16 的区别、存储 vs 计算 | `decoder_net->opt.use_bf16_storage` |
| **量化误差分析** | 量化对精度的影响、哪些层敏感 | 对比 fp32/bf16/int8 输出 |
| **量化感知训练 vs 训练后量化** | PTQ（本项目） vs QAT | 概念性了解 |

### 实践
- 用 `quantize_model.py` 生成 int4/int6/int8 三档，对比精度和大小
- 读 `gemm_wq_int8.h`，理解 block-int8 GEMM 的 descale 逻辑
- **复现 bug**：把 `gemm_wq_int8.h` 换成 perf 版本，观察乱码，再理解为什么

---

## 阶段 4：推理优化（3-4 周，核心）

### 目标
理解边缘设备推理的关键优化手段。

### 学习内容

| 主题 | 要掌握 | 项目对应 |
|---|---|---|
| **KV-cache 机制** | 为什么需要、如何增量更新、内存布局 | `src/ncnn_llm_gpt.h` 的 `KVCache` |
| **GEMV 优化** | M=1 时的向量-矩阵乘优化、AVX2 指令 | `src/kernel/x86/lm_head_x86.cpp` |
| **GEMM tile 策略** | TILE_M/N/K 选择、预打包权重（constantB） | `ncnn/src/layer/x86/gemm_x86.cpp` |
| **算子融合** | 减少内存访问、合并小算子 | 概念性了解 |
| **内存管理** | Allocator、内存池、KV-cache 复用 | `ncnn_llm_gpt.h` 的 `UnlockedPoolAllocator` |
| **多线程** | OpenMP 并行、线程数对性能的影响 | `--threads` 参数 |
| **推理流程** | prefill（并行）vs decode（串行）两阶段 | `prefill()` vs `generate()` |

### 实践
- 用 `--perf --perf-level 2` 跑性能分析，看每层耗时
- 对比不同 `--threads` 的吞吐
- 读 `lm_head_x86.cpp`，理解 AVX2 向量化 GEMV

---

## 阶段 5：硬件后端与部署（2-3 周）

### 目标
理解边缘设备的多样性，能为不同硬件选型/适配。

### 学习内容

| 主题 | 要掌握 | 项目对应 |
|---|---|---|
| **CPU 后端** | x86 AVX2/AVX-512、ARM NEON | `ncnn/src/layer/x86/`、`ncnn/src/layer/arm/` |
| **GPU 后端** | Vulkan compute shader | `--use-vulkan`（当前受限） |
| **NPU/DSP** | 高通 QNN、联发科 APU、华为 Ascend | 概念性了解（ncnn 生态） |
| **内存约束** | 边缘设备内存小 → 量化 + 流式加载 | int8 模型 ~1.7GB vs fp32 ~6.8GB |
| **功耗/散热** | 持续推理的降频问题 | 概念性了解 |
| **跨平台** | Windows/Linux/Android 编译 | CMake 配置 |

### 实践
- 在 WSL/Linux 上编译运行同一项目
- 对比 x86 CPU 和 Vulkan GPU 的性能（如果硬件允许）
- 了解 ncnn 官方支持的硬件后端矩阵

---

## 阶段 6：端到端项目实战（2-4 周）

### 目标
独立完成一个边缘大模型部署项目。

### 可选项目（由易到难）

| 项目 | 涉及技术 | 难度 |
|---|---|---|
| **部署 Qwen3-0.6B 文本聊天** | 导出 + int8 量化 + CPU 推理 | ⭐ |
| **部署 Qwen2.5-VL 图文** | 加 vision tower + 图像预处理 | ⭐⭐ |
| **部署 GLM-OCR** | OCR 专用 pipeline | ⭐⭐ |
| **复现本项目 Xiaomi-OCR-0** | 全流程 + bug 修复 | ⭐⭐⭐ |
| **移植到其他模型** | 自定义导出脚本 | ⭐⭐⭐⭐ |
| **实现一个自定义算子** | 如新的 attention 变体 | ⭐⭐⭐⭐ |

### 实践建议
- 先复现本项目（有教程和 ground truth）
- 再换一个模型（如 Qwen3-0.6B）独立完成
- 最后尝试优化（量化/算子/内存）

---

## 学习资源

| 类型 | 资源 | 用途 |
|---|---|---|
| **ncnn 官方** | https://github.com/Tencent/ncnn | 框架源码 |
| **ncnn_llm** | https://github.com/HougeLangley/ncnn_llm | 本项目 |
| **HuggingFace** | https://huggingface.co/docs/transformers | 模型参考实现 |
| **量化理论** | https://github.com/Tencent/ncnn/tree/master/tools/quantize | ncnn 量化工具 |
| **边缘部署综述** | 搜索 "LLM edge deployment survey" | 全局视野 |
| **RoPE/Attention** | 《Attention Is All You Need》+ RoPE 原论文 | 理论基础 |
| **量化论文** | GPTQ、AWQ、SmoothQuant | 进阶量化 |

---

## 学习锚点：本项目文件速查

| 想了解 | 看哪里 |
|---|---|
| 模型怎么导出的 | `export/qwen35_export_xiaomi_ocr_0.py` |
| 量化怎么做的 | `export/quantize_model.py` + `ncnn/src/layer/x86/gemm_wq_int8.h` |
| 推理主流程 | `src/ncnn_llm_gpt.cpp` 的 `prefill()` / `generate()` |
| 自定义算子 | `src/kernel/shortconv.h` + `gdr.h` |
| lm_head GEMV 优化 | `src/kernel/x86/lm_head_x86.cpp` |
| KV-cache 管理 | `src/ncnn_llm_gpt.h` |
| 多模态处理 | `src/ncnn_llm_gpt.cpp` 的 `prefill(text, image, ctx)` |
| 踩坑记录 | `docs/xiaomi-ocr-0-status.md` |
| 部署教程 | `docs/xiaomi-ocr-deploy-tutorial.md` |

---

## 建议的学习顺序

```
阶段0 基础 → 阶段1 框架 → 阶段2 导出 → 阶段3 量化 → 阶段4 优化 → 阶段5 硬件 → 阶段6 实战
                    ↑                                        ↓
                    └──────── 边学边做本项目，每个阶段都回来改代码 ────────┘
```

**核心原则**：不要只看理论，每个阶段都要动手改这个项目的代码，跑一遍看效果。
