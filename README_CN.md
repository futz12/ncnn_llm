<p align="center">
  <img src="logo.png" alt="ncnn_llm" width="220">
</p>

<h1 align="center">ncnn_llm</h1>

<p align="center">
  <b>基于 ncnn 的 LLM、VLM、OCR、判别器和嵌入模型推理运行时。</b>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="Build" src="https://img.shields.io/badge/build-cmake-064f8c">
  <img alt="Backend" src="https://img.shields.io/badge/backend-ncnn-orange">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Android-lightgrey">
</p>

<p align="center">
  <a href="readme.md">English</a>
  ·
  <a href="#快速开始">快速开始</a>
  ·
  <a href="#支持模型">支持模型</a>
  ·
  <a href="#模型库">模型库</a>
</p>

---

`ncnn_llm` 提供一个轻量级 C++ 运行时，用于在 [ncnn](https://github.com/Tencent/ncnn) 上运行语言模型和嵌入模型。项目关注可落地的本地推理场景，包括边缘设备、桌面 CPU 和支持 Vulkan 的 GPU。

本项目源自 **nihui** 对 ncnn `kvcache` 的实验性工作，并在此基础上扩展出可复用的示例、模型加载、分词器、视觉预处理、OCR 推理和嵌入模型 API。

## 特性

- 统一的聊天和视觉语言模型 CLI 示例
- 支持 KV cache 的自回归解码，可运行在 CPU 或 Vulkan 后端
- 支持 Qwen / MiniCPM 风格的 LLM
- 支持 Qwen VL 图文输入
- 提供 GLM-OCR 图像文字识别示例
- 提供 Laya / Laya-Multilingual 判别器快速决策示例
- 提供文本嵌入和多模态嵌入 API
- 支持 BPE 和 Unigram 分词器
- 基于 CMake 统一构建与工程管理，内置丰富示例程序与 CTest 自动化测试
- 高性能自定义算子（GatedDeltaRule 与 ShortConv）遵循 ncnn 正式算子架构，统一依托 workspace_allocator / blob_allocator 内存池管理临时与输出内存，并通过 AVX-512 / AVX / SSE SIMD 高效加速

## 支持模型

| 类别 | 模型 | 状态 | 说明 |
| --- | --- | --- | --- |
| LLM | YoutuLLM | 已支持 | 聊天 / 文本生成 |
| LLM | MiniCPM4 | 已支持 | 聊天 / 文本生成 |
| LLM | MiniCPM5 | 已支持 | 聊天、思考与 XML 工具调用 |
| LLM | Qwen3 | 已支持 | 聊天 / 文本生成 |
| LLM | Qwen3.5 | 已支持 | GatedDeltaRule 与 ShortConv 混合注意力 |
| VLM | Qwen2.5-VL | 已支持 | 图像 + 文本输入 |
| VLM | Qwen3.5-VL | 已支持 | 图像 + 文本输入与 mRoPE |
| OCR | GLM-OCR | 已支持 | OCR |
| OCR | HunyuanOCR | 已支持 | OCR |
| ASR | Qwen3 ASR | 已支持 | ASR |
| 判别器 | Laya / Laya-Multilingual | 已支持 | System 1 极速决策引擎 (意图分类/严重度打分/强化学习升级) |
| 嵌入 | Jina-Embeddings-v5-Text-Nano | 已支持 | 768 维文本嵌入 |
| 嵌入 | Jina-CLIP-v2 | 已支持 | 1024 维文本 + 图像嵌入 |

## 快速开始

### 1. 环境依赖

- 支持 C++20 的编译器 (MSVC 2019+, GCC 10+, Clang 11+)
- CMake >= 3.15
- Vulkan SDK（可选，用于 Vulkan GPU 加速）

### 2. 克隆仓库

递归克隆项目仓库（参考 LiteOCR、wan-ncnn-vulkan 等项目，内嵌 ncnn 官方算子支持）：

```bash
git clone --recursive https://github.com/futz12/ncnn_llm.git
cd ncnn_llm
```

如果克隆时未加 `--recursive`，请执行以下命令拉取子模块：

```bash
git submodule update --init --recursive
```

### 3. 构建

```bash
# 配置工程（可选选项：-DNCNN_LLM_ENABLE_VULKAN=ON/OFF、-DNCNN_LLM_ENABLE_TOOLS=ON/OFF 构建 ncnn 工具）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_LLM_ENABLE_VULKAN=ON

# 编译
cmake --build build --config Release -j

# 运行单元测试
ctest --test-dir build -C Release --output-on-failure
```

构建单个目标（例如 `llm_ncnn_run`）：

```bash
cmake --build build --config Release --target llm_ncnn_run
```

### 4. 下载模型

从下面的镜像下载已转换的 ncnn 模型目录：

https://mirrors.sdu.edu.cn/ncnn_modelzoo/

将模型目录放到 `assets/` 下，例如：

```text
assets/
└── qwen3_0.6b/
    ├── model.json
    ├── *.ncnn.param
    ├── *.ncnn.bin
    └── tokenizer files
```

## CLI 聊天

`llm_ncnn_run` 是主要的交互式示例，支持文本模型和视觉语言模型。

```bash
./build/llm_ncnn_run --model ./assets/qwen3_0.6b
```

指定运行参数：

```bash
./build/llm_ncnn_run --model ./assets/qwen3_0.6b --threads 4
./build/llm_ncnn_run --model ./assets/qwen3_0.6b --vulkan --vulkan-device 0
```

视觉语言输入：

```bash
./build/llm_ncnn_run --model ./assets/qwen2.5_vl_3b --image ./assets/test.jpg
```

### CLI 选项

| 选项 | 说明 |
| --- | --- |
| `--model <path>` | 模型目录（默认: `./assets/qwen3_0.6b`） |
| `--threads <num>` | CPU 线程数（默认: auto） |
| `--use-vulkan` | 启用 Vulkan GPU 计算（同时支持 `--vulkan`） |
| `--vulkan-device <index>` | Vulkan 设备编号（默认: `0`） |
| `--image <path>` | VLM 视觉语言模型输入图像路径 |
| `--max-new-tokens <num>` | 最大生成 token 数量（默认: `512`） |
| `--enable-thinking` | 开启模型推理思考过程（输出 `<think>...</think>`） |
| `--no-builtin-tools` | 禁用内置演示工具（计算器/随机数） |

示例会话：

```text
llm_ncnn_run (cli). Type 'exit' or 'quit' to end the conversation.
User: Hello
Assistant: Hello! How can I help you today?
```

## 模型量化 (INT8 Block Quantization)

`ncnn_llm` 支持基于 ncnn 最新 Gemm block quant 特性的 INT8 量化推理。通过量化 Decoder 权重（保持 LM Head / Proj Out 原始精度以确保生成质量），大幅减少内存占用并提升 CPU 解码吞吐量。

以 Qwen3-0.6B 为例（Intel i9-13900HX）：
- **体积压缩**：Decoder 权重从 840 MB 降低至 446 MB（减少约 47%）。
- **推理提速**：CPU 解码速度从 17.0 tokens/s 提升至 32.0 tokens/s（提升 71.9%），Prefill 耗时缩短 42%。
- **生成精度**：与原模型输出完全一致，无精度退化。

### 导出量化模型

使用 ncnn 自带的量化工具进行量化（脚本会自动寻找编译好的 `ncnnllm2int` 工具）：

```bash
# 自动量化单个模型目录并生成 model.json 与资产文件
python export/quantize_model.py --model ./assets/qwen3_0.6b --output ./assets/qwen3_0.6b_int8 --bits 8 --block 64

# 或一键批量量化 assets 目录下的所有模型
python export/quantize_model.py --all --bits 8 --block 64

# 或直接对指定 param / bin 文件量化
ncnnllm2int decoder.ncnn.param decoder.ncnn.bin decoder_int8.ncnn.param decoder_int8.ncnn.bin bits=8 block=64 method=minmax
```

### 运行量化模型

```bash
./build/llm_ncnn_run --model ./assets/qwen3_0.6b_int8 --threads 8
```
> 注：Gemm weight block quantization 当前在 CPU 后端提供高效向量化计算（支持 AVX2 / AVX-VNNI / ARM 等），运行时检测到量化层时会自动在 CPU 执行。

## OCR

GLM-OCR 使用专用的图像 prefill 路径，并复用共享文本解码运行时。

```bash
cmake --build build --config Release --target ocr_main
./build/ocr_main --model ./assets/glm_ocr --image ./test_ocr.png --prompt "Read the text in the image."
```

输出示例：

```text
Generating text:
Hello World 123
```

### Xiaomi-OCR-0 部署案例

Xiaomi-OCR-0（Qwen3.5-VL 混合架构：6 层 SDPA + 18 层 GatedDeltaRule）在 ncnn_llm 上的完整部署案例，
包含模型导出、bug 修复、int8/fp32 量化推理全流程。

```bash
# 1. 导出（fp32 → bf16 + int8）
python export/qwen35_export_xiaomi_ocr_0.py \
    -m models/Xiaomi-OCR-0 \
    -o models/Xiaomi-OCR-0_ncnn \
    --int8-dir models/Xiaomi-OCR-0_ncnn_int8 --bits 8

# 2. 运行 OCR（int8 模型）
echo "Extract the text in the image." | ./build/llm_ncnn_run.exe \
    --model models/Xiaomi-OCR-0_ncnn_int8 \
    --image models/Xiaomi-OCR-0/example_pics/receipt.jpg \
    --temperature 0 --threads 4 --enable-thinking --max-new-tokens 200
```

输出示例（receipt.jpg）：

```text
MR. D.I.Y. SDN BHD
(CO.REG :704427-T)
LOT 1851-A & 1853-B, JALAN KPB6,
KAWASAN PERINDUSTRIAN BALAKONG,
(CHERAS LEISURE MALL)-TAX INVOICE-
```

**案例要点**：
- 导出脚本 `export/qwen35_export_xiaomi_ocr_0.py` 自动修复 pnnx 生成的 RMSNormGated reshape 表达式
- 支持 int8（边缘部署，1.2GB）和 fp32（调试验证，6.4GB）两种精度
- 4 张测试图（receipt/qianziwen/dense-equations/handwrite-formula）全部验证通过
- 详细教程见 [docs/xiaomi-ocr-deploy-tutorial.md](docs/xiaomi-ocr-deploy-tutorial.md)
- 调试历程与根因分析见 [docs/xiaomi-ocr-0-status.md](docs/xiaomi-ocr-0-status.md)

## 判别器 / 快速决策 (Laya)

`ncnn_llm_laya` 支持 Convai 的 **Laya**（基于 ModernBERT-large）与 **Laya-Multilingual**（基于 mmBERT-base）判别器。每个模型由 `backbone`、`scorer` 和 `act_head` 三个 ncnn 子模型组成，用于 System 1 级别的意图识别、严重度打分与强化学习代理升级判断。

### 导出模型

```bash
# 先将 Hugging Face 源模型下载到本地目录
huggingface-cli download convaiinnovations/laya --local-dir ./models/laya

# 导出英文版 Laya (BF16 & INT8 block quantization)
python export/laya_export.py --model-dir ./models/laya --output-dir ./assets/laya --int8-dir ./assets/laya_int8

# 下载并导出多语言版 Laya-Multilingual
huggingface-cli download convaiinnovations/laya-multilingual --local-dir ./models/laya_multilingual
python export/laya_export.py --model-dir ./models/laya_multilingual --output-dir ./assets/laya_multilingual --int8-dir ./assets/laya_multilingual_int8
```

`--model-dir` 必须是已经下载的本地源模型目录；导出脚本会在该目录中加载 `rl_agent_api.py` 和 `tokenizer/`。

### CLI 推理

```bash
cmake --build build --config Release --target laya_main

# 运行 INT8 量化多语言判别器
./build/laya_main --model ./assets/laya_multilingual_int8 --json ./examples/laya_multilingual_request.json --threads 4
```

### C++ API

```cpp
#include "ncnn_llm_laya.h"

ncnn_llm_laya laya("./assets/laya_multilingual_int8", false, 4, 0, true);

std::string state = "包裹在运输途中破损了，里面的东西碎了一地，快递员还不承认，要求立刻赔偿退款并道歉！";
nlohmann::json questions = {
    {"intent", {
        {"type", "choice"},
        {"instructions", "判断用户的核心意图类别"},
        {"criteria", {
            {"refund", "要求退款或赔偿"},
            {"logistics", "询问物流进度或包裹位置"},
            {"feedback", "产品普通反馈或建议"}
        }}
    }},
    {"urgent", {
        {"type", "noul"},
        {"instructions", "用户的情绪是否非常愤怒且需要紧急处理？"}
    }}
};

nlohmann::json res = laya.system_one_json(state, questions);
std::cout << res.dump(2) << std::endl;
```

## 嵌入模型

`ncnn_embedding` 为文本嵌入和 CLIP 风格的图文嵌入提供统一 API。

### 文本嵌入

```bash
cmake --build build --config Release --target embedding_main
./build/embedding_main --model ./assets/jina-embeddings-v5-text-nano
```

### CLIP 多模态嵌入

```bash
cmake --build build --config Release --target clip_main
./build/clip_main --model ./assets/jina_clip_v2 --image ./assets/ganyu.jpg
```

### C++ API

```cpp
#include "ncnn_embedding.h"

ncnn_embedding embed("./assets/jina_clip_v2", false, 4);

std::vector<float> text_vec = embed.encode_text("Hello world");

if (embed.supports_image()) {
    std::vector<float> image_vec = embed.encode_image_file("./image.jpg");
    float score = cosine_similarity(text_vec, image_vec);
}
```

## 其他示例

| Target | 用途 |
| --- | --- |
| `llm_ncnn_run` | 统一聊天 / VLM CLI |
| `ocr_main` | GLM-OCR 推理 |
| `embedding_main` | 文本嵌入推理 |
| `clip_main` | CLIP 图文嵌入推理 |
| `laya_main` | Laya / Laya-Multilingual 判别器推理 |
| `benchllm` | LLM 端到端精确 Prefill & Decode 性能测试 |
| `bench_qwen35` | Qwen3.5 线性注意力（GDR 与 ShortConv）性能测试 |
| `test_llm` | 单元测试 |
| `test_bf16` | BF16 精度测试 |
| `test_kernel` | GDR 与 ShortConv 算子内存池及 SIMD 优化单元测试 |

运行单元测试：

```bash
ctest --test-dir build -C Release --output-on-failure
```

运行 benchmark：

```bash
cmake --build build --config Release --target benchllm
./build/benchllm [loop_count] [threads] [powersave] [gpu_device] [cooling_down] [pp] [tg]
```

## 模型库

已转换的 ncnn 模型权重可从下面的镜像下载：

https://mirrors.sdu.edu.cn/ncnn_modelzoo/

每个模型目录应包含 `model.json`、ncnn param/bin 文件和分词器文件。可以将模型目录放在 `assets/` 下，也可以通过 `--model` 直接传入路径。

## 配置

每个模型目录都由 `model.json` 描述。不同模型族的字段会有差异，一个典型文本模型配置如下：

```json
{
  "model_type": "llm",
  "params": {
    "embed_param": "embed.ncnn.param",
    "embed_bin": "embed.ncnn.bin",
    "decoder_param": "decoder.ncnn.param",
    "decoder_bin": "decoder.ncnn.bin",
    "lm_head_param": "lm_head.ncnn.param",
    "lm_head_bin": "lm_head.ncnn.bin"
  },
  "tokenizer": {
    "type": "bbpe",
    "vocab_file": "vocab.txt",
    "merges_file": "merges.txt"
  },
  "setting": {
    "attn_cnt": 32,
    "hidden_size": 1024,
    "rope": {
      "type": "RoPE",
      "rope_head_dim": 64,
      "rope_theta": 1000000.0
    }
  }
}
```

嵌入模型和 OCR 模型使用各自的 `model_type` 和参数区段。具体写法可以参考 `assets/` 下的模型文件。

## 项目结构

```text
ncnn_llm/
├── assets/                 # 本地模型目录和演示资源
├── benchmark/              # Benchmark 入口
├── examples/               # CLI 和功能示例
│   ├── llm_ncnn_run/       # 统一聊天 / VLM 运行器
│   ├── ocr_main.cpp        # OCR 示例
│   ├── embedding_main.cpp  # 文本嵌入示例
│   ├── clip_main.cpp       # CLIP 示例
│   ├── laya_main.cpp       # Laya 判别器示例
│   └── asr_main.cpp        # ASR 示例
├── export/                 # 导出脚本
├── src/                    # 核心运行时
│   ├── ncnn_llm_gpt.*      # LLM / VLM 运行时
│   ├── ncnn_llm_laya.*     # Laya 判别器运行时
│   ├── ncnn_llm_ocr.*      # OCR 图像 prefill + 共享解码
│   ├── ncnn_embedding.*    # 嵌入模型运行时
│   ├── ncnn_text_runtime.* # 共享文本解码辅助函数
│   └── utils/              # 分词器、图像、RoPE、prompt 工具
├── tests/                  # 单元测试
```

## 路线图

- 保持 decoder 和 KV cache 运行时在不同模型族之间共享
- 扩展更多模型架构和分词器支持
- 提升 Vulkan 和 CPU 推理性能
- [x] 增加 INT8 量化支持 (Gemm Block Quantization)
- 更完整地文档化模型导出流程

随着运行时演进，旧导出脚本可能会过时。建议优先参考最新模型示例和 `model.json` 文件。

## 社区

欢迎提交 issue、修复、转换模型和测试结果。

- QQ 群：`767178345`

## 许可证

Apache License 2.0。详见 [LICENSE](LICENSE)。
