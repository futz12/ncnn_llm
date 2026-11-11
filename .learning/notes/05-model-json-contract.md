# 05 · model.json 字段契约（支柱 B）

> 对应源码：`src/ncnn_llm_gpt.cpp`（构造函数 `:120-467`）、`src/ncnn_llm_ocr.cpp`、`src/ncnn_llm_asr.cpp`、`export/llama_export.py`、`export/qwen35_export_xiaomi_ocr_0.py`
> 状态：已讲授待实践

## 用途

读一个模型目录，说出每个字段被谁读、决定什么、缺失会怎样。识别文档与代码的漂移。

---

## 一、四段结构

```json
{
  "type": "qwen3.5",          ← ① 顶层：模型族标识
  "params": { ... },          ← ② 文件清单：6 个 param/bin 路径
  "tokenizer": { ... },       ← ③ 分词器配置
  "setting": { ... }          ← ④ 运行时行为开关
}
```

**为什么「一个目录 = 一个自描述模型」**：目录可整体拷贝分发；运行时不需要知道架构，全靠配置驱动。所以同一个 `llm_ncnn_run` 二进制能跑 Qwen3 / MiniCPM5 / Qwen3.5。

---

## 二、完整字段契约表

### `params` 段——缺一个键直接崩

| 字段 | 读取位置 |
| --- | --- |
| `embed_token_param` / `_bin` | `:203-204` |
| `decoder_param` / `_bin` | `:201-202` |
| `proj_out_param` / `_bin` | `:205-206` |

读取用 `config["params"]["embed_token_param"].get<std::string>()`。nlohmann::json 的 `operator[]` 键不存在时插入 null，再 `.get<std::string>()` 抛类型异常——**崩溃而非静默失败**。

### `tokenizer` 段

| 字段 | 读取位置 | 决定什么 | 缺失时 |
| --- | --- | --- | --- |
| `type` | `:291` | `bpe` / `bbpe`（byte-level） | 默认 `"bpe"` |
| `vocab_file` | `:294` | 词表文件 | 抛异常 |
| `merges_file` | `:295` | BPE 合并优先级 | 抛异常 |
| `eos` | `:306` | 主终止 token | `-1` |
| `eos_ids` | `:310` | **全部**终止 token（可多个） | 只用 `eos` |
| `bos` | `:316` | 句首 token，prefill 手动插入 | `-1` |
| `additional_special_tokens` | `:301` | 注册进分词器 | 抛异常 |

`eos_ids` 存在是因为 MiniCPM5 等模型有多个合法终止符（`llama_export.py:263-274` 专门处理）。

### `setting` 段——最核心

| 字段 | 读取位置 | 决定什么 | 缺失时 |
| --- | --- | --- | --- |
| `attn_cnt` | `:320` | **循环次数**：取几对 KV cache | 默认 32 |
| `sconv_cnt` | `:323` | ShortConv cache 数（Qwen3.5） | 0 |
| `gdr_cnt` | `:326` | GDR cache 数（Qwen3.5） | 0 |
| `rope.type` | `:335-347` | `RoPE`/`LongRoPE`/`NTKRoPE`/`YaRNRoPE` | 默认 `RoPE` |
| `rope.rope_head_dim` | `:332` | cos/sin 表宽度 | 默认 64 |
| `rope.rope_theta` | `:359` | RoPE 基频 | 默认 100000 |
| `rope.short_factor`/`long_factor` | `:337-338` | LongRoPE 插值 | 仅 LongRoPE 需要 |
| `rope.rope_scaling.*` | `:349-357` | NTK/YaRN 的 alpha/beta/factor/mscale | 仅对应类型需要 |
| `functions.type` | `:364` | `close` / `tool_call` | close |
| `functions.tool_call_id` | `:365` | 工具调用起始 token | 不支持 |
| `functions.tool_call_end_id` | `:369` | 工具调用结束 token | 不支持 |
| `vision.type` | `:392` | `close` / `vit` / `qwen3.5_vl` | close |
| `vision.*_param/bin` | `:401-404` | 视觉网络文件 | 仅 VLM 需要 |
| `vision.patch_size` 等 | `:440-443` | 图像切块参数 | 仅 VLM 需要 |
| `vision.rope.*` | `:445-461` | mRoPE 分区 | 仅 VLM 需要 |

**`attn_cnt` 最该记住**：决定 prefill/decode 里 `for (int i = 0; i < attn_cnt; i++)` 循环几次。若与 decoder 实际 SDPA 层数不一致，会少取 cache 或取到不存在的 blob。导出脚本 `llama_export.py:493` 的断言（SDPA 层数 == `num_layers`）正是在防这个——**导出与运行时的契约在此第二次对上**。

---

## 三、发现一：README 示例已过时

`README_CN.md:343-369` 示例用 `embed_param` / `lm_head_param` / `"model_type": "llm"`，
但 `ncnn_llm_gpt.cpp:203-206` 实际读 `embed_token_param` / `proj_out_param`。

**照 README 手写 `model.json` 会在构造函数抛异常。** 唯一权威来源是导出脚本：
- `llama_export.py:517-524`
- `qwen35_export_xiaomi_ocr_0.py:501-508`

> 判断标准是「代码读什么」，不是「文档写什么」。

---

## 四、发现二：两套命名约定并存

| | LLM / GPT 族 | OCR / ASR 族 |
| --- | --- | --- |
| 类型字段 | `type` | `model_type` |
| 输出投影 | `proj_out_param` | `lm_head_param` |
| 嵌入 | `embed_token_param` | `text_embed_param` |
| 解码器 | `decoder_param` | `text_decoder_param` |
| 读取方 | `ncnn_llm_gpt.cpp` | `ncnn_llm_ocr.cpp:44-51`、`ncnn_llm_asr.cpp:60-64` |
| 写出方 | `llama_export.py`、`qwen35_export_xiaomi_ocr_0.py` | `laya_export.py:310`、`extract_tokenizer.py:39` |

**实践含义**：不能把 OCR 模型目录丢给 `llm_ncnn_run`。选错入口的报错是「键不存在」而非「模型类型不符」，所以**先看入口程序，再猜目录格式**。

---

## 五、发现三：两个「不在 json 里」的能力

```cpp
auto it_think = bpe->token_to_id().find("<think>");      // :377
auto it_think_end = bpe->token_to_id().find("</think>");  // :382
auto it = bpe->token_to_id().find("<|image_pad|>");        // :434
```

词表里找不到就**静默不启用**（`think_id` 保持 -1）。对比 `params` 缺键抛异常——**同一配置文件里，有的是硬契约，有的是软约定**。

---

## 六、顺带一个死字段

`ncnn_llm_gpt.cpp:1147` 用成员 `model_type`，但该成员（`ncnn_llm_gpt.h:144`）在整个 `.cpp` 里**从未赋值**。`:173` 的 `const std::string model_type` 是 `#if defined(__riscv)` 块内的**局部变量**，遮蔽了成员。

结果：性能报告的 model_type 永远是空，`perf_profiler.cpp:130` 的 `if (!meta.model_type.empty())` 永不成立。不影响功能，但说明「报告字段」与「实际数据」可能脱节。

---

## 待办练习

1. 删掉 Qwen3.5 模型 `model.json` 里的 `"sconv_cnt": 4`，预测后果。
   - 要点：不崩。`create_ctx()`（`:97-107`）因 `sconv_cnt == 0` 而造 `ncnn_llm_gpt_base_ctx` 而非 `qwen3_5_ctx`，`dynamic_pointer_cast` 失败，sconv/gdr cache 完全不被维护。Qwen3.5 是混合注意力，cache 丢了输出必然错乱。验证点 `:99`。
2. 从 `export/qwen35_export_xiaomi_ocr_0.py:499-525` 反推完整 `model.json`。
   - 要点：`"type": "qwen3.5"` 硬编码；`params` 用 `os.path.basename()` 只存文件名，因为运行时会用 `model_path + "/"` 拼（`:201`）；`attn_cnt`/`sconv_cnt`/`gdr_cnt` 三个计数必须一起写。
