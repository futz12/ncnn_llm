# 03 · 完整 API 调用链（从加载模型到输出答案）

> 对应源码：`examples/llm_ncnn_run/main.cpp`、`src/ncnn_llm_gpt.cpp`、`src/ncnn_text_runtime.cpp`、`src/kernel/lm_head.h`
> 状态：已讲授待实践

## 用途

指着任意一行代码，能说出它属于哪一层、在链条的哪个位置、出错时表现为什么现象。

---

## 一、总览路线图

```
                        ┌──────────────────────────────┐
                        │  main()   main.cpp:26        │
                        └──────────────┬───────────────┘
                                       │
      ┌────────────────────────────────┼──────────────────────────────┐
      │                    │                       │                  │
 get_utf8_args()     parse_options()      detect_template_type()  load_image_to_ncnn_mat()
 (Windows 中文)      options.cpp          cli_runner.cpp:9        (仅 VLM)
      └────────────────────────────────┴──────────────────────────────┘
                                       │
                    normalize_model_path()   ← 相对路径补 ./assets/
                                       │
                                       ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  阶段 1｜模型加载                                                        ║
║  ncnn_llm_gpt 构造函数          ncnn_llm_gpt.cpp:120                      ║
║                                                                          ║
║   ├─ 读 model.json                              :125                     ║
║   ├─ ncnn::get_physical_cpu_count()             :132  决定线程数          ║
║   ├─ make_shared<ncnn::Net>() ×3                :141  decoder/embed/head  ║
║   ├─ opt.num_threads / use_vulkan_compute       :147                     ║
║   │  / use_bf16_storage / vulkan_device_index                             ║
║   ├─ register_gdr_layers()                      :230  注册自定义算子       ║
║   ├─ decoder_net->load_param() → load_model()   :232                     ║
║   ├─ layers() 遍历                              :238  Vulkan 兼容性兜底    ║
║   ├─ lm_head->init_shared() / init_from_file()  :268  tied 权重零拷贝      ║
║   ├─ BpeTokenizer::LoadFromFiles()              :297                     ║
║   ├─ 读 eos / bos / eos_ids                     :306                     ║
║   ├─ 读 attn_cnt / rope / tool_call_id          :320                     ║
║   └─ 读 vision 配置 → load_net() ×N             :388  (仅 VLM)            ║
╚══════════════════════════════════════════════╦═══════════════════════════╝
                                               ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  阶段 2｜首轮 prefill                                                    ║
║                                                                          ║
║  apply_chat_template()            prompt.cpp   裸文本 → 带模板的 prompt     ║
║        ↓                                                                 ║
║  prefill(text)                    ncnn_llm_gpt.cpp:469                    ║
║    ├─ bpe->encode()              :478   文本 → token id                  ║
║    ├─ generate_rope_embed_cache():499   id 长度 → cos/sin 表              ║
║    ├─ embed_net->create_extractor()      :507                            ║
║    │    input("in0") → extract("out0")   :508                            ║
║    ├─ 构造 causal mask             :512   L×L，上三角 -1e38               ║
║    ├─ decoder_net->create_extractor()    :533                            ║
║    │    set_kvcache_allocator()          :535                            ║
║    │    set_kvcache_max_seqlen_hint()    :536                            ║
║    │    input("in0") 嵌入                :538                            ║
║    │    input("in1") mask                :539                            ║
║    │    input("in2"/"in3") cos/sin       :540                            ║
║    │    extract("out_cache_k{i}")        :548  ← 只取，不喂               ║
║    │    extract("out_cache_v{i}")        :549                            ║
║    │    extract("out0")                  :572                            ║
║    ├─ lm_head->forward()            :578   隐状态 → logits                ║
║    ├─ argmax                        :585   第一个 token                  ║
║    └─ 写入 ctx{kv_cache, cur_token, position_id}  :594                   ║
╚══════════════════════════════════════════════╦═══════════════════════════╝
                                               ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  阶段 3｜generate 循环              ncnn_llm_gpt.cpp:962                   ║
║                                                                          ║
║  clone_ctx(ctx)                    :962   复制状态，不动原 ctx            ║
║        │                                                                 ║
║        ▼   ◄────────────────────────── 最多 max_new_tokens 轮             ║
║  ┌───────────────────────────────────────────────────────────────┐       ║
║  │ eos_ids.count(cur_token) ?  break            :973             │       ║
║  │ tool_call_id / think_id 状态机                :975             │       ║
║  │ bpe->decode({cur_token})   id → 文本          :1002            │       ║
║  │ callback(text)             → 流式吐给终端      :1006            │       ║
║  │ llm_run_text_embed()       查嵌入              :1014            │       ║
║  │ generate_rope_embed_cache(1, ...)             :1028            │       ║
║  │ mask = 全 0 一列             :1034                            │       ║
║  │ llm_run_decoder_with_kv()    :1045                            │       ║
║  │    input("cache_k{i}")      ← 喂旧 cache      :1064            │       ║
║  │    extract("out_cache_k{i}")→ 取新 cache      :1086            │       ║
║  │ lm_head->forward()           :1113                            │       ║
║  │ llm_select_next_token()      采样              :1127            │       ║
║  │ ctx->cur_token = next_id     :1130                            │       ║
║  └───────────────────────────────────────────────────────────────┘       ║
║        │                                                                 ║
║        ▼                                                                 ║
║  report.print()   性能报告        :1165                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │  std::cout       │
                                    │  Assistant: ...  │
                                    └──────────────────┘
```

---

## 二、四阶段详解

### 阶段 0｜启动与参数（`main.cpp:26-57`）

| API | 位置 | 作用 |
| --- | --- | --- |
| `enable_utf8_console()` | `:28` | Windows 控制台切 UTF-8，否则中文乱码 |
| `get_utf8_args()` | `:29` | 把 argv 转 UTF-8（Windows 特有） |
| `parse_options()` | `:34` | 解析 `--model` / `--threads` / `--use-vulkan` 等 |
| `normalize_model_path()` | `:35` `:15` | 相对路径且无父目录时补 `./assets/` 前缀 |
| `detect_template_type()` | `:42` `cli_runner.cpp:9` | 读 `model.json` 的 `type` 字段选对话模板 |
| `load_image_to_ncnn_mat()` | `:50` | 仅 VLM，读图成 `ncnn::Mat` |

### 阶段 1｜模型加载（构造函数 `:120-467`）

| API | 位置 | 作用 |
| --- | --- | --- |
| 读 `model.json` | `:125` | **唯一一次读配置文件** |
| `ncnn::get_physical_cpu_count()` | `:132` | 线程数默认值，失败退 `get_cpu_count()`，再退 4 |
| `make_shared<ncnn::Net>()` ×3 | `:141` | decoder / embed / proj_out 三个独立网络 |
| `opt.num_threads` 等 | `:147` | 全局选项，**每个 Net 单独设** |
| `register_gdr_layers()` | `:230` | 注册 GDR 自定义算子（Qwen3.5） |
| `decoder_net->load_param()` / `load_model()` | `:232` `:246` | 读计算图 + 权重 |
| `layers()` 遍历 | `:238` | Vulkan 兼容性兜底 |
| `lm_head->init_shared()` / `init_from_file()` | `:268` `:281` | tied 权重零拷贝 / 独立加载 |
| `BpeTokenizer::LoadFromFiles()` | `:297` | 加载分词器 |
| `AddAdditionalSpecialToken()` | `:303` | 追加特殊 token |
| 读 `eos` / `bos` / `eos_ids` | `:306` | 生成终止条件 |
| 读 `attn_cnt` / `rope` / `tool_call_id` | `:320` | 决定循环次数与位置编码类型 |
| 读 `vision` 配置 → `load_net()` ×N | `:388` | 仅 VLM |

> **关键节点**：构造函数结束（`:467`）那一刻，`model.json` 已全部读进成员变量。之后的 prefill/generate **不再碰文件**。

### 阶段 2｜首轮 prefill（`:469-611`）

见 [02-prefill-and-decode.md](02-prefill-and-decode.md)，此处只列 API：

| API | 位置 |
| --- | --- |
| `bpe->encode()` | `:478` |
| `generate_rope_embed_cache()` | `:499` |
| `embed_net->create_extractor()` + `input`/`extract` | `:507` |
| 构造 causal mask | `:512` |
| `decoder_net->create_extractor()` + `set_kvcache_allocator` + `set_kvcache_max_seqlen_hint` | `:533` `:535` `:536` |
| `input("in0"~"in3")` | `:538` |
| `extract("out_cache_k{i}" / "out_cache_v{i}")` | `:548` |
| `extract("out0")` | `:572` |
| `lm_head->forward()` | `:578` |
| argmax | `:585` |
| 写 `ctx` | `:594` |

### 阶段 3｜generate 循环（`:962-1169`）

| API | 位置 | 备注 |
| --- | --- | --- |
| `clone_ctx(ctx)` | `:962` | 复制状态，**不动调用方的 ctx** |
| `eos_ids.count(cur_token)` | `:973` | 终止判断 |
| `tool_call_id` / `think_id` 状态机 | `:975` | 工具调用与思考链 |
| `bpe->decode({cur_token})` | `:1002` | 单 token → 文本 |
| `callback(text)` | `:1006` | **流式输出的出口** |
| `llm_run_text_embed()` | `:1014` | 查嵌入 |
| `generate_rope_embed_cache(1, ...)` | `:1028` | 单位置 RoPE |
| `mask` 全 0 一列 | `:1034` | decode 的特征 |
| `llm_run_decoder_with_kv()` | `:1045` | 喂 cache → 跑 → 取新 cache |
| `lm_head->forward()` | `:1113` | 隐状态 → logits |
| `llm_select_next_token()` | `:1127` | 采样（temperature/top-k/top-p/重复惩罚） |
| `report.print()` | `:1165` | 性能报告 |

---

## 三、本项目用到的 ncnn API 全集

**整个 LLM 运行时（聊天/VLM/OCR/ASR/Embedding）只依赖约 20 个 ncnn API。**

| 分类 | API | 出现位置 | 作用 |
| --- | --- | --- | --- |
| **网络生命周期** | `ncnn::Net::Net()` | `:141` | 建空网络 |
| | `load_param(path)` | `:218` | 读计算图（`.param`） |
| | `load_model(path)` | `:223` | 读权重（`.bin`） |
| | `layers()` | `:238` `:258` | 遍历层：Vulkan 检查 / 找 Embed 层 |
| | `opt`（成员） | `:147` | 该网络的全局选项 |
| **单次推理** | `create_extractor()` | `:507` `:533` | 建一次推理的句柄 |
| | `input(name, mat)` | `:508` | 按 blob 名喂输入 |
| | `extract(name, mat[, type])` | `:509` | 按 blob 名取输出 |
| | `set_kvcache_allocator(a)` | `:535` | 指定 kvcache 内存池 |
| | `set_kvcache_max_seqlen_hint(n)` | `:536` | 预分配容量，避免反复扩容 |
| **张量** | `ncnn::Mat(w,h,ptr)` | `:503` | 构造（**零拷贝包装外部内存**） |
| | `.clone()` | `:503` | 复制成自己管理的内存 |
| | `.row(i)` / `.row_range(s,n)` | `:577` | 取行 / 取行区间 |
| | `.fill(v)` / `.release()` | `:515` `:847` | 填充 / 释放引用 |
| | `.h` `.w` `.c` `.empty()` | `:679` | 元信息 |
| **内存池** | `UnlockedPoolAllocator` | `:104` | KV cache 专用池 |
| | `set_size_compare_ratio(0.f)` | `:105` | 关掉尺寸合并，避免 cache 被复用踩踏 |
| **选项** | `Option::num_threads` | `:147` | CPU 线程数 |
| | `Option::use_vulkan_compute` | `:164` | 是否走 GPU |
| | `Option::use_bf16_storage` | `:196` | 权重 BF16 存储 |
| | `Option::vulkan_device_index` | `:157` | 选哪块 GPU |
| **设备查询** | `get_physical_cpu_count()` | `:132` | 物理核数 |
| | `get_cpu_count()` | `:133` | 逻辑核数（兜底） |
| | `create_gpu_instance()` | `ncnn_llm_base.h:117` | 初始化 Vulkan（基类） |
| **层类型** | `ncnn::Embed` | `:260` | 嵌入层，取 `weight_data` |
| | `Layer::support_vulkan` | `:239` | 该层能否跑 GPU |

---

## 四、三层 API 边界（排错先定位层）

```
第 1 层  应用层      parse_options / run_cli / apply_chat_template
                    → 关心「用户输入什么、怎么显示」
                    → 出错表现：参数不对、模板拼错

第 2 层  LLM 运行时  ncnn_llm_gpt::prefill / generate / define_tools
                    → 关心「token 怎么流动、cache 怎么维护」
                    → 出错表现：输出乱码、越生成越怪、工具调用不触发

第 3 层  ncnn 引擎   Net / Extractor / Mat / Option / Allocator
                    → 关心「图怎么执行、内存怎么分」
                    → 出错表现：load_param 失败、extract 失败、形状不匹配
```

| 报错信息样例 | 层 |
| --- | --- |
| `Model path does not exist`（`main.cpp:38`） | 1 |
| `image placeholder token was not found`（`:662`） | 2 |
| `decoder load_param failed`（`:234`） | 3 |
| `ncnn extract failed for ...`（`:111`） | 3 |

---

## 五、三个值得注意的设计细节

### 1. lm_head 权重共享（tied embeddings，零拷贝）

```cpp
bool tied_embeddings = (embed_bin == proj_out_bin);          // :253
...
embed_weight = el->weight_data;                              // :261
lm_head->init_shared(proj_out_param, embed_weight, lm_opt);  // :268
```

输入嵌入和输出投影共用同一份权重时，直接从 `embed_net` 的 `Embed` 层抠出 `weight_data` 喂给 lm_head，**不加载第二个 bin**。日志打印省了多少 MB（`:271`）。

不理解 param/bin 结构就做不出这个优化。

### 2. Vulkan 兜底降级

```cpp
for (const auto* layer : decoder_net->layers()) {
    if (layer && !layer->support_vulkan) {
        printf("... Switching decoder to CPU.\n");
        decoder_net->opt.use_vulkan_compute = false;   // :241
        break;
    }
}
```
（`:237-245`）

INT8 block quant 的 Gemm 层不支持 Vulkan。**「开了 GPU」不等于「真的在用 GPU」**，要跑起来看这行日志。这也解释了声明的「量化层自动在 CPU 执行」。

### 3. `vocab_size` 运行时刻才算出来

```cpp
const int vocab_size = bpe->vocab_size();   // :920
sample_cfg.vocab_size = vocab_size;          // :1117
```

来自 `vocab.txt` 行数，不写在 `model.json` 里。`llm_select_next_token()` 用它截断 logits（`ncnn_text_runtime.cpp:106-108`），防止越界读词表外的垃圾值。

---

## 待办练习

1. 对照 `main.cpp:44` → `cli_runner.cpp:56-60` → `ncnn_llm_gpt.cpp:469`，按调用顺序写出每步的函数名、返回值类型，并标出「`model.json` 已全部读进内存、之后不再看文件」的时刻。
   - 答案要点：`prefill` 和 `generate` 都返回 `shared_ptr<ncnn_llm_gpt_ctx>`；分界点是构造函数结束（`:467`）。
2. 用户传了 `--use-vulkan` 但最终跑在 CPU 上，可能是哪个环节？作者为什么选择**整个 decoder 退回 CPU** 而不是逐层混合？
   - 答案要点：`:237-245` 的兼容性检查。逐层混合会让 cache 在 CPU/GPU 间反复拷贝，代价可能大于收益。
