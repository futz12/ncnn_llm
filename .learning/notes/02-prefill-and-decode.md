# 02 · Prefill 与 Decode（支柱 D）

> 对应源码：`src/ncnn_llm_gpt.cpp`（`prefill()` `:469`、`generate()` `:962`）、`src/ncnn_text_runtime.cpp`
> 状态：已讲授待实践

## 总纲：两个世界

LLM 一次生成一个 token，但**开头 prompt 的处理方式和后续 token 完全不同**。

| | prefill（预填充） | decode（解码） |
| --- | --- | --- |
| 一次处理 token 数 | 整条 prompt，N 个 | 1 个 |
| 能否并行 | 能，N 个一起算 | 不能，必须等上一个出来 |
| 瓶颈 | 算力（矩阵大） | 访存（反复读写 KV cache） |

对应 `prefill()`（`:469`）和 `generate()` 的循环（`:962`）。
**prefill 的任务：一次性把 prompt 嚼碎，并把 KV cache 建好。**

## prefill 五阶段

### 阶段 1｜分词（`:475-480`）

```cpp
token_ids = bpe->encode(input_text, false, false);
if (bos >= 0) token_ids.insert(token_ids.begin(), bos);
```
两个 `false`：不加 eos、不加其他特殊 token。BOS 手动插最前——decoder-only 模型惯例，标记「一句话从这里开始」。

### 阶段 2｜一次性生成整条 prompt 的 RoPE（`:487-501`）

```cpp
generate_rope_embed_cache(token_ids.size(), rope_head_dim, 0, cos_cache, sin_cache, rope_theta);
```

RoPE = 旋转位置编码，让模型知道每个 token 在第几位，实现是预计算 cos/sin 表。**第一个参数是整条 prompt 长度**——这是 prefill 的第一个特征。

上方 if/else 链（`:490-500`）按模型配置选变体：RoPE / LongRoPE / NTK-RoPE / YARN-RoPE，都是为延长上下文窗口的不同插值策略（属支柱 E）。

### 阶段 3｜查嵌入表（`:503-510`）

```cpp
ncnn::Mat input_ids_mat = ncnn::Mat((int)token_ids.size(), 1, (void*)token_ids.data()).clone();
ncnn::Extractor ex = embed_net->create_extractor();
ex.input("in0", input_ids_mat);
require_extract(ex.extract("out0", token_embed), token_embed, "embed/out0");
```

首次出现的 ncnn 概念：

- **`ncnn::Extractor`**：执行一次推理的句柄。`create_extractor()` 新建 → `input()` 喂输入 → `extract()` 取输出。每次推理都要新建，因为 Extractor 内部持有中间结果
- **`ncnn::Mat`**：ncnn 张量类型，`w/h/c` 三维。`Mat(w, h, 指针)` 是**零拷贝包装**，直接拿外部内存用。所以后面必须接 `.clone()` 拿自己管理的副本，否则 `token_ids` 一销毁就成野指针
- **`require_extract`**（`:109-114`）：薄封装，extract 失败抛异常。ncnn 的 extract 返回 int 错误码不抛异常，全靠调用方检查——C API 常见样板

### 阶段 4｜构造因果掩码（`:512-522`）

```cpp
ncnn::Mat mask((int)token_ids.size(), (int)token_ids.size());
mask.fill(0.0f);
for (int i = 0; i < (int)token_ids.size(); i++) {
    float* row = mask.row(i);
    for (int j = i + 1; j < (int)token_ids.size(); j++) {
        row[j] = -1e38f;
    }
}
```

因果掩码是自回归模型的铁律：**第 i 个 token 只能看前 i 个，不能偷看未来**。

- L×L 方阵，L = prompt 长度
- 上三角（`j > i`）填 `-1e38f`，下三角和对角线填 0
- 语义：第 i 行第 j 列 = 第 i 个 token 对第 j 个的注意力权重；填 `-1e38` 使 softmax 后权重趋近 0

> **为什么是 `-1e38` 而不是 `-inf`**：`-inf` 参与 softmax 的 exp 会产生 `NaN`（`inf - inf`）。用足够大的负数既能屏蔽又不 NaN。所有 Transformer 实现的共同选择。

`.row(i)` 取第 i 行首指针，**不检查越界**——高性能库的常见取舍。

### 阶段 5｜跑 decoder，取出 KV cache（`:533-573`）

```cpp
ncnn::Extractor ex = decoder_net->create_extractor();
ex.set_kvcache_allocator(kv_alloc);
ex.set_kvcache_max_seqlen_hint(max_seqlen_hint);
ex.input("in0", token_embed);   // 嵌入
ex.input("in1", mask);          // 因果掩码
ex.input("in2", cos_cache);     // RoPE cos
ex.input("in3", sin_cache);     // RoPE sin
```

**这四个输入正好是导出时 pnnx 收到的那四个 `inputshape`**（`llama_export.py:426-429`）。导出与运行时的契约在这里对上——所以导出时必须保证包装类 forward 签名与运行时喂的一致。

```cpp
for (int i = 0; i < attn_cnt; i++) {
    std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
    const int k_ret = ex.extract(name_k_out, k_cache, 1);
    ...
}
```

- `attn_cnt` 来自 `model.json` 的 `setting.attn_cnt`（Qwen3-0.6B = 32），**层数决定取多少对 cache**
- `out_cache_k%d` 这些名字正是 `add_kvcache.py:102` 写进 param 的 blob 名。**导出时命名、运行时按名取回**
- extract 第三参数 `1` 标记这块输出走 kvcache 专用分配路径，与 `set_kvcache_allocator` 配套

### 收尾｜lm_head + 第一个 token + 存档（`:574-604`）

```cpp
ncnn::Mat last_hidden = decode_out.row_range(total_prompt_tokens - 1, 1);
logits = lm_head->forward(last_hidden, embed_net->opt);
```

**只取最后一个位置**。decoder 对 N 个 token 都输出隐状态，但自回归只关心「最后一个之后接什么」。

```cpp
ctx->kv_cache = std::move(kv_cache);        // :594
ctx->cur_token = next_token_id;             // :595  第一个待生成 token
ctx->position_id = total_prompt_tokens;     // :596  已处理位置数
```

**`ctx` = 对话状态**（KV cache + 当前 token + 位置计数），`generate()` 靠它续上。

## prefill / decode 的分界：三行代码

decode 单步的 mask（`:1034-1038`）：

```cpp
ncnn::Mat mask(ctx->kv_cache[0].first.h + 1, 1);
mask.fill(0.f);
```

**decode 的 mask 是一列全 0 的向量**，形状 `(历史长度+1) × 1`。

对比 prefill 的 L×L 方阵、上三角 `-1e38`，差异说明一切：

- prefill：N 个 token，第 i 个要屏蔽后面 N-i-1 个 → 需要完整方阵
- decode：只有 1 个新 token，它要**看全部历史**，没有需要屏蔽的未来 → 一列 0 就够

**这一列 0 就是「增量解码」的全部秘密**：新 token 无条件 attend 历史，历史 K/V 已在 cache 里，不用重算。

decode 的 cache 喂入（`:1060-1068`）：

```cpp
for (int i = 0; i < attn_cnt; ++i) {
    ex.input("cache_k%d", ctx->kv_cache[i].first);
    ex.input("cache_v%d", ctx->kv_cache[i].second);
    ctx->kv_cache[i].first.release();
    ...
}
```
先喂旧 cache → 跑 → 取出新 cache 覆盖。prefill 是**只取不喂**（cache 从空开始）。一个产出、一个消费，闭环。

## 三个 prefill 重载

| 重载 | 位置 | 用途 | 关键差异 |
| --- | --- | --- | --- |
| `prefill(text)` | `:469` | 首轮，从零建 ctx | `create_ctx()` 新建；mask 是 L×L |
| `prefill(text, ctx)` | `:778` | 多轮对话续接 | `clone_ctx(ctx)` 复制旧状态；mask 宽 = 新 token + 历史；`position_id +=` |
| `prefill(text, bgr, ctx)` | `:613` | 图文输入 | 多 `get_visiual_features()` 抽图像特征、`inject_image_embeds()` 把图像向量替换到 `<|image_pad|>` 占位符位置 |

后两个都用 `clone_ctx()`（`:90-95` → `ctx->clone()`），**不动调用方传入的 ctx**。所以 `generate()` 处理工具调用时可放心重新 prefill（`:959`）。

`create_ctx()`（`:97-107`）按 `sconv_cnt`/`gdr_cnt` 是否 > 0 决定造 `qwen3_5_ctx` 还是 `ncnn_llm_gpt_base_ctx`。Qwen3.5 用混合注意力（GDR + ShortConv），除 KV cache 外还要多维护两种 cache（`ncnn_llm_gpt.h:99-100`）。

## 待确认的观察

prefill 末尾选第一个 token 用**纯 argmax**（`:581-592`），而 `generate()` 循环用完整 `llm_select_next_token`——带 temperature / top-k / top-p / 重复惩罚（`ncnn_text_runtime.cpp:99-168`）。

即：**第一个 token 恒为贪心，后续才走采样策略**。读码观察，待讲到支柱 F（采样）时再判断是刻意设计还是疏漏。

## 待办练习

1. 对照 `:512-522` 和 `:1034-1038`，用「第 i 个 token 能看到哪些 token」解释：
   - prefill 为什么需要 L×L 方阵且上三角填 `-1e38`？
   - decode 为什么一列全 0 就够？
   - 若 decode 也用方阵会发生什么？（关键：decode 时只有 1 个新 token，喂方阵形状不匹配，ncnn 会在 SDPA 层报错。约束：**mask 形状必须与当前处理的 token 数匹配**）
2. 预测：`:577` 的 `row_range(total_prompt_tokens - 1, 1)` 改成 `row_range(0, 1)` 会怎样？
   （答案：模型退化成只根据 prompt 第一个词猜后续，完全忽略其余内容——第 0 个位置的隐状态只经过对前 1 个 token 的注意力）
