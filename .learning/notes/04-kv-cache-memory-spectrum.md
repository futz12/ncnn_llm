# 04 · KV Cache 内存管理谱系（PagedAttention 对照）

> 对应源码：`src/ncnn_llm_gpt.cpp`（`max_seqlen_hint` / `UnlockedPoolAllocator`）、`export/`、外部参照 [vLLM 论文 arXiv:2309.06180](https://arxiv.org/abs/2309.06180)（SOSP 2023）
> 状态：已讲授待实践
> 定位：支柱 E（边缘优化）的开篇——「KV cache 怎么放」这个问题域的全景

## 用途

回答一个问题：**同样是管 KV cache，为什么 vLLM 要搞分页，而本项目一个内存池就够了？**

结论先行：不是技术优劣，是**问题规模**决定方案复杂度。

---

## 一、两个战场

| | vLLM | ncnn_llm |
| --- | --- | --- |
| 部署位置 | 数据中心 GPU | 手机 / 嵌入式 / 桌面 CPU |
| 服务对象 | 几百上千并发请求 | 1 个用户 1 路对话 |
| 优化目标 | **吞吐**（tokens/s/卡） | **延迟 + 内存占用** |
| 核心矛盾 | KV cache 不够，batch 上不去 | 权重都放不下，别提 cache |
| 硬件 | A100/H100，80 GB 显存 | 4 核 CPU / 手机 4–8 GB RAM |

PagedAttention 是**服务端**问题的答案。但不理解它，就说不清本项目「没做什么、为什么可以不做」。

---

## 二、PagedAttention 机制

### 问题：KV cache 动态增长，但传统系统按上限预分配

- 请求进来先按 `max_seq_len`（如 2048）把 cache 整块占住，实际只生成 200 token → 90% 白占
- 多请求长度不一 → 外部碎片，显存有空闲但拼不出连续块
- **真正致命的**：batch size 被 KV cache 卡死。想多塞请求进 batch 提升 GPU 利用率？没内存。batch 上不去，吞吐就上不去

### 做法：借操作系统的虚拟内存分页

1. KV cache 切成固定大小 **block**（每块存固定数量 token 的 K/V）——像内存的「页」
2. 每请求维护一张 **block table**：逻辑块号 → 物理块号——像「页表」
3. **按需分配**：生成多少 token 分多少块，除最后一块外零浪费
4. **copy-on-write 共享**：parallel sampling / beam search 的多分支共享同一物理块，只在写时复制
5. **prefix caching**：多请求有相同 system prompt 时共享前缀块

### 论文结论（摘要原文要点）

- 实现 near-zero waste in KV cache memory + flexible sharing within and across requests
- 同延迟下吞吐比 FasterTransformer/Orca 提升 **2-4×**
- 序列越长、模型越大、解码算法越复杂，优势越明显

### 关键洞察

> **它优化的不是「算得快」，而是「同样的显存能塞下更多请求」。**

batch 从 1 提到 32，GPU 利用率上去，吞吐才上去。2-4× 是这个连锁反应的结果，不是 attention 算法本身变快了。

---

## 三、为什么本项目不需要它（五条）

1. **单流，没有碎片问题**。`ncnn_llm_gpt` 一次只处理一个 ctx，cache 从空开始（`:527`）单调增长，不存在多请求长度不一导致的外部碎片
2. **三个共享场景一个都不沾**。没有 parallel sampling、beam search、并发请求共享 system prompt，CoW 和 prefix caching 无处施展
3. **batch 恒为 1**。`generate()` 主循环（`:972`）一次一个 token，没有请求间 batching（prefill 的「序列内并行」是另一回事）。PagedAttention 是为把 batch 从 1 提到几十才需要的
4. **内存管理在更下层已做掉**。`ncnn::UnlockedPoolAllocator`（`:104`）是 ncnn 框架级内存池，PagedAttention 是 LLM 层分配策略，这里被框架吸收了
5. **硬件量级不同**。PagedAttention 的动机是「80 GB 显存也不够」；边缘设备瓶颈是「权重都放不下」——README 数字，INT8 后 decoder 446 MB，cache 增长不是首要矛盾

---

## 四、同一个思想的「穷人版」

PagedAttention 的核心手段，在本项目里都有雏形：

| PagedAttention | 本项目对应物 | 位置 | 差距 |
| --- | --- | --- | --- |
| 按需分配，避免按上限占满 | `set_kvcache_max_seqlen_hint(n)` | `:536` `:1048` `:1053` | 整块预分配 vs 分块按需 |
| 池化复用，避免频繁 malloc/free | `UnlockedPoolAllocator` | `:104` | 无块表，无逻辑/物理分离 |
| 块不能随意共享（防踩踏） | `set_size_compare_ratio(0.f)` | `:105` | 禁止合并 vs 引用计数 + CoW |

`max_seqlen_hint` 传 `position_id + max_new_tokens`——**按「已知要生成多少」估容量**，不按理论上限。与「按需分配」同向，粒度粗一个量级。

`set_size_compare_ratio(0.f)` 关掉池子尺寸合并，避免不同层的 cache 被合并复用导致互相踩踏——这正是 PagedAttention「块不能随便共享」的同一约束，只是用最粗暴的方式解决。

> **思想相通，复杂度差一个量级——因为问题规模差一个量级。**

---

## 五、内存管理策略谱系

```
        同样的内存，能塞下多少「正在进行的工作」？
   ┌──────────────────────────────────────────────────┐
   粗放                                    精细
    │                                        │
    ▼                                        ▼
 一次性 malloc          ncnn 本项目            vLLM
 整个 cache            UnlockedPoolAllocator  PagedAttention
 （最浪费）             + max_seqlen_hint     block table
                                              + CoW + prefix cache
    │                                        │
   batch=1，单用户                          batch=32+，多请求
   边缘设备                                 数据中心 GPU
```

同一个问题（KV cache 怎么放），三种答案。选择由**并发请求数 + 内存大小**决定。

---

## 六、什么时候需要回头学 vLLM

1. 要在一台机器上服务多个用户——哪怕 4 个并发，cache 管理立刻变瓶颈
2. 要做 parallel sampling / beam search——多分支共享前缀，CoW 直接省 N 倍
3. 要做 prefix caching——相同 system prompt 共享前缀，长 prompt 收益巨大
4. 想把 ncnn_llm 改成服务端——要引入 batch 维度，cache 管理问题随之而来

反之，做单机单用户边缘聊天应用，现在深挖 PagedAttention 性价比很低。

**建议节奏**：先把本项目 E 支柱（量化、内存池、SIMD）吃透，再读 vLLM 论文正文和 `block_table` 源码——因为那时已经亲手见过「不管 cache 会怎样」。

---

## 待办练习

1. 对比 `:525` 与 `:1048` 两处 `max_seqlen_hint` 算法：
   ```cpp
   const int max_seqlen_hint = (int)token_ids.size() + 512;    // :525 prefill
   ... ctx->position_id + cfg.max_new_tokens ...               // :1048 decode
   ```
   为什么 decode 能算准、prefill 只能拍 `+512`？
   - 要点：prefill 时不知道模型会回答多长；decode 时 `max_new_tokens` 是用户给的确定值。且 hint 只影响预分配容量、**不影响正确性**——它是性能参数不是功能参数。

2. 要把本项目改成「一个树莓派服务 4 个家庭成员，每人一路独立对话」，在不引入完整 PagedAttention 的前提下先做哪个改进？
   - A. `UnlockedPoolAllocator` 换成按 token 数增长的动态分配
   - B. 每个 ctx 单独 allocator，禁止任何共享
   - C. 引入请求间 batching，4 个请求一起过 decoder
   - 要点：选 C 要意识到改动远不止 cache 管理——mask 要从一列变成 `4×(历史+1)` 块结构、`ctx` 从单个变数组、eos 判断按请求分开。这正说明 PagedAttention 是**服务端架构**的一部分，不是能单独插进来的一个算法。
