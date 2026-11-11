# 01 · 模型导出管线（支柱 A）

> 对应源码：`export/llama_export.py`、`export/add_kvcache.py`、`export/bf16_convert.py`、`export/quantize_model.py`
> 状态：已讲授待实践

## 要解决的问题

Hugging Face 模型是 PyTorch 格式（`config.json` + `model.safetensors`），运行需要 Python + PyTorch + transformers。边缘设备上这三样都不能要。

ncnn 的世界只有两种文件：

| 文件 | 内容 |
| --- | --- |
| `*.ncnn.param` | 计算图：有哪些层、怎么连、每层什么参数 |
| `*.ncnn.bin` | 权重：所有层的数值 |

**导出 = 把 PyTorch 动态计算图固化成 ncnn 静态计算图 + 权重。** 这一步做完，Python 可以彻底扔掉。

## 为什么切成三段

`llama_export.py:91-194` 定义三个包装类：

| 类 | 作用 | 一次对话执行次数 |
| --- | --- | --- |
| `EmbedExport` | token id → 隐状态向量 | 1 次 |
| `DecoderExport` | 隐状态 → 隐状态（N 层 transformer） | N 次 |
| `LMHeadExport` | 隐状态 → logits | N 次 |

三个真实原因：

1. **量化精度**：README 明确「保持 LM Head / Proj Out 原始精度」。lm_head 是词表维度输出层（Qwen3 约 15 万维，`151936 × 1024 ≈ 1.5 亿`参数），是量化重灾区，压坏直接表现成胡言乱语。切开才能「只量化 decoder」。
2. **执行频率不同**（见上表）。
3. **pnnx 更可靠**：图越小，`torch.jit.trace` 越不易出错。脚本给每个模块传两组输入长度（`L1=32`、`L2=64`，`:422-429`）告诉 pnnx 图要支持动态长度。

## 8 步流程

| 步 | 做什么 | 代码位置 | 验证 |
| --- | --- | --- | --- |
| 1 | 加载 FP32 模型 | `:378` | 强制 `dtype=torch.float32`，作为后续所有基准 |
| 2 | 导出分词器 → `vocab.txt` / `merges.txt` | `:366` `write_tokenizer()` | — |
| 3 | 包装 + 与 PyTorch 对拍 | `:382-404` | decoder `max|d| < 2e-3`；lm_head `< 5e-3` |
| 4 | `torch.jit.trace` → `pnnx`（`fp16=0`） | `:406-435` | 产出 FP32 param/bin |
| 5 | ncnn 图 vs torch 图交叉验证 | `:497-502` | `< 2e-3` |
| 6 | **改写 SDPA 支持 KV cache** | `:504-506` → `add_kvcache.py` | 空 cache 必须复现整序列结果 |
| 7 | `ncnnoptimize flag=2` → BF16 | `:570-577` | `rel_l2 < 0.02`、`cos > 0.999` |
| 8 | `ncnnllm2int` → INT8（独立目录） | `:594-601` | `rel_l2 < 0.10`、`cos > 0.995` |

另有步骤 8（model.json 生成）在 `:516-559`。

### 关键工具

- `torch.jit.trace`：用示例输入跑一遍，把 Python 执行路径录成静态图（TorchScript）
- `pnnx`：把 TorchScript 翻译成 ncnn param/bin

### 结构断言（和数值断言同等重要）

```python
if op == "SDPA" and n != num_layers:
    raise RuntimeError(f"expected {num_layers} SDPA layers, found {n}")
```
（`:493-494`）decoder 里的 SDPA 层数必须正好等于模型层数。

## KV Cache 改写（技术核心）

**问题**：pnnx 导出的 SDPA 是 **4 输入 / 1 输出** `(q, k, v, mask) → out`，语义是每次前向把整条序列完整算一遍。生成第 100 个 token 时要把前 99 个的 K/V 全部重算，总算力 `1+2+...+N ≈ N²/2`，**平方级增长**。

**ncnn 原生支持增量解码**，签名 **6 输入 / 3 输出**：
```
SDPA: (q, k, v, mask, cache_k, cache_v) → (out, out_cache_k, out_cache_v)
```

`add_kvcache.py` 做的事：

```python
pd["5"] = "1"   # has_mask
pd["7"] = "1"   # kvcache_enabled ← 打开增量解码
```
（`:92-95`）

并补一个输入层产出每层 cache blob：
```
Input kv_cache 0 N cache_k0 cache_v0 cache_k1 cache_v1 ...
```
（`:114-119`）

效果：每个 token 的 K/V **只算一次**，总算力 O(N²) → O(N)。512 token 场景省约 256 倍计算。

**验证方法极优雅**（`:510-514`）：

```python
got2 = ncnn_infer(dec_param, dec_bin, check_inputs)   # cache 为空
assert np.abs(got2 - check_target).max() < 2e-3
```

> **空 cache 的增量图，必须和整序列图算出完全相同的结果。**
> 因为「只处理最后一个 token、但 cache 里有之前全部 K/V」在数学上等价于「一次性处理整条序列」。这个不变式成立，改写就对了。

## 验证哲学（最值得学的）

模型转换是最容易**静默出错**的工作——图能跑、不报错、输出像人话，但数值已偏。等端上发现模型变傻，无法定位是哪一步坏的。所以每步都留数值证据。

| 阶段 | 验证什么 | 阈值 |
| --- | --- | --- |
| 包装后 | 手写包装器 vs 原模型 | `2e-3` / `5e-3` |
| pnnx 后 | ncnn 图 vs torch 图 | `2e-3` |
| 图结构 | Gemm 权重是否折叠、SDPA 层数是否等于 `num_layers` | 精确相等 |
| KV cache 后 | 空 cache 增量图 vs 整序列图 | `2e-3` |
| BF16 后 | 降精度 vs FP32 基准 | `rel_l2<0.02, cos>0.999` |
| INT8 后 | 降精度 vs FP32 基准 | `rel_l2<0.10, cos>0.995` |

## 已踩过的坑

### 坑 1：包装类里不要缓存子模块

`DecoderExport.__init__` 上方注释（`:120-127`）：

> 不要写 `self.layers = self.model.layers`。

原因：同一个 `ModuleList` 以两个属性名注册，pnnx 分不清参数归属——**一边把权重烤进 Gemm 层，一边又额外生成一个带权重的 Gemm 输入（MemoryData 节点）**。ncnn 的 Gemm 在 `constantA=0` 时读 `bottom_blobs[0]` 当 A 矩阵，权重被当成激活值用掉，隐状态形状从 `(L, hidden)` 塌成 `(hidden, hidden)`。

防御性检查（`:475-487`）：
```python
if types.get("MemoryData", 0) != 0 or set(nin) - {1}:
    raise RuntimeError(f"{what}: linear weights were not folded into the Gemm layers ...")
```
扫产出的 param，发现 `MemoryData` 节点或 Gemm 输入数不是 1 就直接报错。

**共性**：错误发生在导出期，症状出现在推理期，中间隔一堆步骤。所以在导出脚本里加断言，把问题拦在产生的地方。

### 坑 2：torch 和 ncnn 不能同一个进程

`ncnn_infer()`（`:303-325`）用**子进程**跑 ncnn，注释写明：torch 和 ncnn 各带一套 OpenMP 运行时，同一 Python 解释器里 import 两个，Windows 上一开始计算就死锁。

## 产出物

`assets/<model>/` 目录：
- `*_embed_token.ncnn.param/bin`
- `*_decoder.ncnn.param/bin`
- `*_proj_out.ncnn.param/bin`
- `vocab.txt`、`merges.txt`
- `model.json`

`prefix` 默认取输出目录名（`:643`）。

## 待办练习

1. 打开 `export/llama_export.py:346-351`，找出六个文件名，说出 `assets/qwen3_0.6b/` 下应有哪些文件。检查点：decoder 那组体积远大于另外两组。
2. 预测：删掉第 7 步 `add_kvcache` 直接推理，程序能跑吗？输出对吗？速度怎样？（提示：KV cache 不是「优化」，而是运行时与模型的契约）
