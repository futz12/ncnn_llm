# ncnn_llm 学习笔记库

围绕 `ncnn_llm` 项目建立的「大模型边缘部署」知识体系。目标：能独立解释本项目从模型导出到端上推理的每一环及其设计原因。

## 知识地图（6 支柱）

```
A 模型从哪来          B 模型长什么样        C 文本怎么进模型
export/*.py    →    model.json      →    tokenizer/
                              ↓
                     D 模型怎么算
                  ncnn_llm_gpt.* + ncnn
                              ↓
        E 怎么算得快、算得省  ← 横向贯穿 →  F 怎么变成应用
     KV Cache / 量化 / 后端              examples/ + 采样
```

| 支柱 | 解决什么问题 | 主要文件 | 笔记 |
| --- | --- | --- | --- |
| A | PyTorch 模型 → ncnn 格式 | `export/*.py` | [01](01-export-pipeline.md) |
| B | 一个目录自描述一个模型 | `model.json` | [05](05-model-json-contract.md) |
| C | 文本 ↔ token 编号 | `src/utils/tokenizer/` | 待写 |
| D | 加载网络、前向计算、生成文本 | `src/ncnn_llm_gpt.*` | [02](02-prefill-and-decode.md) · [03](03-api-call-chain.md) |
| E | 省内存、提速度（横向贯穿） | `export/quantize_model.py`、`src/kernel/` | [04](04-kv-cache-memory-spectrum.md) |
| F | 包成用户能用的程序 | `examples/`、`src/sampling.*` | 待写 |

## 第一性原理：三堵墙

大模型（如 Qwen3-0.6B）FP32 约 2.4 GB，上边缘设备会同时撞上三堵墙：

| 墙 | 表现 | 本项目对策 |
| --- | --- | --- |
| 内存墙 | 权重放不进设备 RAM | BF16 / INT8 量化 |
| 算力墙 | 没有 GPU，只有 4 核 CPU | ncnn SIMD 优化 + 多线程 |
| 框架墙 | PyTorch 运行时几百 MB、依赖一堆 | ncnn 纯 C++ 推理图 |

README 基准数字（Qwen3-0.6B，i9-13900HX）：
- 权重 840 MB → 446 MB（INT8，省 47%）→ 打内存墙
- 解码 17.0 → 32.0 tokens/s（提速 71.9%）→ 打算力墙
- 整个项目 C++ + ncnn，无 Python 依赖 → 打框架墙

> 后面每个技术都可以问一句「它在打哪堵墙？」答不上来的就是还没理解到位。

## 笔记清单

| # | 主题 | 对应支柱 | 状态 |
| --- | --- | --- | --- |
| 01 | [模型导出管线](01-export-pipeline.md) | A | 已讲授待实践 |
| 02 | [Prefill 与 Decode](02-prefill-and-decode.md) | D | 已讲授待实践 |
| 03 | [完整 API 调用链](03-api-call-chain.md) | D | 已讲授待实践 |
| 04 | [KV Cache 内存管理谱系](04-kv-cache-memory-spectrum.md) | E | 已讲授待实践 |
| 05 | [model.json 字段契约](05-model-json-contract.md) | B | 已讲授待实践 |

## 存档约定

学习者要求**每轮对话都存入知识库**。执行方式：

- 每轮教学结束 → 写或更新一篇主题笔记（编号递增）→ 更新本索引 → 更新 `../state.json`
- 笔记是**提炼后的知识**，不是对话逐字稿：保留机制、代码行号锚点、坑、练习，去掉寒暄和过程
- 纯问答若产生新知识，同样成篇；若只是澄清，追加到相关笔记的对应小节
- 若日后需要原始对话流水，再加 `.learning/journal/`，当前不维护

## 课程进度

见 [`../state.json`](../state.json)。

## 环境备注

- 工作区 `H:\work\deploy_model\ncnn_llm`，Windows
- 项目尚未构建，`assets/` 仅有 `mclip_unigram_tokenizer.txt`，未下载模型权重
- 该工作区内所有 `pwsh` 命令被 Windows ACL 拒绝（`SetNamedSecurityInfoW failed (Win32 5)`），但文件读写与 glob/read 可用。实践命令需由学习者在自己的终端执行
- `.learning/` 未加入 `.gitignore`，提交前需决定是否排除
