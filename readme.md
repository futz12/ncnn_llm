# ncnn llm
ncnn llm 旨在为ncnn框架提供大语言模型（LLM）的支持。ncnn 是一个高性能的神经网络前向计算框架，专为移动设备和嵌入式设备设计。通过集成大语言模型，ncnn llm 使得在资源受限的环境中运行复杂的自然语言处理任务成为可能。

ncnn llm is designed to provide support for large language models (LLM) within the ncnn framework. ncnn is a high-performance neural network inference framework optimized for mobile and embedded devices. By integrating large language models, ncnn llm enables the execution of complex natural language processing tasks in resource-constrained environments.

## 项目起源

本项目起源于nihui为ncnn添加了kvcache功能，这使得在ncnn上运行大语言模型成为可能。本人本着为社区贡献的精神，决定将这一功能进行整理和扩展，形成一个独立的项目，以便更多的开发者能够方便地使用和贡献。

The project originated from nihui's addition of the kvcache feature to ncnn, which made it possible to run large language models on ncnn. Motivated by a spirit of community contribution, I decided to organize and expand this functionality into an independent project, making it easier for more developers to use and contribute.

**ncnn对kvcache的支持处于实验性阶段，请编译master分支以获得最新功能。**

**ncnn's support for kvcache is in an experimental stage; please compile the master branch to obtain the latest features.**

## 目前状态

目前，ncnn llm 仍处于早期开发阶段，实现了基本的tokenizer和nllb模型的支持。

Currently, ncnn llm is still in the early stages of development, with basic support for tokenizers and the nllb model implemented.

本项目尽可能提供了详尽的文档和示例代码和完整的导出pipeline，帮助用户快速上手。但是不可避免的，随着库的更新老的导出pipeline可能会失效，用户可以参考示例代码进行调整或者提出issue寻求帮助。

The project provides detailed documentation, example code, and a complete export pipeline to help users get started quickly. However, as the library evolves, some older export pipelines may become obsolete. Users can refer to the example code for adjustments or raise issues for assistance.

## 未来计划

未来计划包括但不限于：

- 为上游提供相关优化补丁，提升ncnn对大语言模型的支持（直接提交上游，而不会出现在本项目中）
- 支持更多的模型和tokenizer
- 优化性能，提升推理速度和降低内存占用
- 增加更多的示例和文档，帮助用户更好地理解和使用本项目

Future plans include but are not limited to:
- Providing relevant optimization patches to upstream to enhance ncnn's support for large language models (directly submitted upstream and not appearing in this project)
- Supporting more models and tokenizers
- Optimizing performance to improve inference speed and reduce memory usage
- Adding more examples and documentation to help users better understand and use the project

欢迎大家关注和参与本项目，共同推动ncnn在大语言模型领域的发展！

TODO LIST:
- [ ] PaddleOCR VL
- [ ] QWen3 0.6B
- [ ] INT8 量化

## 模型获取方法

模型可以从以下链接获取：
[nllb-600M](https://www.modelscope.cn/models/futz12/ncnn_llm_nllb_600m)

## 编译和使用

git clone https://github.com/futz12/ncnn_llm.git

cd ncnn_llm

./build.sh

然后下载模型
下载完成后 指定路径 即可运行

nllb_main --model-dir ../../assets/ncnn_llm_nllb_600m/

