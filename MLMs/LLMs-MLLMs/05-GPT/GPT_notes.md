# GPT notes

## 实验内容

大家好，本文将向大家介绍GPT(Generative Pretrained Model)的原理和原型实现。

和BERT的编码器架构不同，GPT采用解码器架构，主要用在文本生成，问答系统中，是当前流行的大语言模型的前身。
和BERT不同，GPT的生成式模型只能使用已经生成的内容信息进行上下文建模。
实现上采用因果注意力机制使用之前生成的内容作为上下文，通过自回归的方式生成序列。

文中对于GPT给出了原型实验的实验代码：
- `https://gitcode.com/kindlytrees/ArtificialIntelligence/blob/master/MLMs/LLMs-MLLMs/05-GPT/GPT_from_Scratch.ipynb`
- `https://gitcode.com/kindlytrees/ArtificialIntelligence/blob/master/MLMs/LLMs-MLLMs/05-GPT/transformers_decoders_from_scratch.py`

特别的针对解码器模型在推理时的自回归生成，给出了具体的原型实现并加以注释说明，同时附录部分对一些加速的方法技术做了更多扩展介绍。

GPT模型也是后续的大语言模型的基础，其模型架构和自回归生成的特性会贯穿于大语言模型的迭代升级过程中，为了使得回答的内容更加的对于人类可阅读，以及使得模型有更强的推理能力，编程语言代码生成能力，或者垂直领域的专业能力，通常需要在预训练模型的基础上进行较为复杂的微调训练，后续也将介绍更多相关的技术内容。



## 关于推理加速框架的简要对比说明
一个高性能的LLM推理服务通常是多种技术的组合：
在专用GPU上，运行一个经过INT8/INT4量化、并使用了FlashAttention等优化Kernel的模型，
同时利用KV缓存进行基础加速，并在此之上采用推测解码算法来进一步提升生成速度。

| 特性 | Llama.cpp | vLLM | TensorRT-LLM | Hugging Face TGI |
|------|-----------|------|----------------|------------------|
| **主要硬件** | CPU / Apple Silicon / (GPU) | NVIDIA GPU | NVIDIA GPU（A100/H100 最优） | NVIDIA GPU |
| **量化支持** | GGUF（4/5/6/8-bit） | AWQ, SqueezeLLM, FP8（实验） | FP8, INT8, INT4（NVIDIA 专属） | GPTQ, AWQ, bitsandbytes（8-bit） |
| **核心优化** | 量化 + SIMD | PagedAttention + Continuous Batching | TensorRT 图优化 + Kernel Fusion | Dynamic Batching + FlashAttention |
| **易用性** | ⭐⭐⭐⭐（本地运行极简） | ⭐⭐⭐（Docker 一键部署） | ⭐⭐（需编译模型） | ⭐⭐⭐⭐（HF 生态无缝集成） |
| **吞吐性能** | 低（CPU 限制） | 高 | 极高（NVIDIA 专属优化） | 中高 |
| **OpenAI API 兼容** | 否（需封装） | 是 | 否（需额外服务层） | 是（原生支持） |
| **适合场景** | 本地/边缘设备 | 云服务高并发 | 数据中心极致性能 | 快速 HF 模型部署 |