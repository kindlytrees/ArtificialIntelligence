# GPT notes

一个高性能的LLM推理服务通常是多种技术的组合：

在专用GPU上，运行一个经过INT8/INT4量化、并使用了FlashAttention等优化Kernel的模型，、同时利用KV缓存进行基础加速，并在此之上采用推测解码算法来进一步提升生成速度。