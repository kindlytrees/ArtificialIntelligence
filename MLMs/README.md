# LLM

## 简要总结

- 分词器，python里正则表达式library re的基本使用进行基本分词，子词分词，BPE(GPT)基于相邻子词出现的次数，WordPiece(Bert)基于相邻子词出现的频率，uft-8的可变长跨语言编码
- word2vec,word embedding技术，主要有cbow(从上下文预测当前)，skip-gram（从当前预测上下文），glove（基于窗口邻域的共现矩阵，构建基于当前词向量和上下文词向量的损失函数）等，基本基于浅层网络建模上下文，构建出含有语言语义信息的词向量
- transformer架构，self attention的计算，支持批量的mask计算，支持时序的causual attention head的mask计算，transformer的三种架构，encoder only，decoder only，encoder decoder
- transformer架构应用实践，基于编码器的文本分类，基于解码器的nanoGPT，基于编码器和解码器架构的语言翻译等(基于编码器编码器架构的mask的设置及其原理分析可参看第三部分的notes说明)
- BERT原理和实践，BERT的原生实现，基于BERT进行文本分类，基于BERT和SQuad数据实现抽取式问答应用
- GPT原理和实践，自回归语言的基本原理实践，自回归生成推理的一些细节分析, 基于temperature参数的生成多样性的控制，基于topk限制的采样生成
- RLHF，active policy，rollout policy，ref policy，-KL penalty参与reward的计算，定义为DKL(pi_active||pi_ref),将kl散度以reward的形式嵌入到策略梯度的回顾，实现了参数变化的软约束


