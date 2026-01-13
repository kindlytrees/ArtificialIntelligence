# Word Embeddings notes

- cbow算法介绍：https://www.kaggle.com/code/alincijov/nlp-starter-continuous-bag-of-words-cbow
- 基于twitter论坛评论数据的情感分析：https://colab.research.google.com/drive/1ricskXPPGKX7qiYd1Zm8wUdFbF7Vh7es?pli=1&authuser=2#scrollTo=E5E1UwMc5QsQ
- Byte Pair Encoding tokenization: https://huggingface.co/learn/nlp-course/en/chapter6/5
- https://www-nlp.stanford.edu/projects/glove/

## glove算法中目标函数的由来

- 辅助gemini回答：

这是 GloVe 的精髓所在。研究人员发现，词与词之间有意义的关系，通常蕴含在共现概率的比例中。
我们来看一个经典的例子：
假设我们想考察 ice (冰) 和 steam (蒸汽) 这两个词。
我们选择两个探针词 (probe words)：solid (固体) 和 gas (气体)。
P(solid | ice) 是 solid 出现在 ice 上下文中的概率，这个值会很大。
P(solid | steam) 是 solid 出现在 steam 上下文中的概率，这个值会很小。
因此，比例 P(solid | ice) / P(solid | steam) 会是一个非常大的数。
反之：
P(gas | ice) 会很小。
P(gas | steam) 会很大。
因此，比例 P(gas | ice) / P(gas | steam) 会是一个非常小的数。
对于像 water (水) 这种与两者都有关的词，或者 fashion (时尚) 这种与两者都无关的词，这个比例会趋近于 1。
结论：共现概率的比例能够有效地把相关的词（solid vs ice）和不相关的词（gas vs ice）区分开来。

F(vi,vj,vk) = p(i,k)/p(j,k)
F(vi-vj, vk) = p(i,k)/p(j,k)
exp(vi-vj, vk) = p(i,k)/p(j,k)
(vi-vj,vk) = logp(i,k) - logp(j,k)
(vi,vk) = log(i,k) + c

基于共现概率的关系，构建向量之间的差和单词k之间的内积关系，
和对数的概率之比之间存在某种关联，因此将F函数构建为指数函数，然后取对数
最后得出两个向量之间的内积和其共现矩阵值的对数存在着关联