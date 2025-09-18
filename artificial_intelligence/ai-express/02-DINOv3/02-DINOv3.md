# DINOv3

## DINOv3的要点和关键创新思路想法
- 采用SSL的蒸馏方法实现视觉基础模型
- 学生网络和老师网络同样架构，老师网络为学生网络的EMA
- 基于DINOv2以及相关的算法（聚类等）实现data collection和curation 
- 加入了正则化koleo loss，使得patch的embedding向量的角度分布更加均匀，embedding的长度分布在一个范围内，主要是DINO的主损失函数来进行限制
- 加入gram anchoring loss（gram teacher和main teacher，gram teacher在main teacher每更新10k个iteration后再更新参数）student的gram和gram teacher的gram matrix要对齐
- 在多种视觉任务上表现出色
    - segmentation
    - depth estimation
    - video object(mask) tracking
    - video classifier

## dinov3等视觉基础模型，采用transformer架构，在输出的时候再将特征转化为二维的patch布局方式，一个patch大小为16*16，这样空间大小就成为原来的1/16了对吗，然后基于特征的pca还原到3维空间，并用rgb可视化出来后，得出的特征图看上去也挺清晰，每个patch的embedding的维度是否蕴含着该patch内更加丰富的细节信息，而这些信息如何去抵消掉16分之一的小采样带来的空间分辨率的损失呢？

### dinov3中KoLeo Loss要对embedding做归一化或再算loss，这样主要是空间的夹角分布计较均匀对吗，在实际中会对embedding的向量长度没有什么限制吧？

### 为什么dinov3中用data collection and curation为什么不用clean，curation和clean相比有哪些不同？

摘自大模型回答的两个关键句子：

- 数据清洗是一个相对基础和被动的过程。它的主要目标是识别并处理数据集中有问题、有错误或低质量的样本。
- 数据策展是一个更高级、更主动、更具目标导向性的过程。它不仅包含了数据清洗的所有步骤，更重要的是，它关注整个数据集的组成、多样性、平衡性和最终目标。

### 关于蒸馏算法的一些总结

总结对比(答案摘自大模型的部分回答)
特性	经典知识蒸馏	DINO 的自蒸馏
目标	模型压缩，让小模型学到大模型的性能。	自监督学习，让模型在没有标签的情况下学习视觉表示。
老师网络	预训练好的、固定的、通常比学生大。	与学生同构、无预训练、参数是学生的 EMA。
学生网络	通常比老师小，从头学习。	与老师同构，通过反向传播积极学习。
蒸馏的知识	关于特定任务（如分类）的软标签。	关于视觉内容一致性的高维语义分布。
蒸馏的含义	将“全知者”的知识传授给“初学者”。	将“历史共识”的稳定性引导“当前探索者”，防止其坍塌。

因此，DINO 中的“蒸馏”是一个非常巧妙的比喻。它保留了“学生-老师”、“软标签”、“知识传递”这些核心框架，但将其应用到了一个全新的领域——自监督学习，并赋予了每个角色全新的含义。