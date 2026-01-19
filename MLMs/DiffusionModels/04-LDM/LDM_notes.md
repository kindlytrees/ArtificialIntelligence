# LDM notes

## VAE中的相关的损失函数的计算

$$
\begin{aligned}
& K L\left(N\left(\mu, \sigma^2\right) \| N(0,1)\right) \\
= & \int \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-(x-\mu)^2 / 2 \sigma^2}\left(\log \frac{e^{-(x-\mu)^2 / 2 \sigma^2} / \sqrt{2 \pi \sigma^2}}{e^{-x^2 / 2} / \sqrt{2 \pi}}\right) d x \\
= & \int \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-(x-\mu)^2 / 2 \sigma^2} \log \left\{\frac{1}{\sqrt{\sigma^2}} \exp \left\{\frac{1}{2}\left[x^2-(x-\mu)^2 / \sigma^2\right]\right\}\right\} d x \\
= & \frac{1}{2} \int \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-(x-\mu)^2 / 2 \sigma^2}\left[-\log \sigma^2+x^2-(x-\mu)^2 / \sigma^2\right] d x \\
& K L\left(N\left(\mu, \sigma^2\right) \| N(0,1)\right)=\frac{1}{2}\left(-\log \sigma^2+\mu^2+\sigma^2-1\right)
\end{aligned}
$$

上述公式后面的部分的计算基于如下的期望和方差定义

$$
\begin{aligned}
& \int_x \frac{x^2}{2} \mathcal{N}\left(\mu, \sigma^2\right) dx -\int_x \frac{(x-\mu)^2}{2 \sigma^2} \mathcal{N}\left(\mu, \sigma^2\right) dx \\
& = \quad \frac{\mathbb{E}\left[x^2\right]}{2}-\frac{\mathbb{E}\left[(x-\mu)^2\right]}{2 \sigma^2} \\
& \operatorname{Var}(x)=\mathbb{E}\left[x^2\right]-(\mathbb{E}[x])^2 \\
& \mathbb{E}\left[x^2\right]=\sigma^2+\mu^2 \\
& \frac{\mathbb{E}\left[(x-\mu)^2\right]}{2 \sigma^2} = 1
\end{aligned}
$$

交叉注意力实现文本和隐空间的cross attention的计算，如下公式的表述有变成习惯和数学物理公式表述上的习惯上的不同，计算上和之前介绍的交叉注意力的介绍是一致的

$$
\begin{aligned}
& 文本特征： \tau_\theta(y) \in \mathbb{R}^{M \times d_\tau} \\
& 展平的输入特征图 \varphi_i\left(z_t\right) \in \mathbb{R}^{N \times d_\epsilon^i} \\
& //下面的式子是表述习惯的问题，Q=ZW_Q,K=\tau_\theta(y)W_K,V=\tau_\theta(y)W_V  \\
& Q=W_Q^{(i)} \cdot \varphi_i\left(z_t\right), K=W_K^{(i)} \cdot \tau_\theta(y), V=W_V^{(i)} \cdot \tau_\theta(y) \\
& \mathrm{W}_{\mathrm{Q}}^{(i)} \in \mathbb{R}^{d \times d_\tau} \quad \mathrm{W}_K^{(i)} \in \mathbb{R}^{d \times d_\tau} \quad \mathrm{W}_V^{(i)} \in \mathbb{R}^{d \times d_\epsilon^i} \\
& \text { Attention }(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) \cdot V
\end{aligned}
$$

### 问题1
在基于ldm的文本约束生成的时候，text和图像作为pair为原始样本，text embedding和图像的隐空间的z0的t时刻采样，以及时间信息的embedding进行求和融合后作为输入，输出为error的预测，这是一种跨模态的融合方式，还可以通过交叉注意力，以及两种方式结合的方式实现基于约束的条件生成，在推理的时候，基于zT和text embedding及时间的每一步的迭代去噪的推理上的输入和计算方法也是一样的。请对上述的理解做更多详细的补充说明

有没有将文本的编码和也直接add到注意力特征的输入上，即三个embedding的加法，和bert中的nsp的任务类中的三个embedding相加类似

辅助gemini回答：
您的想法本身是合理的，并且“加法”这种融合方式确实在模型中被使用了（用于时间条件）。但对于需要精细对齐的图文跨模态任务，直接相加会丢失太多信息，其能力远不如交叉注意力。因此，LDM的设计者明智地为不同的条件选择了最适合它们的融合机制

直接相加没有充分利用不同通道之间的充分对齐机制，生硬的直接相加反而影响不同通道方面相关性的影响，而交叉注意力机制实现了不同通道的序列中的点时刻的两两相关性进行的计算

ldm中条件信息可以直接和z0进行concat（或者类似位置编码直接相加），或者作为cross attention参与计算，但是相加会出现不同模态信息对齐的问题，基于token级别的concat可以也是一种通常的跨模态融合的机制，如实现文本token和空间（图像）token的拼接，并且在许多先进的视觉语言模型（Vision-Language Models）中被广泛应用，将这些不同来源的token序列在序列维度上直接拼接起来，形成一个统一的、更长的序列，每个token（无论是来自文本还是图像）都可以与序列中的任何其他token进行交互。这意味着一个文本token可以“关注”相关的图像块，一个图像块也可以“关注”描述它的文本token。这种交互是深度、双向且非常灵活的，能够学习到非常精细和复杂的跨模态对应关系，但也存在着计算量更大（Transformer的自注意力机制的计算复杂度与序列长度的平方成正比（O(n²)）。将多个模态的token拼接在一起会使序列变得非常长，从而导致巨大的计算和内存开销，尤其是在处理高分辨率图像时会产生大量图像块token），但来自不同模态的嵌入向量在统计分布上可能存在巨大差异。直接将它们输入同一个模型可能会使训练不稳定或效果不佳，同时如果训练数据不足或不佳，这种对齐关系就学不好。具体的不同模态的token的concat方式主要有：
- 方法一：使用特殊分隔Token (Special Separator Tokens)
[CLS] a cat [SEP] [IMG] image_patch_1 image_patch_2 image_patch_3 image_patch_4
- 方法二：使用模态类型嵌入 (Modality Type Embeddings)
Final_Token_Embedding = Word_Embedding + Positional_Embedding + Modality_Type_Embedding
- 实践中的组合应用
在先进的多模态大模型中，通常会将方法一和方法二组合使用，这是最强大和最鲁棒的策略。

### 问题2 AutoencoderKL 在训练的过程中采用了gan的训练框架，在生成阶段，fake的损失要偏向生成真的结果，在判别阶段，fake和原始图片要有好的区分度

### 问题3 super resolution的条件生成

ldm用来做super resolution的条件生成的具体实现方法是什么，是将低清晰度的图像作为条件输入，然后以高清晰都的图像结果为目标对吗？

关于数据准备部分的辅助gemini回答：
目标 (Target): 高分辨率图像 y (e.g., 512x512)。这就是我们希望模型最终能生成的东西。
条件 (Condition): 低分辨率图像 x (e.g., 128x128)。这是我们给模型的输入提示。
制作配对数据: 我们通常从一个高分辨率图像数据集开始。对于每一张高分辨率图像 y，我们通过一个下采样算法（如双三次插值 Bicubic Downsampling）来创建其对应的低分辨率版本 x。这样，我们就得到了大量的 (x, y) 配对数据。

## References
- PyTorch-VAE/models/vanilla_vae.py at master · AntixK/PyTorch-VAE (github.com)
- https://zhuanlan.zhihu.com/p/627616358
- https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

