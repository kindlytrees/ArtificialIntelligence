# Diffusion Models

## 简要总结
- DDPM算法原理
    - 两个高斯概率分布的随机变量之和仍为高斯分布，基于特征函数的证明，请根据求期望和方差的公式得出求和后的高斯概率分布参数
    - vlb为生成交叉熵损失上界，以及负对数似然的下界，通过优化vlb去求解扩散模型，再优化vlb的过程中，可以推导展开得出模型和p(x_{t-1}|(x_{t},x_{0}))的分布近似
- DDPM算法实践
    - U-Net结构，视觉的自注意力机制的计算
    - 系数因子序列提前计算并cache
    - q_sample采样数据进行自监督训练，q_sample进行去噪推理，迭代后生成清晰图像
- DDIM算法原理和应用
    - 加速ddpm的推理生成，构造了逆向生成的概率分布，sigma及eta参数的定义和含义
- LDM算法原理和应用
    - VAE，隐空间，AutoEncoderKL实现了图像空间压缩感知，其基于GAN的训练框架，KL的系数权重较低，重在重建高质量图像的目标上
    - 基于隐空间的DDPM算法
    - 基于条件约束的图像生成,基于交叉注意力机制的实现
        - class guidance
        - layout 
        - semantic segmentation
        - super resolution
- Stable Diffusion LoRA
    - Low Rank Adaption， 学习\delta W,将其转换为两个地址参数矩阵的乘积
    - attention实现LoRA，其他的FFN等也可以实现LoRA，以满足更好的finetune
    - 基于LoRA可以实现基于特定风格的图像生成，如中国山水画，某个画家（如fango）风格的作品生成等
- Stable Diffusion Text Inversion
    - 基于特定新的token id的加入，finetune词嵌入层的新加入token id的参数部分（其他参数冻结）
    - 学习特定token和新的视觉概念(visual concepts)的关联
- Stable Diffusion DreamBooth
    - 特定subject或多个subjects的finetune任务
    - 加入class-specifc prior preservation loss防止出现language drift线性
- Stable Diffusion ControlNet
    - 加入更强的约束，实现更强的可控生成
    - 各种edge或草图(canny edge,hough lines, hed edge， scribbles)
    - depth image,semantic segmentaion image, normal image
    - 结构上控制条件通过zero conv层后和隐空间直接融合
    - 只有编码层会将网络结构和参数进行复制拷贝并进行参数学习，其他原始网络的参数冻结，推理时直接附件上特定控制条件的controlnet模型，实现整体的图像生成模型



