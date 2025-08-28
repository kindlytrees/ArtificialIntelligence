# VA-VAE

## LDM回顾
- VAE实现图像到隐空间的映射
    - GAN训练框架，加上视觉感知损失
    - 实现高质量感知压缩
- UNet结构，在卷积的框架内部实现的self attention的计算 

## VA-VAE的创新要点
- VAE中的latent space和视觉基础模型（DINOv2等）对齐
- 用DiT实现去噪网络，输出通过逆过程在latent space和error真值算loss

## References
- https://zhuanlan.zhihu.com/p/1938959743377974787
