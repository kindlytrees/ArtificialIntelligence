# Stable Diffusion 3

## 问题1：mmdit架构的image token和text token通过拼接串联起来实现的模态内部和模态之间的这种attention计算，而不是只采用cross attention的计算，这样其信息的交互的计算就更加全面和丰富，可以这么理解吗? 之前好像也说到模态之间通过cross attention计算是transformer架构的更主要的模式? 请对该内容做更多详细些的说明。

## 问题2：请对rectified flow算法做一下介绍，并和ddpm的扩散模型躲一下对比说明，rectified flow是否也属于扩散模型呢，是不是stable diffusion 3只是沿用的名称版本而已？

## 问题3：速度场对于不同时刻的输入，输出的每个二维空间位置的变化量，模型由于在中间时间位置相对难以学习，因此其采用非均匀采样？模型学习完成之后，推理的时候时间的步骤的不同对结果有多大的影响？如果从积分公式来看，好像是等价的？