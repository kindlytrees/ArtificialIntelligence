# VGGT notes

## VGGT的关键思路和实现要点

- 支持多帧场景，heads有
  - camera pose的head
  - dpt的head(dense prediction head)
    - 深度信息预测
    - 三维点信息预测
    - point tracking
- 多个帧之间保持sequence-agnostic
- camera的位姿变化相对于第一帧
- 其他的帧会融合如相关帧的信息，但是单独预测dense的结果(dpt layer后接下游任务的head)

$$
D_i(\mathbf{y}) \in \mathbb{R}^{+}
$$

$$
P_i(\mathbf{y}) \in \mathbb{R}^3
$$

## VGGT中的camera token，registers tokens的作用，为什么第一帧算一类token，其他的帧算另一类tokens，camera的预测结果是相对于第一帧的吗？

其主要用camera token进行区分，其他的camera数据都是相对于第一帧的数据的camera位姿的不变化，至于其他信息就没有相对性了，以及gt也是相对于第一帧的标准坐标系构建的，register token的作用是什么？

但是论文中有如下的描述，好像说有一个坐标帧作为第一帧？the camera extrinsics output for the first camera are set to the identity, i.e., the first rotation quaternion is q1 = [0, 0, 0, 1] and the first translation vector is t1 = [0, 0, 0].
the special camera and register tokens allow the transformer to identify the first camera.

基于第一帧作为相对帧得出相机的位姿变换更加相对位置的学习更加方便
但其他的dense head的输出真值是否基于自己帧的结果就可以，而不需要和第一帧进行相对计算？

描述介绍摘自大模型：

## vggt实现cotracker相当于用vggt做tracker的视觉特征提取backbone，这时的任务也将以帧的顺序输入以便cotrakcer中保持时序性，其他的depth以及semantic和camera的位姿变化预测的功能保持不变，这样的功能是否是很好的slam的前端？是否基于此的准确性是否不需要slam的后端了呢

大模型给出的答案总结： [VGGT + CoTracker 超强前端] + [高效的后端优化器 (如g2o, GTSAM, Ceres)] = 顶级SLAM系统，但当前也有不少工程上的限制和挑战

## vggt中camera token，register tokens的作用介绍，以及实现方法，为什么区分第一帧和其他帧，以及在vggt训练的时候基于的不同的数据集的特性，如场景中图片的数量大小，训练过程是否也存在着支持的图片的多少的问题，比如某个数据集场景中有30张图片，有的数据集中只有几张，在训练的时候是否每个iteration固定到数据集上比较合适，或者数据集中也存在着场景中图片多少不一样的问题，其如何解决这种问题的呢，如果图片量差异较大，通过padding和mask会存在着计算量较大的问题？

摘自大模型的回答：

在实践中，像VGGT这样的模型通常会采用混合策略：
设定一个硬件允许的最大视图数 N_max。
对于图片数 > N_max 的场景，进行随机采样。
对于图片数 <= N_max 的场景，采用分桶 + 动态填充与掩码的策略来构建batch进行高效训练。

## vggt中如何融合text的信息？实现visual ground的功能？
