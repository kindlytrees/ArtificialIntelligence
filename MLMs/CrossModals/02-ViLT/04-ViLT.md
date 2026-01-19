# ViLT

- 跨模态的语言实现中，视觉token和文本token进行拼接后基于transforme计算和视觉和文本进行cross attention的计算两者的区别以及性能的影响？拼接的计算量和参数量都要大一些，性能上提升呢？

- modal embedding 

$$
\begin{aligned}
& t \in \mathbb{R}^{L \times|V|} \quad T \in \mathbb{R}^{|V| \times H} \\
& \bar{t} \in \mathbb{R}^{L \times H} \quad T^{\mathrm{pos}} \in \mathbb{R}^{(L+1) \times H} \\
& I \in \mathbb{R}^{C \times H \times W} \\
& v \in \mathbb{R}^{N \times\left(P^2 \cdot C\right)} \quad N=H W / P^2 \\
& V \in \mathbb{R}^{\left(P^2 \cdot C\right) \times H} \quad \bar{v} \in \mathbb{R}^{N \times H}
\end{aligned}
$$

- 模型结构描述

$$
\begin{array}{rlr}
\bar{t} & =\left[t_{\text {class }} ; t_1 T ; \cdots ; t_L T\right]+T^{\mathrm{pos}} & \\
\bar{v} & =\left[v_{\text {class }} ; v_1 V ; \cdots ; v_N V\right]+V^{\mathrm{pos}} & \\
z^0 & =\left[\bar{t}+t^{\text {type }} ; \bar{v}+v^{\text {type }}\right] & \\
\hat{z}^d & =\operatorname{MSA}\left(\operatorname{LN}\left(z^{d-1}\right)\right)+z^{d-1}, & d=1 \ldots D \\
z^d & =\operatorname{MLP}\left(\operatorname{LN}\left(\hat{z}^d\right)\right)+\hat{z}^d, & d=1 \ldots D \\
p & =\tanh \left(z_0^D W_{\text {pool }}\right) &
\end{array}
$$

sequence $z^D . p$ is a pooled representation of the whole multimodal input, and is obtained by applying linear projection $W_{\text {pool }} \in \mathbb{R}^{H \times H}$ and hyperbolic tangent upon the first index of sequence $z^D$.

- 请具体描述IPOT的ITMLoss的计算细节，如Wasserstein距离以及相关传输矩阵的定义和用法等
ipot计算了传输矩阵，并提前做了转置，以便后续和成本矩阵相乘，其结果的对角线上的值为对应的token传输所带来的loss

- ipot实现分析解读,计算成本矩阵 C 对应的最优传输矩阵 T

```
@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T
```

## 参考资料
- RT-1训练数据：https://console.cloud.google.com/storage/browser/gresearch/rt-1-data-release
- https://github.com/google-research/robotics_transformer
- https://robotics-transformer2.github.io/
- https://github.com/dandelin/ViLT
- https://visualqa.org/download.html
