# PCA

## PCA的逆变换过程

- 将归一化的数据和 k 个主成分（投影矩阵）进行计算得出PCA后的数据Z
- 将Z乘以投影矩阵的转置恢复到原始数据空间维度的有损失的归一化后数据 $Z=X_{\text {norm }} V_k$
- 同时右乘$V_k^T$，将数据进一步进行反Z－score normalizationd的操作恢复到原始的数据语义空间 $X_{\mathrm{norm}}^{\prime}=Z V_k^T$, 可以看成只在前面k个子空间里

## SVD

- SVD中的v向量是A^TA的特征向量的时候，其V的正交向量基通过A变换即可变换为另一个不同长度维度的正交向量基U
- SVD中的A^TA方阵不是满秩矩阵，其不为半正定的矩阵，但仍然可以基于特征值特征向量的方式来进行求解对吗？

摘自大模型回答：

SVD的巧妙之处：
我们想理解一个可能很复杂的变换 A（它可能会旋转、拉伸、甚至改变维度）。
直接分析 A 可能很困难。所以我们构造一个“更好”的矩阵 AᵀA。
这个 AᵀA 矩阵有非常好的性质：
- 是方阵（不改变维度）
- 它是对称的（保证特征向量相互正交）
- 它对自己的特征向量 vᵢ 的作用很简单：只有纯粹的缩放，没有旋转
- 通过找到这些在 AᵀA 变换下方向不变的“特殊”正交向量 vᵢ（它们构成了输入空间的一组完美的坐标轴），我们就有了一把“尺子”

$$
\begin{aligned}
& \left\|A v_i\right\|^2=\left(A v_i\right)^T *\left(A v_i\right) \\
& \Rightarrow\left\|A v_i\right\|^2=v_i^T A^T A v_i \\
& \Rightarrow\left\|A v_i\right\|^2=\lambda_i v_i^T v_i=\lambda_i \\
& \therefore\left\|A v_i\right\|=\sqrt{\lambda_i}
\end{aligned}
$$

$$
A\left(v_1, v_2, \ldots, v_k \mid v_{k+1}, v_{k+2}, \ldots, v_n\right)=\left(u_1, u_2, \ldots, u_k \mid u_{k+1}, u_{k+2}, \ldots, u_m\right)\left(\begin{array}{ccccc}
\sigma_1 & & & & \\
& \sigma_2 & & & \\
& \ddots & & & \\
& & \sigma_k & & \\
& & & 0 & \\
& & & \ddots & \\
& & & & 0
\end{array}\right)
$$