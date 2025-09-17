# 牛顿法

## 相关公式及推导

- 参数w的二阶导数计算，hessian矩阵的具体计算推导，其中$\left[x_i\left(y_i-\sigma\left(\theta^T x_i\right)\right)\right]$的内容为一阶导数

$$
L(\theta)=-\ell(\theta)=-\sum_{i=1}^n\left[y_i \log \sigma\left(\theta^T x_i\right)+\left(1-y_i\right) \log \left(1-\sigma\left(\theta^T x_i\right)\right)\right]
$$

$$
\begin{aligned}
& \frac{\partial }{\partial \theta_j}L(\theta)=-\sum_{i=1}^n (y_i \frac {1} {\sigma( \theta ^T x_i)} -(1-y_i) \frac {1}{1-\sigma(\theta ^Tx_i)})\frac {\partial} {\partial \theta_j}g(\theta ^T x_i) \\
= & -\sum_{i=1}^n (y \frac {1} {\sigma( \theta ^T x_i)} -(1-y) \frac {1}{1-\sigma(\theta ^Tx_i)})\sigma(\theta ^T x_i)(1-\sigma(\theta ^Tx_i))\frac {\partial}{\partial \theta_j} \theta ^Tx_i \\
= & -\sum_{i=1}^n (y_i(1-\sigma(\theta ^Tx_i))-(1-y_i)\sigma(\theta ^T x_i))x_i^{(j)} \\
= & -\sum_{i=1}^n (y_i-\sigma(\theta ^Tx_i))x_i^{(j)} \\
\end{aligned}
$$


$\nabla_{\theta} \left[x_i\left(y_i-\sigma\left(\theta^T x_i\right)\right)\right]=-x_i x_i^T \sigma\left(\theta^T x_i\right)\left(1-\sigma\left(\theta^T x_i\right)\right) \quad H(\theta)=\sum_{i=1}^n x_i x_i^T \sigma\left(\theta^T x_i\right)\left(1-\sigma\left(\theta^T x_i\right)\right)$

$$
\begin{aligned}
& \nabla_{\theta} \left[x_i\left(y_i-\sigma\left(\theta^{\top} x_i\right)\right)\right] \\
= & \nabla_{\theta} \left[-x_i \sigma\left(\theta^{\top} x_i\right)\right] \\
= & -x_i \sigma\left(\theta^{\top} x_i\right)\left(1-\sigma\left(\theta^{\top} x_i\right)\right) \nabla_{\theta} \theta^{\top} x_i \\
= & -x_i x_i^{\top} \sigma\left(\theta^{\top} x_i\right)\left(1-\sigma\left(\theta^{\top} x_i\right)\right.
\end{aligned}
$$

$$
H(\theta)=X R X^T \quad R=\operatorname{diag}\left(\sigma\left(\theta^T x_i\right)\left(1-\sigma\left(\theta^T x_i\right)\right)\right)
$$

$$
\theta:=\theta-{H(\theta)}^{-1} \nabla_\theta \ell(\theta)
$$

## 矩阵乘法的含义理解

### 空间几何意义解释

- 第一种解释
  AB矩阵相乘，矩阵A的列向量可以看成是标准基向量通过A矩阵进行变换后的结果，
  如果我们知道变换A是如何作用于基向量的（即的列向量），那么任何向量都可以分解为这些基向量的组合，然后我们用
  作用于这个组合，根据线性变换的性质，结果就是作用于每个基向量的结果的相同组合。
  某个具体的向量经过这个变换后变成了什么，比如左边是旋转矩阵，相当于左了旋转变换
- 第二种解释：
  第二种解释方法：左矩阵的每一行和右矩阵的每一列的相乘所得矩阵分量得和。
  这种方法直接定义了结果矩阵的每一个元素是如何计算出来的。每个元素都代表了左矩阵的第i行与右矩阵的第j列之间的“关联强度”或“投影”。
- 第三种解释：
  矩阵乘法的第三种几何解释，它被称为外积（Outer Product）之和
  单个外积形成的矩阵的秩为1的矩阵,如[m,1]*[1,n]=[m,n]，

矩阵的奇异值分解（SVD）就是这种思想的延伸。SVD 将一个矩阵分解为一系列外积之和，其中每个外积项的贡献由其对应的奇异值衡量。通过只保留最大的几个奇异值对应的外积项，可以得到原矩阵的低秩近似，这在降维、图像压缩、推荐系统等领域有重要应用。

矩阵A的列向量可以看作是“特征基”，而B的行向量则指示了这些特征在输入数据中的“权重”或“存在程度”。通过外积求和，我们能够从这些基本特征中重构出完整的变换或数据。

矩阵乘法可以写作一组＂外积＂的和：

$$
A B=\sum_{k=1}^n A_{\cdot k} \otimes B_{ k\cdot}
$$

－这里的 $A_{. k} \otimes B_{k .}$ 是一个 rank－1 矩阵，大小是 $m \times p$ ，表示将一个列向量 $A_{. k}$ 和行向量 $B_k$ ．的外积拼成一个矩阵。
－几何上，它相当于将每个列向量按某种＂方向张成一个平面＂。