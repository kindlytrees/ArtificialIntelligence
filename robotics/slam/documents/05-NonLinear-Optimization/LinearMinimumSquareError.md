# 线性最小二乘


- 问题2：假设y和H，协方差已知，请给出如下的基于多维高斯分布的最小二乘x的解析解。

$$
\begin{gathered}
P\left(\boldsymbol{z}_{j, k} \mid \boldsymbol{x}_k, \boldsymbol{y}_j\right)=N\left(h\left(\boldsymbol{y}_j, \boldsymbol{x}_k\right), \boldsymbol{Q}_{k, j}\right)  \\
P(\boldsymbol{x})=\frac{1}{\sqrt{(2 \pi)^N \operatorname{det}(\boldsymbol{\Sigma})}} \exp \left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\mathrm{T}} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)\\ 
{(x,y)=\arg \min \left(\left(\boldsymbol{z}_{k, j}-h\left(\boldsymbol{x}_k, \boldsymbol{y}_j\right)\right)^{\mathrm{T}} \boldsymbol{Q}_{k, j}^{-1}\left(\boldsymbol{z}_{k, j}-h\left(\boldsymbol{x}_k, \boldsymbol{y}_j\right)\right)\right)}
\end{gathered}
$$

### 问题1：假设y和H，协方差已知，请给出如下的基于多维高斯分布的最小二乘x的解析解。

$(\vec{y}-H \vec{x})^{\top} \Sigma^{-1}(\vec{y}-H \vec{x})$

辅助大模型回答：

好的，我们来详细推导这个问题。这是一个非常经典的**加权最小二乘（Weighted Least Squares, WLS）**问题，其解在统计学、机器学习、机器人学（状态估计、SLAM）等领域有广泛应用。当噪声服从高斯分布时，最小二乘解等价于**最大似然估计（Maximum Likelihood Estimation, MLE）**

---

### 1. 问题设定

我们有一个线性观测模型：

$\mathbf{y} = H\mathbf{x} + \mathbf{v}$

其中：
*   $\mathbf{x} \in \mathbb{R}^n$ 是我们想要估计的**未知状态向量**。
*   $\mathbf{y} \in \mathbb{R}^m$ 是**观测向量**（已知）。
*   $H \in \mathbb{R}^{m \times n}$ 是**观测矩阵**（已知），它将状态空间映射到观测空间。
*   $\mathbf{v} \in \mathbb{R}^m$ 是**观测噪声**，是一个随机向量。

我们被告知，噪声 $\mathbf{v}$ 服从一个零均值的多维高斯分布，其协方差矩阵为 $\Sigma$（已知）：$\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \Sigma)$

其中 $\Sigma \in \mathbb{R}^{m \times m}$ 是一个对称正定矩阵。

由此可得，观测向量 $\mathbf{y}$ 也服从一个高斯分布：$\mathbf{y} \sim \mathcal{N}(H\mathbf{x}, \Sigma)$

我们的目标是：给定已知的 $\mathbf{y}$, $H$ 和 $\Sigma$，找到一个最优的 $\mathbf{x}^*$，使得这个观测 $\mathbf{y}$ 出现的**概率最大**。

---

### 2. 方法一：最大似然估计 (MLE)

最大似然估计的目标是找到一个参数 $\mathbf{x}$，使得给定该参数时，观测数据 $\mathbf{y}$ 出现的概率（即似然函数）最大。

**第一步：写出似然函数**

$\mathbf{y}$ 的概率密度函数（PDF）就是我们的似然函数 $L(\mathbf{x})$:
$L(\mathbf{x}) = p(\mathbf{y} | \mathbf{x}; H, \Sigma) = \frac{1}{\sqrt{(2\pi)^m \det(\Sigma)}} \exp\left( -\frac{1}{2} (\mathbf{y} - H\mathbf{x})^T \Sigma^{-1} (\mathbf{y} - H\mathbf{x}) \right)$

**第二步：取对数并简化**

为了方便求导，我们通常最大化对数似然函数 $\log L(\mathbf{x})$，这与最大化 $L(\mathbf{x})$ 是等价的，因为对数函数是单调递增的。

$\log L(\mathbf{x}) = \log\left(\frac{1}{\sqrt{(2\pi)^m \det(\Sigma)}}\right) - \frac{1}{2} (\mathbf{y} - H\mathbf{x})^T \Sigma^{-1} (\mathbf{y} - H\mathbf{x})$

第一项是与 $\mathbf{x}$ 无关的常数。因此，最大化 $\log L(\mathbf{x})$ 等价于**最小化**下面这个二次型（也称为马氏距离的平方）：

$J(\mathbf{x}) = (\mathbf{y} - H\mathbf{x})^T \Sigma^{-1} (\mathbf{y} - H\mathbf{x})$

**第三步：求导并令其为零**

为了找到使 $J(\mathbf{x})$ 最小的 $\mathbf{x}$，我们计算 $J(\mathbf{x})$ 对 $\mathbf{x}$ 的梯度，并令其等于零。

首先展开 $J(\mathbf{x})$:
$J(\mathbf{x}) = (\mathbf{y}^T - \mathbf{x}^T H^T) \Sigma^{-1} (\mathbf{y} - H\mathbf{x})$

$J(\mathbf{x}) = \mathbf{y}^T \Sigma^{-1} \mathbf{y} - \mathbf{y}^T \Sigma^{-1} H \mathbf{x} - \mathbf{x}^T H^T \Sigma^{-1} \mathbf{y} + \mathbf{x}^T H^T \Sigma^{-1} H \mathbf{x}$

注意到标量项 $\mathbf{x}^T H^T \Sigma^{-1} \mathbf{y}$ 等于它的转置 $\mathbf{y}^T (\Sigma^{-1})^T H \mathbf{x}$。因为 $\Sigma$ 是对称的，所以 $\Sigma^{-1}$ 也是对称的，即 $(\Sigma^{-1})^T = \Sigma^{-1}$。因此：
$\mathbf{x}^T H^T \Sigma^{-1} \mathbf{y} = \mathbf{y}^T \Sigma^{-1} H \mathbf{x}$

所以，
$J(\mathbf{x}) = \mathbf{y}^T \Sigma^{-1} \mathbf{y} - 2 \mathbf{y}^T \Sigma^{-1} H \mathbf{x} + \mathbf{x}^T H^T \Sigma^{-1} H \mathbf{x}$

现在求梯度。我们使用两个矩阵求导的结论：
1.  $\frac{\partial (\mathbf{a}^T \mathbf{x})}{\partial \mathbf{x}} = \mathbf{a}$
2.  $\frac{\partial (\mathbf{x}^T A \mathbf{x})}{\partial \mathbf{x}} = 2 A \mathbf{x}$ (当 A 是对称矩阵时)

在我们的问题中，$H^T \Sigma^{-1} H$ 是对称的。因此：
$\frac{\partial J(\mathbf{x})}{\partial \mathbf{x}} = \frac{\partial}{\partial \mathbf{x}} (\mathbf{y}^T \Sigma^{-1} \mathbf{y}) - \frac{\partial}{\partial \mathbf{x}} (2 \mathbf{y}^T \Sigma^{-1} H \mathbf{x}) + \frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T H^T \Sigma^{-1} H \mathbf{x})$

$\frac{\partial J(\mathbf{x})}{\partial \mathbf{x}} = \mathbf{0} - 2 (2 \mathbf{y}^T \Sigma^{-1} H)^T + 2 (H^T \Sigma^{-1} H) \mathbf{x}$

$\frac{\partial J(\mathbf{x})}{\partial \mathbf{x}} = -2 H^T \Sigma^{-1} \mathbf{y} + 2 (H^T \Sigma^{-1} H) \mathbf{x}$

令梯度为零：
$-2 H^T \Sigma^{-1} \mathbf{y} + 2 (H^T \Sigma^{-1} H) \mathbf{x} = \mathbf{0}$

$(H^T \Sigma^{-1} H) \mathbf{x} = H^T \Sigma^{-1} \mathbf{y}$

**第四步：求解 $\mathbf{x}$**

这个方程组称为**正规方程（Normal Equations）**。如果矩阵 $(H^T \Sigma^{-1} H)$ 是可逆的（通常当 $H$ 是列满秩且 $m \ge n$ 时成立），我们可以直接求解 $\mathbf{x}$。

$\mathbf{x}^* = (H^T \Sigma^{-1} H)^{-1} H^T \Sigma^{-1} \mathbf{y}$

这就是基于多维高斯分布的最小二乘问题的解析解。


### 3. 方法二：几何投影（加权最小二乘视角）

我们可以将这个问题看作是将向量 $\mathbf{y}$ 投影到一个由矩阵 $H$ 的列向量张成的子空间上，但这个投影是在一个由 $\Sigma^{-1}$ 定义的“加权”内积空间中进行的。

**第一步：数据白化 (Whitening)**

由于噪声的协方差不是单位矩阵 $I$，不同维度的噪声方差不同，且可能存在相关性。我们可以通过一个线性变换来“白化”噪声，使其协方差变为单位矩阵。

因为 $\Sigma$ 是对称正定的，我们可以对其进行 Cholesky 分解或特征值分解，得到 $\Sigma = L L^T$（或 $\Sigma^{-1} = L' (L')^T$）。我们定义白化矩阵 $W = L^{-1}$，则 $W^T W = (L^{-1})^T L^{-1} = (L L^T)^{-1} = \Sigma^{-1}$。

将原始方程 $ \mathbf{y} = H\mathbf{x} + \mathbf{v} $ 两边同时左乘白化矩阵 $W$：
$ W\mathbf{y} = W H \mathbf{x} + W\mathbf{v} $

令 $\mathbf{y}' = W\mathbf{y}$, $H' = WH$, $\mathbf{v}' = W\mathbf{v}$。
新的模型是：
$\mathbf{y}' = H' \mathbf{x} + \mathbf{v}'$

现在我们来考察新噪声 $\mathbf{v}'$ 的协方差：
$\text{Cov}(\mathbf{v}') = E[\mathbf{v}' (\mathbf{v}')^T] = E[W\mathbf{v} (W\mathbf{v})^T] = E[W\mathbf{v}\mathbf{v}^T W^T]$
$ = W E[\mathbf{v}\mathbf{v}^T] W^T = W \Sigma W^T = W (L L^T) W^T = (W L)(W L)^T $
因为 $W=L^{-1}$，所以 $WL=I$。
$ = I I^T = I $

**第二步：应用标准最小二乘**

现在我们有了一个新的线性模型，其噪声协方差是单位矩阵 $I$。这正是**标准（或普通）最小二乘（Ordinary Least Squares, OLS）**的设定。其目标函数是最小化误差的欧几里得范数的平方：

$ J(\mathbf{x}) = || \mathbf{y}' - H' \mathbf{x} ||^2 = (\mathbf{y}' - H' \mathbf{x})^T (\mathbf{y}' - H' \mathbf{x}) $

标准最小二乘的解是众所周知的：
$ \mathbf{x}^* = ((H')^T H')^{-1} (H')^T \mathbf{y}' $

**第三步：代回原始变量**

现在我们将 $\mathbf{y}' = W\mathbf{y}$ 和 $H' = WH$ 代回上式：
$ \mathbf{x}^* = ((WH)^T (WH))^{-1} (WH)^T (W\mathbf{y}) $
$ = (H^T W^T W H)^{-1} H^T W^T W \mathbf{y} $

因为我们定义了 $W^T W = \Sigma^{-1}$，所以：
$ \mathbf{x}^* = (H^T \Sigma^{-1} H)^{-1} H^T \Sigma^{-1} \mathbf{y} $

我们得到了与最大似然估计完全相同的解析解。

---

### 总结与解释

**解析解**:
$ \mathbf{x}^* = (H^T \Sigma^{-1} H)^{-1} H^T \Sigma^{-1} \mathbf{y} $

**解释**:
*   **信息矩阵**: $H^T \Sigma^{-1} H$ 被称为**信息矩阵**。它融合了观测模型 ($H$) 和噪声不确定性 ($\Sigma$) 的信息。
*   **协方差的作用**: $\Sigma^{-1}$ 起到了**加权**的作用。如果某个观测维度 $y_i$ 的方差 $\sigma_i^2$ 很小（即这个观测很精确），那么 $\Sigma^{-1}$ 中对应的对角元素就会很大，从而在计算中给予这个观测更高的权重。反之，噪声大的观测权重就小。
*   **与标准最小二乘的对比**: 如果所有噪声独立同分布且方差为 $\sigma^2$，即 $\Sigma = \sigma^2 I$，那么 $\Sigma^{-1} = \frac{1}{\sigma^2} I$。代入解中：
    $ \mathbf{x}^* = (H^T (\frac{1}{\sigma^2} I) H)^{-1} H^T (\frac{1}{\sigma^2} I) \mathbf{y} = (\frac{1}{\sigma^2} H^T H)^{-1} (\frac{1}{\sigma^2} H^T \mathbf{y}) $
    $ = (\sigma^2 (H^T H)^{-1}) (\frac{1}{\sigma^2} H^T \mathbf{y}) = (H^T H)^{-1} H^T \mathbf{y} $
    这正是标准最小二乘的解。这说明标准最小二乘是加权最小二乘在噪声协方差为单位矩阵（或其标量倍）时的特例。