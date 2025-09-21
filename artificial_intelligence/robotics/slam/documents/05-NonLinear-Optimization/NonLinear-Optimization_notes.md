# NonLinear Optimization notes

## Outline
- 问题1：非线性最小二乘和一般的最小二乘的求解方法，主要是迭代的过程中一个采用梯度下降，一个采用比梯度计算更优的方向进行参数更新，为什么不能直接求解析解，关键在于其过程是非线性方程对吗？优化目标定义为非线性函数（和线性回归中的最小二乘问题进行类比，线性回归中定义的函数为线性函数，参数变量空间和数据空间为线性关系，而在李群李代数表示的运动与观测方程中，观测结果，如图像中像素点与数据空间的关系为非线性的关系，表现为一，成像模型中要除以深度z，为非线性操作，二，欧式变换为约束高的SE(3)参数变量空间，其中的旋转变换约束为正交基且行列式为1的李群流形，或者其对应为的李代数的指数映射，也可以看成是对应到罗德里格斯公式去理解，其中的theta角度为李代数的模长，因此也引入了非线性）

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

---

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


### 问题2：非线性最小二乘和一般的最小二乘的求解方法，主要是迭代的过程中一个采用梯度下降，一个采用比梯度计算更优的方向进行参数更新，为什么不能直接求解析解，关键在于其过程是非线性方程对吗？

辅助大模型回答：

您的问题提得非常精准，完全抓住了线性与非线性最小二乘问题的核心区别。您的断言**“关键在于其过程是非线性方程”是完全正确的**。

我们来详细剖析一下这个问题。

---

### 1. 线性最小二乘 (Linear Least Squares) - 为什么有解析解？

我们回顾一下线性最小二乘问题：
*   **模型**: $ \mathbf{y} = H\mathbf{x} $ (暂时忽略噪声)
*   **目标函数**: 最小化误差的平方和 $J(\mathbf{x}) = ||\mathbf{y} - H\mathbf{x}||^2$

**关键点**：这个目标函数 $J(\mathbf{x})$ 是一个关于未知参数 $\mathbf{x}$ 的**二次函数**。
我们可以展开它：
$ J(\mathbf{x}) = (\mathbf{y} - H\mathbf{x})^T(\mathbf{y} - H\mathbf{x}) = \mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T H\mathbf{x} + \mathbf{x}^T H^T H \mathbf{x} $

这是一个标准的向上开口的抛物面（在多维空间中）。这种函数有唯一的全局最小值。为了找到这个最小值，我们求它对 $\mathbf{x}$ 的导数（梯度）并令其为零：
$ \frac{\partial J(\mathbf{x})}{\partial \mathbf{x}} = -2H^T\mathbf{y} + 2H^T H \mathbf{x} $

令导数为零：
$ -2H^T\mathbf{y} + 2H^T H \mathbf{x} = \mathbf{0} $
$ \implies H^T H \mathbf{x} = H^T\mathbf{y} $

这个方程组被称为**正规方程 (Normal Equations)**。最重要的一点是，这是一个关于 $\mathbf{x}$ 的**线性方程组**。解一个线性方程组 $A\mathbf{x}=\mathbf{b}$ 是非常直接的，只要 $A$ 可逆，就有唯一的解析解 $\mathbf{x} = A^{-1}\mathbf{b}$。

因此，线性最小二乘的解析解为：
$ \mathbf{x}^* = (H^T H)^{-1} H^T \mathbf{y} $

**结论**：线性最小二乘有解析解，是因为其目标函数是二次的，导致其导数为**线性的**，从而可以一步求解。

---

### 2. 非线性最小二乘 (Non-linear Least Squares) - 为什么没有解析解？

现在我们来看非线性最小二乘问题：
*   **模型**: $ \mathbf{y} = f(\mathbf{x}) $，其中 $f$ 是一个关于 $\mathbf{x}$ 的**非线性函数**。
*   **目标函数**: $J(\mathbf{x}) = ||\mathbf{y} - f(\mathbf{x})||^2$

**关键点**：由于 $f(\mathbf{x})$ 是非线性的，目标函数 $J(\mathbf{x})$ 通常是一个复杂的、**非二次的**函数。它可能有多个局部最小值、平坦区域或鞍点。

我们同样尝试求导并令其为零来寻找最小值：
$ \frac{\partial J(\mathbf{x})}{\partial \mathbf{x}} = \frac{\partial}{\partial \mathbf{x}} \left( (\mathbf{y} - f(\mathbf{x}))^T (\mathbf{y} - f(\mathbf{x})) \right) $

使用链式法则，我们得到：
$ \frac{\partial J(\mathbf{x})}{\partial \mathbf{x}} = -2 \left( \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \right)^T (\mathbf{y} - f(\mathbf{x})) $

这里的 $\frac{\partial f(\mathbf{x})}{\partial \mathbf{x}}$ 就是**雅可比矩阵 (Jacobian)**，我们记作 $J_f(\mathbf{x})$。

令导数为零：
$ J_f(\mathbf{x})^T (\mathbf{y} - f(\mathbf{x})) = \mathbf{0} $

**这就是问题的核心症结所在！**
这是一个关于 $\mathbf{x}$ 的**非线性方程组**。
1.  $f(\mathbf{x})$ 本身是关于 $\mathbf{x}$ 的非线性函数。
2.  雅可比矩阵 $J_f(\mathbf{x})$ 的元素通常也依赖于 $\mathbf{x}$。

我们无法像线性情况那样，通过简单的矩阵逆运算一步就解出 $\mathbf{x}$。想象一下解一个简单的非线性方程，比如 $\cos(x) + x^2 = 5$，你无法直接写出 $x = \dots$ 的解析形式。多维非线性方程组的求解更是如此。

**结论**：非线性最小二乘没有解析解，是因为其目标函数是非二次的，导致其导数为**非线性的**，无法直接求解。

---

### 3. 迭代求解：用线性方法逼近非线性问题

既然无法一步求解，我们就必须采用迭代的方法，从一个初始猜测 $\mathbf{x}_0$ 开始，一步步地逼近最优解 $\mathbf{x}^*$。

**核心思想**：在每一步迭代中，我们将复杂的非线性问题在当前点附近**线性化**，将其近似为一个简单的**线性最小二乘问题**来求解一个“更新方向”。

1.  **线性化**：在当前估计值 $\mathbf{x}_k$ 附近，我们用一阶泰勒展开来近似非线性函数 $f(\mathbf{x})$：
    $ f(\mathbf{x}) \approx f(\mathbf{x}_k) + J_f(\mathbf{x}_k)(\mathbf{x} - \mathbf{x}_k) $
    令更新量为 $\Delta\mathbf{x} = \mathbf{x} - \mathbf{x}_k$。

2.  **构建局部线性最小二乘问题**：将上述近似代入原始目标函数：
    $ J(\mathbf{x}_k + \Delta\mathbf{x}) \approx ||\mathbf{y} - (f(\mathbf{x}_k) + J_f(\mathbf{x}_k)\Delta\mathbf{x})||^2 $
    $ = ||(\mathbf{y} - f(\mathbf{x}_k)) - J_f(\mathbf{x}_k)\Delta\mathbf{x}||^2 $
    这现在是一个关于**更新量 $\Delta\mathbf{x}$** 的标准**线性最小二乘问题**！它的形式正是 $ ||\mathbf{b} - A\mathbf{z}||^2 $，其中 $\mathbf{b} = \mathbf{y} - f(\mathbf{x}_k)$, $A = J_f(\mathbf{x}_k)$, $\mathbf{z} = \Delta\mathbf{x}$。

3.  **求解更新方向**：我们可以直接用线性最小二乘的解析解来求解这个局部问题，得到最优的更新量 $\Delta\mathbf{x}$：
    $ (J_f^T J_f) \Delta\mathbf{x} = J_f^T (\mathbf{y} - f(\mathbf{x}_k)) $
    这个方程正是**高斯-牛顿法 (Gauss-Newton method)** 的核心。

4.  **更新参数**：得到 $\Delta\mathbf{x}$ 后，更新我们的估计：
    $ \mathbf{x}_{k+1} = \mathbf{x}_k + \Delta\mathbf{x} $

5.  **重复**：回到第一步，直到 $\Delta\mathbf{x}$ 足够小或目标函数不再下降。

### 为什么比梯度下降更优？

*   **梯度下降 (Gradient Descent)**：只使用一阶梯度信息。它的更新方向是 $ \Delta\mathbf{x} \propto -\nabla J(\mathbf{x}) = J_f(\mathbf{x})^T (\mathbf{y} - f(\mathbf{x})) $。它告诉你最陡峭的下降方向，但没有考虑“走多远”以及“地形的曲率”。
*   **高斯-牛顿法**：它的更新方向是 $ \Delta\mathbf{x} = (J_f^T J_f)^{-1} J_f^T (\mathbf{y} - f(\mathbf{x}_k)) $。这里的 $(J_f^T J_f)^{-1}$ 是对目标函数二阶导数（Hessian矩阵）的近似。它不仅告诉了我们方向，还利用了曲率信息来估计一个更好的步长，因此收敛速度通常远快于梯度下降。**Levenberg-Marquardt** 算法则是高斯-牛顿法的一个更稳健的改进版。

### 总结

| 特性 | 线性最小二乘 (LLS) | 非线性最小二乘 (NLS) |
| :--- | :--- | :--- |
| **模型** | $\mathbf{y} = H\mathbf{x}$ (线性) | $\mathbf{y} = f(\mathbf{x})$ (非线性) |
| **目标函数** | $J(\mathbf{x}) = ||\mathbf{y} - H\mathbf{x}||^2$ (二次型) | $J(\mathbf{x}) = ||\mathbf{y} - f(\mathbf{x})||^2$ (非二次型) |
| **导数 = 0 的方程** | $H^T H \mathbf{x} = H^T\mathbf{y}$ (**线性方程组**) | $J_f(\mathbf{x})^T (\mathbf{y} - f(\mathbf{x})) = \mathbf{0}$ (**非线性方程组**) |
| **可解性** | **有解析解** | **无通用解析解** |
| **求解方法** | 直接求解正规方程 | 迭代求解 (如高斯-牛顿法, Levenberg-Marquardt) |