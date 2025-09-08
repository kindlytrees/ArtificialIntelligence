# TRPO notes

## 知识要点
- 优势函数的定义（GAE，Generalized Advantage Estimation, GAE）中使用到了多步时间差分
- KL损失函数定义为Reverse KL divergence
- 转化为基于广义拉格朗日乘数子求解基于二次约束下的线性优化问题
- 采用共轭梯度的方式迭代计算更新方向，采用Hp相结合绕开直接计算Hessian矩阵
- 基于异策略的在线数据采集方式，基于重要性采样机制来实现数据样本的多次复用，提升训练效率
- 代理目标函数定义为重要性ratio乘以优势函数（代码里的compute_surrogate_obj函数），思想和策略梯度的代理损失函数的思想类似

## 具体详细内容

- 目标损失函数

$$
\operatorname{maximize} \mathbb{E}_{\pi_{\theta_{\text {old }}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)} A_{\theta_{\text {old }}}(s, a)\right] \approx g^T\left(\theta-\theta_{old}\right)
$$

- 约束函数

$$
\text { subject to } \mathbb{E}_{\pi_{\theta_{\text {old }}}}\left[D_{\mathrm{KL}}\left(\pi_{\theta_{\text {old }}} \| \pi_\theta\right)\right] \leq \delta \approx \frac{1}{2}\left(\theta-\theta_{old}\right)^T H\left(\theta-\theta_{old}\right)
$$

- 约束优化函数定义为

$$
\begin{aligned}
\operatorname{maximize} \mathbb{E}_{\pi_{\theta_{\text {old }}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text {old }}}(a \mid s)} A_{\theta_{\text {old }}}(s, a)\right] \approx g^T\left(\theta-\theta_{old}\right) \\
\text { subject to } \mathbb{E}_{\pi_{\theta_{\text {old }}}}\left[D_{\mathrm{KL}}\left(\pi_{\theta_{\text {old }}} \| \pi_\theta\right)\right] \leq \delta \approx \frac{1}{2}\left(\theta-\theta_{old}\right)^T H\left(\theta-\theta_{old}\right) \\
\end{aligned}
$$

- 约束优化函数定义简化表示为

$$
\theta=\underset{\theta^{\prime}}{\arg \max } g^T\left(\theta-\theta_{old}\right) \quad \text { s.t. } \quad \frac{1}{2}\left(\theta-\theta_{old}\right)^T H\left(\theta-\theta_{old}\right) \leq \delta
$$

- 目标函数的常数项部分为0的说明

$$
\begin{aligned}
\mathbb{E}_{a \sim \pi_{\theta_k}(\cdot \mid s)}\left[A^{\pi_{\theta_k}}(s, a)\right] & =\mathbb{E}_{a \sim \pi_{\theta_k}(\cdot \mid s)}\left[Q^{\pi_{\theta_k}}(s, a)-V^{\pi_{\theta_k}}(s)\right] \\
& =\mathbb{E}_{a \sim \pi_{\theta_k}(\cdot \mid s)}\left[Q^{\pi_{\theta_k}}(s, a)\right]-\mathbb{E}_{a \sim \pi_{\theta_k}(\cdot \mid s)}\left[V^{\pi_{\theta_k}}(s)\right]
\end{aligned}
$$

## 问题： trpo算法中，目标函数(期望回报最大化）基于重采样机制进行表示，为什么其常数项近似为零（当两个策略的参数相同时），以及基于kl散度约束的目标函数最大化的公式\theta_{k+1}=\theta_k+\sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g如何推到而来？

辅助gemini回答：

好的，我们来详细拆解并推导TRPO（Trust Region Policy Optimization）算法中的这两个关键问题。

### 问题一：为什么目标函数的“常数项”在 $\theta_{new} \approx \theta_{old}$ 时近似为零？

这要从TRPO的目标函数，即“代理目标函数”（Surrogate Objective Function）说起。

**1. 代理目标函数的构建**

TRPO的目标是最大化新策略 $\pi_{\theta}$ 的期望回报 $\eta(\pi_{\theta})$。直接优化这个目标很困难，因为它依赖于新策略产生的状态分布。因此，TRPO借鉴了重要性采样（Importance Sampling）的思想，构建了一个代理目标函数 $L_{\theta_k}(\theta)$，它用来近似策略性能的提升量 $\eta(\theta) - \eta(\theta_k)$。

这个代理目标函数写作：
$$
L_{\theta_k}(\theta) = \mathbb{E}_{s \sim \rho_{\theta_k}, a \sim \pi_{\theta_k}} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a) \right]
$$
其中：
- $\theta_k$ 是当前（旧的）策略参数。
- $\theta$ 是我们想要优化的新策略参数。
- $\rho_{\theta_k}$ 是在旧策略 $\pi_{\theta_k}$ 下产生的状态访问分布。
- $A^{\pi_{\theta_k}}(s, a) = Q^{\pi_{\theta_k}}(s, a) - V^{\pi_{\theta_k}}(s)$ 是在旧策略下的优势函数（Advantage Function）。

**2. “常数项”的来源与证明**

你所说的“常数项”其实是指当新旧策略相同时，即 $\theta = \theta_k$ 时，代理目标函数 $L_{\theta_k}(\theta)$ 的值。我们来计算一下这个值：

当 $\theta = \theta_k$ 时，策略比率 $\frac{\pi_{\theta_k}(a|s)}{\pi_{\theta_k}(a|s)} = 1$。
此时，代理目标函数变为：
$$
L_{\theta_k}(\theta_k) = \mathbb{E}_{s \sim \rho_{\theta_k}, a \sim \pi_{\theta_k}} \left[ 1 \cdot A^{\pi_{\theta_k}}(s, a) \right]
$$

现在我们来展开这个期望。这个期望是针对状态 $s$ 和动作 $a$ 的。我们可以先对给定状态 $s$ 的动作 $a$ 求期望，然后再对状态 $s$ 求期望：
$$
L_{\theta_k}(\theta_k) = \mathbb{E}_{s \sim \rho_{\theta_k}} \left[ \mathbb{E}_{a \sim \pi_{\theta_k}( \cdot | s)} \left[ A^{\pi_{\theta_k}}(s, a) \right] \right]
$$

我们来看内部的期望 $\mathbb{E}_{a \sim \pi_{\theta_k}( \cdot | s)} \left[ A^{\pi_{\theta_k}}(s, a) \right]$。根据优势函数的定义 $A^{\pi_{\theta_k}}(s, a) = Q^{\pi_{\theta_k}}(s, a) - V^{\pi_{\theta_k}}(s)$：
$$
\begin{aligned}
\mathbb{E}_{a \sim \pi_{\theta_k}( \cdot | s)} \left[ A^{\pi_{\theta_k}}(s, a) \right] &= \mathbb{E}_{a \sim \pi_{\theta_k}( \cdot | s)} \left[ Q^{\pi_{\theta_k}}(s, a) - V^{\pi_{\theta_k}}(s) \right] \\
&= \mathbb{E}_{a \sim \pi_{\theta_k}( \cdot | s)} \left[ Q^{\pi_{\theta_k}}(s, a) \right] - \mathbb{E}_{a \sim \pi_{\theta_k}( \cdot | s)} \left[ V^{\pi_{\theta_k}}(s) \right]
\end{aligned}
$$

- 根据状态价值函数 $V$ 和动作价值函数 $Q$ 的定义，在策略 $\pi_{\theta_k}$ 下，状态 $s$ 的价值等于在该状态下遵循该策略所能获得的期望动作价值。即：
  $V^{\pi_{\theta_k}}(s) = \sum_{a} \pi_{\theta_k}(a|s) Q^{\pi_{\theta_k}}(s, a) = \mathbb{E}_{a \sim \pi_{\theta_k}( \cdot | s)} \left[ Q^{\pi_{\theta_k}}(s, a) \right]$。
- $V^{\pi_{\theta_k}}(s)$ 本身不依赖于动作 $a$，所以对 $a$ 求期望后它仍然是 $V^{\pi_{\theta_k}}(s)$。

将这两点代入，我们得到：
$$
\mathbb{E}_{a \sim \pi_{\theta_k}( \cdot | s)} \left[ A^{\pi_{\theta_k}}(s, a) \right] = V^{\pi_{\theta_k}}(s) - V^{\pi_{\theta_k}}(s) = 0
$$

因为对于任何状态 $s$，内部期望都为零，所以对所有状态 $s$ 再求期望，结果仍然是零。
$$
L_{\theta_k}(\theta_k) = \mathbb{E}_{s \sim \rho_{\theta_k}} [0] = 0
$$

**结论**：当新旧策略参数相同时 ($\theta = \theta_k$)，代理目标函数的值**精确地等于零**。这在TRPO的理论推导中非常重要，因为它意味着 $L_{\theta_k}(\theta)$ 的一阶泰勒展开式中没有常数项，可以近似为 $L_{\theta_k}(\theta) \approx g^T(\theta - \theta_k)$，其中 $g$ 是 $L_{\theta_k}(\theta)$ 在 $\theta=\theta_k$ 处的梯度。这表明我们优化的目标是**从当前策略出发的“改进量”**，而当前点的改进量自然是0。

---

### 问题二：TRPO更新公式的推导

这个公式是TRPO核心思想的数学体现：**在一定约束（信赖域）内，最大化策略的提升**。

**1. 建立优化问题**

TRPO将策略优化问题建模为一个带约束的优化问题。它使用代理目标函数 $L_{\theta_k}(\theta)$ 的一阶泰勒展开来近似性能提升，并使用KL散度的二阶泰勒展开来作为信赖域约束。

- **目标函数（近似）**: 最大化策略性能提升。我们使用一阶近似：
  $$ \text{maximize}_{\theta} \quad L_{\theta_k}(\theta) \approx g^T (\theta - \theta_k) $$
  其中 $g = \nabla_{\theta} L_{\theta_k}(\theta) \Big|_{\theta = \theta_k}$ 是代理目标函数在当前策略参数 $\theta_k$ 处的梯度。

- **约束条件**: 保证新策略与旧策略不要相差太远。使用KL散度来度量两个策略的“距离”，并要求它小于一个很小的常数 $\delta$。我们使用KL散度的二阶近似：
  $$ \text{subject to} \quad \bar{D}_{KL}(\pi_{\theta_k} || \pi_{\theta}) \approx \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) \leq \delta $$
  其中 $H$ 是KL散度关于 $\theta$ 在 $\theta_k$ 处的**海森矩阵（Hessian Matrix）**。在强化学习中，这个海森矩阵就是**费雪信息矩阵（Fisher Information Matrix, FIM）**。

**2. 求解带约束的优化问题**

令 $\Delta\theta = \theta - \theta_k$，优化问题可以重写为：
$$
\begin{aligned}
\text{maximize}_{\Delta\theta} & \quad g^T \Delta\theta \\
\text{subject to} & \quad \frac{1}{2} \Delta\theta^T H \Delta\theta \leq \delta
\end{aligned}
$$

这是一个典型的二次约束下的线性最大化问题。我们可以使用**拉格朗日乘子法**来求解。构造拉格朗日函数：
$$
\mathcal{L}(\Delta\theta, \lambda) = g^T \Delta\theta - \lambda \left( \frac{1}{2} \Delta\theta^T H \Delta\theta - \delta \right)
$$
其中 $\lambda \ge 0$ 是拉格朗日乘子。

对 $\Delta\theta$ 求导并令其为零，以找到极值点：
$$
\nabla_{\Delta\theta} \mathcal{L} = g - \lambda H \Delta\theta = 0
$$
解得：
$$
\lambda H \Delta\theta = g \implies \Delta\theta = \frac{1}{\lambda} H^{-1} g
$$
这个结果告诉我们，最优的更新方向 $\Delta\theta$ 与**自然梯度** $H^{-1}g$ 的方向相同。$\frac{1}{\lambda}$ 决定了步长的大小。

**3. 确定步长（求解 $\lambda$）**

为了最大化目标函数，我们应该在约束边界上取值，即让约束条件取等号：
$$
\frac{1}{2} \Delta\theta^T H \Delta\theta = \delta
$$
将我们上面求得的 $\Delta\theta = \frac{1}{\lambda} H^{-1} g$ 代入这个等式：
$$
\frac{1}{2} \left( \frac{1}{\lambda} H^{-1} g \right)^T H \left( \frac{1}{\lambda} H^{-1} g \right) = \delta
$$
$$
\frac{1}{2\lambda^2} (g^T (H^{-1})^T) H (H^{-1} g) = \delta
$$
由于费雪信息矩阵 $H$ 是对称正定的，所以它的逆 $H^{-1}$ 也是对称的，即 $(H^{-1})^T = H^{-1}$。因此 $(H^{-1})^T H = H^{-1}H = I$（单位矩阵）。
$$
\frac{1}{2\lambda^2} g^T H^{-1} g = \delta
$$
现在我们可以解出 $\frac{1}{\lambda}$：
$$
\frac{1}{\lambda^2} = \frac{2\delta}{g^T H^{-1} g} \implies \frac{1}{\lambda} = \sqrt{\frac{2\delta}{g^T H^{-1} g}}
$$
（我们取正根，因为我们希望更新方向与梯度 $g$ 一致，即 $\lambda > 0$）。

**4. 得到最终的更新公式**

将求得的 $\frac{1}{\lambda}$ 代回到 $\Delta\theta$ 的表达式中：
$$
\Delta\theta = \frac{1}{\lambda} H^{-1} g = \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g
$$
最后，更新策略参数：
$$
\theta=\theta_{old}+\Delta \theta=\theta_{old}+\sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g
$$
这就完成了整个推导。

**实践中的注意点**：
在实际算法中，计算并求逆巨大的海森矩阵 $H$ 是非常耗时且不稳定的。TRPO的精髓在于使用**共轭梯度法（Conjugate Gradient）**来高效地近似计算 $H^{-1}g$ 这一项，从而避免了直接求逆，使得算法在处理高维参数空间时变得可行。

## 请详细解释一下TRPO 通过共轭梯度法求解参数梯度的过程。

好的，我们来详细拆解TRPO算法中一个最精妙也最关键的部分：**如何使用共轭梯度法（Conjugate Gradient, CG）来求解更新方向**。

### 1. 回顾问题：我们需要求解什么？

在上一问的推导中，我们得到了理想的更新步长 $\Delta\theta$：
$$ \Delta\theta = \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g $$
这里的核心计算瓶颈在于两点：
1.  **计算 $H^{-1}$**：$H$是费雪信息矩阵（Fisher Information Matrix, FIM），对于一个有数百万参数的神经网络策略，这是一个巨大（百万 x 百万）的矩阵。直接计算它的逆矩阵，计算复杂度约为 $O(N^3)$，其中 $N$ 是参数数量，这在计算上是完全不可行的。
2.  **存储 $H$**：即使不求逆，仅仅是构建和存储这个 $N \times N$ 的矩阵，内存开销也是无法承受的。

观察上面的公式，我们发现我们并不需要 $H^{-1}$ 本身，我们只需要计算出向量 $x = H^{-1}g$。

换句话说，我们的核心任务是求解一个大规模的线性方程组：
$$ Hx = g $$
其中，$H$ 是一个我们不想（也不能）显式构建的巨大矩阵，$g$ 是我们已知的策略梯度向量，$x$ 是我们要求解的未知向量（即“自然梯度”方向）。

这就是共轭梯度法大显身手的地方。

### 2. 共轭梯度法（CG）的核心思想

共轭梯度法是一种用于求解形如 $Ax=b$ 的线性方程组的**迭代算法**，它特别适用于当矩阵 $A$ 是**对称正定**的时候。幸运的是，费雪信息矩阵 $H$ 正是如此。

**CG的优势：**
*   **无需求逆**：它完全避免了计算 $A^{-1}$。
*   **无需存储矩阵**：它只需要能够计算矩阵-向量乘积（Matrix-Vector Product, MVP），即给定一个向量 $v$，能计算出 $Av$ 即可。它不需要访问或存储矩阵 $A$ 的任何单个元素。
*   **迭代求解**：它能在少数几次迭代后（远少于参数维度 $N$），给出一个非常好的近似解。

### 3. CG的关键：Hessian-Vector Product (HVP)

既然CG的核心是计算矩阵-向量乘积 $Hv$，那么TRPO是如何在不构建 $H$ 的情况下计算出 $Hv$ 的呢？这就是所谓的**Hessian-Vector Product (HVP)** 技巧。

回忆一下，$H$ 是KL散度关于策略参数 $\theta$ 的海森矩阵（二阶导数矩阵），在 $\theta = \theta_k$ 处求值：
$$ H = \nabla^2_{\theta} D_{KL}(\pi_{\theta_k} || \pi_{\theta}) \Big|_{\theta=\theta_k} $$
我们要计算的是 $Hv$。让我们考虑一个函数 $f(\theta) = (\nabla_{\theta} D_{KL}(\pi_{\theta_k} || \pi_{\theta}))^T v$。这是一个标量函数，它是KL散度的梯度与一个固定向量 $v$ 的点积。

现在，我们对这个标量函数 $f(\theta)$ 求关于 $\theta$ 的梯度：
$$ \nabla_{\theta} f(\theta) = \nabla_{\theta} \left( (\nabla_{\theta} D_{KL})^T v \right) $$
根据向量微积分的法则，这个结果恰好是海森矩阵乘以向量 $v$：
$$ \nabla_{\theta} f(\theta) = (\nabla^2_{\theta} D_{KL}) v = Hv $$
**这就找到了关键点：**
计算 $Hv$ 的问题，被转化为了计算**某个标量函数的梯度**的问题。

**如何实现？**
现代的自动微分框架（如TensorFlow, PyTorch）可以非常高效地完成这个操作：
1.  **计算KL散度的梯度**：首先，我们写出计算 $D_{KL}(\pi_{\theta_k} || \pi_{\theta})$ 的代码，并用自动微分得到其梯度 $g_{KL} = \nabla_{\theta} D_{KL}$。这是一个计算图。
2.  **计算点积**：将上一步得到的梯度向量 $g_{KL}$ 与我们给定的输入向量 $v$ 做点积，得到一个标量值 `L = torch.dot(g_kl, v)`。
3.  **对点积结果再次求导**：最后，我们对这个标量 `L` 再次调用自动微分，计算 `L` 相对于策略参数 $\theta$ 的梯度。这个结果就是我们想要的 $Hv$。

这个过程被称为**“二次反向传播”（double backpropagation）**或利用了**R-operator**。整个过程完全没有显式地构建或存储巨大的海森矩阵 $H$。

### 4. TRPO中CG的完整流程

现在我们可以将所有部分组合起来，看看TRPO如何使用CG求解 $x = H^{-1}g$。

**输入:**
*   策略梯度 `g`
*   一个能计算 HVP 的函数 `compute_hvp(v)`
*   CG的迭代次数 `cg_iters` (通常是一个较小的数，如10)
*   一个小的阻尼系数 `damping` (为了数值稳定性)

**算法 (求解 `Hx = g`)：**
1.  **初始化**:
    *   解 `x` 初始化为零向量: `x = 0`
    *   残差 `r` 初始化为梯度: `r = g`
    *   搜索方向 `p` 初始化为残差: `p = r`
    *   计算残差的模的平方: `rdotr = r.dot(r)`

2.  **迭代循环** (for `i` from 0 to `cg_iters - 1`):
    *   **计算矩阵-向量乘积**: 调用 `z = compute_hvp(p)`。为了增加数值稳定性，实际计算的是 `(H + λI)p = Hp + λp`，所以 `z = compute_hvp(p) + damping * p`。
    *   **计算步长α**: `alpha = rdotr / p.dot(z)`
    *   **更新解**: `x += alpha * p`
    *   **更新残差**: `r -= alpha * z`
    *   **检查收敛**: 如果 `r` 的模已经非常小，可以提前终止。
    *   **更新搜索方向**:
        *   计算新的残差模的平方: `new_rdotr = r.dot(r)`
        *   计算 `beta`: `beta = new_rdotr / rdotr`
        *   更新搜索方向 `p`: `p = r + beta * p`
        *   更新 `rdotr`: `rdotr = new_rdotr`

3.  **输出**: 返回最终的解 `x`。这个 `x` 就是 $H^{-1}g$ 的一个高质量近似。

### 5. 将CG的解代入TRPO更新

1.  **收集数据**: 使用当前策略 $\pi_{\theta_k}$ 采样，计算优势 `A`。
2.  **计算策略梯度**: 计算 $g = \nabla L_{\theta_k}(\theta)|_{\theta_k}$。
3.  **求解搜索方向**: **使用上面的CG算法，求解 $s = H^{-1}g$。**
4.  **计算步长**: 计算缩放系数 $\beta = \sqrt{\frac{2\delta}{s^T g}}$。（注意 $s^T g = (H^{-1}g)^T g = g^T H^{-1} g$）。
5.  **计算最终更新**: 提议的更新步长为 $\Delta\theta = \beta \cdot s$。
6.  **线性搜索 (Line Search)**: 由于我们用了一阶和二阶近似，这个步长不一定能保证满足约束和提升性能。因此，需要从 $\Delta\theta$ 开始，不断缩小步长（例如，$\Delta\theta, 0.5\Delta\theta, 0.25\Delta\theta, ...$），直到找到一个新策略 $\theta_{new} = \theta_k + \Delta\theta_{shrunk}$，它同时满足：
    *   **性能提升**: $L_{\theta_k}(\theta_{new}) \ge 0$
    *   **KL约束**: $D_{KL}(\pi_{\theta_k} || \pi_{\theta_{new}}) \le \delta$
7.  **更新策略**: $\theta_{k+1} = \theta_{new}$。

### 总结

TRPO通过共轭梯度法（CG）巧妙地解决了自然梯度更新中的巨大计算障碍。其核心在于：

*   **问题转化**：将求解 $H^{-1}g$ 转化为求解线性系统 $Hx=g$。
*   **避免构建矩阵**：利用**Hessian-Vector Product (HVP)** 技巧，使得CG算法可以在不显式构建和存储$H$的情况下运行。
*   **高效近似**：CG是一个迭代算法，仅需少量迭代就能得到足够精确的搜索方向 $s=H^{-1}g$，使得整个更新步骤在计算上变得可行。

具体的求解过程python的代码和注释说明

共轭法求解过程：

$$
\begin{aligned}
& 1．初始化： \\
& - 解 x 初始化为零向量： \mathrm{x}=0 \\
& - 残差 r 初始化为梯度：r= g-Hx=g  \\
& - 搜索方向 p 初始化为残差：p=r  \\
& - 计算残差的模的平方：r \operatorname{dot} r=r \cdot \operatorname{dot}(r) \\

& 2．迭代循环： \\
& - 计算矩阵－向量内积：调用 z=compute＿hvp（ p）。为了增加数值稳定性，\\
& - 实际计算的是（ H+\lambda I ）p=H p+\lambda p ，所以 z=compute＿hvp（ p ）+ \lambda*p  \\
& - 计算步长 \boldsymbol{\alpha} ：\alpha＝rdotr／p \cdot \operatorname{dot}(z)  \\
& - 更新解：x += \alpha*p  \\
& - 更新残差r－＝\alpha*z  \\
& - 检查收敛：如果 r 的模已经非常小，可以提前终止。 \\
& - 更新搜素方向： \\
& \quad 计算新的残差模的平方：new＿rdotr＝r．dot(r)  \\
& \quad 计算 beta：beta＝new＿rdotr / rdotr \\
& \quad 更新搜索方向 \mathrm{p}: \mathrm{p}=\mathrm{r}+ beta { }^* \mathrm{p} \\
& \quad 更新 rdotr：rdotr＝new＿rdotr  \\
\end{aligned}
$$