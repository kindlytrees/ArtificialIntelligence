# DDIM Inference

## DDIM加速基本原理

以下公式为一种观点在(但并不是原文的思路)，构造成线性加权可以这么理解：xt可以由x0的一个权重加噪声，x-1也可以由x0的权重加噪声构成，因此x-1也可以由x0和xt的线性加权去构造

$$
\begin{aligned}
& q\left(x_{t-1} \mid x_t, x_0\right)=\frac{q\left(x_t \mid x_{t-1}, x_0\right) q\left(x_{t-1} \mid x_0\right)}{q\left(x_t \mid x_0\right)} \\
& \ne q\left(x_{t-1} \mid x_t, x_0\right)=\frac{q\left(x_t \mid x_{t-1}\right) q\left(x_{t-1} \mid x_0\right)}{q\left(x_t \mid x_0\right)}
\end{aligned}
$$

$$
\begin{aligned}
& q\left(x_{t-1} \mid x_t, x_0\right) \sim N\left(k x_0+m x_t, \sigma^2 I\right) \\
& x_{t-1}=k x_0+m x_t+\sigma \epsilon \\
& x_t=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon^{\prime} //前向扩散定义\\
& x_{t-1}=k x_0+m\left(\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon^{\prime}\right)+\sigma \epsilon //将上式代入 \\
& x_{t-1}=\left(k+m \sqrt{\bar{\alpha}_t}\right) x_0+\left(m \sqrt{1-\bar{\alpha}_t}\right) \epsilon^{\prime}+\sigma \epsilon \\
& x_{t-1}=\left(k+m \sqrt{\bar{\alpha}_t}\right) x_0+\sqrt{m^2\left(1-\bar{\alpha}_t\right)+\sigma^2} \epsilon //高斯分布的和仍为高斯分布\\
& k+m \sqrt{\bar{\alpha}_t}=\sqrt{\bar{\alpha}_{t-1}} \text { 和 } m^2\left(1-\bar{\alpha}_t\right)+\sigma^2=1-\bar{\alpha}_{t-1} //和x_{t-1}的前向扩散公式对照 \\
& //根据上面式子先解出m，然后解出k \\
& m=\frac{\sqrt{1-\bar{\alpha}_{t-1}-\sigma^2}}{\sqrt{1-\bar{\alpha}_t}} \text { 和 } k=\sqrt{\bar{\alpha}_{t-1}}-\sqrt{1-\bar{\alpha}_{t-1}-\sigma^2} \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{1-\bar{\alpha}_t}} \\
& //根据求解出的m和k带入得出分布如下 \\
& q\left(x_{t-1} \mid x_t, x_0\right)=N\left(\sqrt{\bar{\alpha}_{t-1}} x_0+\sqrt{1-\bar{\alpha}_{t-1}-\sigma^2} \frac{x_t-\sqrt{\bar{\alpha}_t} x_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma^2 I\right)
\end{aligned}
$$

$$
\begin{aligned}
& q\left(x_{t-1} \mid x_t, x_0\right)=N\left(\sqrt{\bar{\alpha}_{t-1}} x_0+\sqrt{1-\bar{\alpha}_{t-1}-\sigma^2} \frac{x_t-\sqrt{\bar{\alpha}_t} x_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma^2 I\right) \\
& x_t=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon^{\prime} //前向扩散定义,下方公式的基于该等式进行的变换\\
& x_{t-1}=\sqrt{\bar{\alpha}_{t-1}}\left(\frac{x_t-\sqrt{1-\bar{\alpha}_t} \epsilon_\theta\left(x_t\right)}{\sqrt{\bar{\alpha}_t}}\right)+\sqrt{1-\bar{\alpha}_{t-1}-\sigma^2} \epsilon_\theta\left(x_t\right)+\sigma \epsilon \\
& 取消时间间隔相邻的约束\\
& x_s=\sqrt{\bar{\alpha}_s}\left(\frac{x_t-\sqrt{1-\bar{\alpha}_k} \epsilon_\theta\left(x_t\right)}{\sqrt{\bar{\alpha}_t}}\right)+\sqrt{1-\bar{\alpha}_s-\sigma^2} \epsilon_\theta\left(x_t\right)+\sigma \epsilon
\end{aligned}
$$

### DDIM的公式中的$\sigma$如何定义和计算？
如果eta为1是否可以理解为原始的ddpm中的多个累计跨度步骤的(多个高斯噪声的方差和的)叠加？
直觉思路有正确的部分，但不是简单的和的叠加
精确地等于将DDPM中从 τ_{i-1} 到 τ_i 这一整个“大步”的加噪过程，视为一个单一的马尔可夫步骤时，其反向去噪过程所对应的后验分布的方差。

### ddim的推理加速过程的基本原理

好的，我们来详细解析一下 DDIM 中这个核心公式的由来。这部分内容是理解 DDIM 如何实现对 DDPM 泛化并加速采样的关键。

您的提问非常精准，这个公式不是像 DDPM 那样从贝叶斯公式“推导”出来的，而是被巧妙地“构造”出来的。其设计的核心目标是：

**在引入一个可控的随机性参数 $\sigma_t$ 的同时，保证整个（非马尔可夫）过程的边缘分布 $q(\mathbf{x}_t|\mathbf{x}_0)$ 与原始 DDPM 的边缘分布完全一致。**

这样做的巨大好处是：**我们可以使用一个在标准 DDPM 目标上训练好的模型，无需任何修改，就能用于 DDIM 的采样过程。**

下面我们来分步拆解这个公式是如何被构造出来的。

---

### 第一步：回顾 DDPM 的逆向过程

在 DDPM 中，标准的（马尔可夫）前向过程是：
$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$

其逆向后验分布 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ 可以通过贝叶斯定理精确推导出来，结果是一个高斯分布。其均值和方差是固定的，由 $\beta_t$ 决定。这是 DDPM 随机性的来源，也是其必须一步步采样，无法跳步的原因。

### 第二步：DDIM 的核心思想——构造非马尔可夫过程

DDIM 的作者们思考：我们真的需要一个马尔可夫过程吗？我们能不能设计一个更通用的**非马尔可夫**前向过程 $q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0)$，它也能得到和 DDPM 一样的边缘分布 $q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t}\mathbf{x}_0, (1-\alpha_t)\mathbf{I})$？

如果能做到这一点，我们就可以反过来定义一个对应的逆向过程 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$，这个过程可能具有更好的性质（比如确定性）。

DDIM 没有先定义前向过程，而是反其道而行之，直接**定义（构造）了逆向过程** $q_\sigma(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$，并证明这个构造是合理的。

### 第三步：构造 $q_\sigma(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ 公式

我们来分析这个公式的构造逻辑：

$$
q_\sigma\left(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{x}_0\right)=\mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\sigma(\mathbf{x}_t, \mathbf{x}_0), \sigma_t^2 \boldsymbol{I}\right)
$$

其中均值为：
$$
u_a\left(x_t, x_0\right)=\sqrt{\overline{\alpha_{t-1}}} x_0+\sqrt{1-\overline{\alpha_{t-1}}-\sigma_t^2} \cdot \frac{x_t-\sqrt{\overline{\alpha_t}} x_t}{\sqrt{1-\overline{\alpha_t}}}
$$

这个公式的构造可以理解为三个部分的组合：

1.  **确定性部分 "Predicted $\mathbf{x}_0$"**:
    $\sqrt{\overline{\alpha_{t-1}}} \boldsymbol{x}_0$
    这部分直接将最终的干净图像 $\mathbf{x}_0$ 按照 $t-1$ 时刻的信噪比进行缩放，构成了生成 $\mathbf{x}_{t-1}$ 的“确定性骨架”。

2.  **方向部分 "Direction to $\mathbf{x}_t$"**:
    $\frac{\boldsymbol{x}_t-\sqrt{\bar \alpha_t} \boldsymbol{x}_0}{\sqrt{1-\bar \alpha_t}}$
    我们知道 DDPM 的前向过程可以写成 $\mathbf{x}_t = \sqrt{\bar \alpha_t}\mathbf{x}_0 + \sqrt{1-\bar \alpha_t}\boldsymbol{\epsilon}$，其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$。
    因此，上面这个分数项**恰好就等于标准正态分布的噪声 $\boldsymbol{\epsilon}$**。它代表了从 $\mathbf{x}_0$ 到 $\mathbf{x}_t$ 所加噪声的“方向”。

3.  **随机性部分 "Controlled Stochasticity"**:
    *   $\sqrt{1-\overline{\alpha_{t-1}}-\sigma_t^2}$ 是方向部分（噪声 $\boldsymbol{\epsilon}$）的系数。
    *   $\sigma_t^2 \mathbf{I}$ 是整个高斯分布的方差，代表了与 $\boldsymbol{\epsilon}$ 正交的、新加入的随机噪声。

所以，生成 $\mathbf{x}_{t-1}$ 的过程可以直观地理解为：
$$
\mathbf{x}_{t-1}=\underbrace{\sqrt{\overline{\alpha_{t-1}}} \mathbf{x}_0}_{\text {确定性骨架 }}+\underbrace{\sqrt{1-\overline{\alpha_{t-1}}-\sigma_t^2} \cdot \boldsymbol{\epsilon}}_{\text {沿已知噪声方向的偏移 }}+\underbrace{\sigma_t \cdot \boldsymbol{\epsilon}^{\prime}}_{\text {新的随机噪声 }}
$$
其中 $\boldsymbol{\epsilon}$ 是从 $\mathbf{x}_t$ 和 $\mathbf{x}_0$ 中计算得到的，而 $\boldsymbol{\epsilon'} \sim \mathcal{N}(0, \mathbf{I})$ 是新采样的噪声。

### 第四步：证明这个构造的合理性（核心）

现在我们来证明，为什么这样构造的均值和方差，能够保证边缘分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_0)$ 是正确的。

我们要证明的是：如果我们遵循上述采样过程，最终得到的 $\mathbf{x}_{t-1}$ 的分布就是 $\mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\alpha_{t-1}}\mathbf{x}_0, (1-\alpha_{t-1})\mathbf{I})$。

根据上面的采样公式，$\mathbf{x}_{t-1}$ 是一个高斯随机变量（因为它是常数和高斯变量的线性组合）。我们只需要计算它的均值和方差。

*   **计算均值 $\mathbb{E}[\mathbf{x}_{t-1}]$**:
    由于 $\mathbb{E}[\boldsymbol{\epsilon}] = 0$ 且 $\mathbb{E}[\boldsymbol{\epsilon'}] = 0$，我们有：
    $$
    \mathbb{E}[\mathbf{x}_{t-1}] = \mathbb{E}[\sqrt{\overline{\alpha_{t-1}}} \mathbf{x}_0 + \sqrt{1-\overline{\alpha_{t-1}}-\sigma_t^2} \cdot \boldsymbol{\epsilon} + \sigma_t \cdot \boldsymbol{\epsilon'}] = \sqrt{\overline{\alpha_{t-1}}}\mathbf{x}_0
    $$
    均值匹配成功！

*   **计算方差 $\text{Var}[\mathbf{x}_{t-1}]$**:
    由于 $\mathbf{x}_0$ 是给定的常数，并且 $\boldsymbol{\epsilon}$ 和 $\boldsymbol{\epsilon'}$ 是独立的标准正态分布，它们的方差都是 $\mathbf{I}$。
    $$
    \begin{aligned}
    \text{Var}[\mathbf{x}_{t-1}] & = \text{Var}[\sqrt{\overline{\alpha_{t-1}}} \mathbf{x}_0 + \sqrt{1-\overline{\alpha_{t-1}}-\sigma_t^2} \cdot \boldsymbol{\epsilon} + \sigma_t \cdot \boldsymbol{\epsilon'}] \\
    & = (\sqrt{1-\overline{\alpha_{t-1}}-\sigma_t^2})^2 \cdot \text{Var}[\boldsymbol{\epsilon}] + (\sigma_t)^2 \cdot \text{Var}[\boldsymbol{\epsilon'}] \\
    & = (1-\overline{\alpha_{t-1}}-\sigma_t^2)\mathbf{I} + \sigma_t^2 \mathbf{I} \\
    & = (1-\overline{\alpha_{t-1}})\mathbf{I}
    \end{aligned}
    $$
    方差匹配成功！

**结论**：这个精巧的构造，通过将总方差 $(1-\overline{\alpha_{t-1}})$ 分解为两部分——一部分沿着已知噪声方向的方差 $(1-\overline{\alpha_{t-1}}-\sigma_t^2)$ 和一部分新的随机方差 $\sigma_t^2$——完美地保证了无论 $\sigma_t$ 如何取值（在 $1-\overline{\alpha_{t-1}}-\sigma_t^2 \geq 0$ 的前提下），其边缘分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_0)$ 始终不变。

---

### 与 DDPM 和 DDIM (deterministic) 的关系

*   **当 $\sigma_t^2 = \frac{1-\alpha_{t-1}}{1-\alpha_t}(1-\alpha_t)$ 时**：这个过程退化为标准的 DDPM 逆向过程。
*   **当 $\sigma_t^2 = 0$ 时**：这是 DDIM 的最常用设置。方差项变为0，整个逆向过程变成完全**确定性**的：
    $$
    \mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}} \mathbf{x}_0 + \sqrt{1-\alpha_{t-1}} \cdot \underbrace{\frac{\boldsymbol{x}_t-\sqrt{\alpha_t} \boldsymbol{x}_0}{\sqrt{1-\alpha_t}}}_{=\boldsymbol{\epsilon}}
    $$
    在实际采样中，我们没有真实的 $\mathbf{x}_0$，但我们可以用神经网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 来预测噪声 $\boldsymbol{\epsilon}$，然后用它来计算一个对 $\mathbf{x}_0$ 的预测值 $\hat{\mathbf{x}}_0$，再代入上式进行确定性采样。正是因为确定性，采样路径变得稳定，才允许我们进行大步长的跳步采样，从而实现加速。

### 总结

DDIM 中 $q_\sigma(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ 的公式**不是推导出来的，而是被精心构造出来的**。其构造的核心思想是**解耦**：将生成 $\mathbf{x}_{t-1}$ 的过程分解为一个**确定性骨架** ($\sqrt{\alpha_{t-1}}\mathbf{x}_0$)、一个**指向 $\mathbf{x}_t$ 的方向** (由噪声 $\boldsymbol{\epsilon}$ 体现) 和一个**可控的随机项** ($\sigma_t$)。通过巧妙地设置各项系数，使得无论随机项 $\sigma_t$ 多大，最终生成的 $\mathbf{x}_{t-1}$ 的边缘分布都与 DDPM 保持一致，从而实现了对 DDPM 模型的复用和采样过程的泛化。

### 1. DDIM的跳步采样原理 (`t-1` vs `t-steps`)

DDIM 的采样公式之所以能够支持跳步，根本原因在于它**不依赖于马尔可夫假设**。它的每一步都直接与预测的 `x̂_0` 挂钩。

我们回顾一下确定性DDIM (`σ=0`) 的单步采样公式：
$$ \mathbf{x}_{t-1} = \sqrt{\overline{\alpha_{t-1}}}\hat{\mathbf{x}}_0 + \sqrt{1-\overline{\alpha_{t-1}}} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) $$
其中 `x̂_0` 是由 `x_t` 和 `ε_θ` 计算得来的。

现在，我们要从 `t` 时刻直接跳到 `t-k` 时刻（`k > 1`）。我们只需要将公式中所有 `t-1` 的地方换成 `t-k` 即可：
$$ \mathbf{x}_{t-k} = \sqrt{\overline{\alpha_{t-k}}}\hat{\mathbf{x}}_0 + \sqrt{1-\overline{\alpha_{t-k}}} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) $$
这个公式依然成立！因为 `x̂_0` 和 `ε_θ` 都是在当前 `t` 时刻计算的，`α_{t-k}` 也是从预先计算好的 `α` 序列中直接查找得到的。整个过程与中间的 `t-1, t-2, ..., t-k+1` 时刻完全无关。

**实践中的做法：**

1.  **定义采样序列：** 假设DDPM总步数 `T=1000`。我们不想走1000步，只想走 `S=50` 步。我们就先定义一个长度为50的采样时间步序列 `τ`，例如 `τ = [981, 961, 941, ..., 21, 1]`。
2.  **迭代采样：** 我们从 `i = 1` 到 `S-1` 进行迭代：
    *   当前时刻是 `t_curr = τ_i` (例如 981)
    *   下一个时刻是 `t_prev = τ_{i+1}` (例如 961)
    *   当前样本是 `x_{t_curr}`。
    *   计算 `ε_θ(x_{t_curr}, t_curr)` 和 `x̂_0`。
    *   使用通用公式生成下一个样本 `x_{t_prev}`：
        $$ \mathbf{x}_{t_{\text{prev}}} = \sqrt{\overline{\alpha_{t_{\text{prev}}}}}\hat{\mathbf{x}}_0 + \sqrt{1-\overline{\alpha_{t_{\text{prev}}}}-\sigma_{t_{\text{curr}}}^2} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_{t_{\text{curr}}}, t_{\text{curr}}) + \sigma_{t_{\text{curr}}} \cdot \boldsymbol{\epsilon'} $$

### 2. 如何定义和选择σ (或η)

您的理解再次正确：如果采样步数是 `S=50`，那么 `σ` 序列的长度就是50。`σ` 的取值需要满足的唯一硬性约束是其平方 `σ_t^2` 不能超过 `1 - α_{t-1}`，以保证根号内的项 `1 - α_{t-1} - σ_t^2` 大于等于0。

在实践中，我们不直接定义 `σ`，而是通过一个更直观的超参数 `η` (eta) 来控制。`η` 在 `[0, 1]` 之间取值，用来插值DDIM（确定性）和DDPM（随机性）。

`σ_t` 的值通过 `η` 定义如下：
$$ \sigma_t^2(\eta) = \eta^2 \cdot \tilde{\beta}_t $$
这里的 `β̃_t` 是对应于DDPM过程的方差。在跳步采样中，它被定义为：
$$ \tilde{\beta}_{t_{\text{curr}}} = \frac{1-\alpha_{t_{\text{prev}}}}{1-\alpha_{t_{\text{curr}}}} \left(1 - \frac{\alpha_{t_{\text{curr}}}}{\alpha_{t_{\text{prev}}}}\right) $$


其中sigma为生成过程添加了随机性，在DDPM算法中，其逆向过程的近似分布为

$$
p\left(x_{t-1} \mid x_t, x_0\right)=\mathcal{N}\left(x_{t-1} ; \mu_t\left(x_t, x_0\right), \tilde{\beta}_t I\right)
$$


其中用表达式展开描述为

$$
\mu_t\left(x_t, x_0\right)=\sqrt{\overline{\alpha_{t-1}}} x_0+\sqrt{1-\overline{\alpha_{t-1}}-\tilde{\beta}_t} \cdot \frac{x_t-\sqrt{\bar\alpha_t} x_0}{\sqrt{1-\bar \alpha_t}}
$$


方差的形式

$$
\tilde{\beta}_t=\frac{1-\overline{\alpha_{t-1}}}{1-\bar \alpha_t} \cdot \beta_t
$$

为了让 DDIM 的逆向过程在随机性上与 DDPM 一致，我们希望方差sigma满足：

$$
\sigma_t^2=\tilde{\beta}_t=\frac{1-\overline{\alpha_{t-1}}}{1-\bar \alpha_t} \cdot \beta_t
$$


但是，DDIM 引入了超参数来灵活控制随机性，因此方法被定义为

$$
\sigma_t=\eta \cdot \sqrt{\tilde{\beta}_t}=\eta \cdot \sqrt{\frac{1-\overline{\alpha_{t-1}}}{1-\bar \alpha_t} \cdot \beta_t} \quad \beta_t=1-\frac{\bar \alpha_t}{\overline{\alpha_{t-1}}}
$$

代码实现：

```
sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
```

通过这个公式，`η` 的含义变得非常清晰：

*   **`η = 0`**: 这使得 `σ_t` 永远为0。这就是**标准的、确定性的DDIM采样**。这是最常用、最快、通常生成图像最清晰的设置。
*   **`η = 1`**: 这使得 `σ_t^2` 等于 `β̃_t`，完全恢复了DDPM在该步长下的随机性。这被称为**随机性的DDIM采样**。
*   **`0 < η < 1`**: 这提供了介于纯确定性和纯随机性之间的采样过程。`η` 越大，引入的随机噪声就越多。

DDIM的生成流程可以表示为：

$$
x_{t-1}=\sqrt{\overline{\alpha_{t-1}}} \cdot \frac{x_t-\sqrt{1-\bar \alpha_t} \cdot \epsilon_\theta\left(x_t, t\right)}{\sqrt{\bar\alpha_t}}+\sqrt{1-\overline{\alpha_{t-1}}-\sigma_t^2} \cdot \epsilon_\theta\left(x_t, t\right)+\sigma_t \cdot \epsilon
$$

跨多个timestamp的写法

$$
\mathbf{x}_{t_{\text{prev}}} = \sqrt{\overline{\alpha_{t_{\text{prev}}}}}\hat{\mathbf{x}}_0 + \sqrt{1-\overline{\alpha_{t_{\text{prev}}}}-\sigma_{t_{\text{curr}}}^2} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_{t_{\text{curr}}}, t_{\text{curr}}) + \sigma_{t_{\text{curr}}} \cdot \boldsymbol{\epsilon'} 
$$

其中pred(x0)为：

$$
\hat{\mathbf{x}}_0 =\frac{1}{\sqrt{\bar{\alpha}_t}}\left(\boldsymbol{x}_t-\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}_\theta\left(\boldsymbol{x}_t, t\right)\right)
$$

**哪个效果更好？**
这取决于具体任务和需求。
*   `η = 0` (DDIM) 通常能获得**最高保真度**（Fidelity）的图像，细节清晰。
*   `η > 0` (随机DDIM) 引入的随机性可以**增加样本的多样性**（Diversity），并且在某些情况下可以**帮助纠正模型的错误**。如果模型在某个去噪步骤中预测跑偏了，一点随机噪声有时可以把它“踢”回正确的轨道上。但过大的`η`会降低样本质量，使图像看起来更模糊或不协调。

**一般建议：** 从 `η = 0` 开始。如果发现生成样本多样性不足或效果不佳，可以尝试引入少量随机性，如 `η = 0.1` 或 `η = 0.2`。

### 3. σ大小差异性的直观表现

`σ` 的大小直接控制了每一步去噪过程中引入的**新随机噪声的强度**。

让我们把采样公式看成两部分：
$$ \mathbf{x}_{t-1} = \underbrace{\left( \sqrt{\alpha_{t-1}}\hat{\mathbf{x}}_0 + \sqrt{1-\alpha_{t-1}-\sigma_t^2} \cdot \boldsymbol{\epsilon}_\theta \right)}_{\text{确定性部分}} + \underbrace{\sigma_t \cdot \boldsymbol{\epsilon'}}_{\text{随机性部分}} $$

*   **当 `σ = 0` (η=0, 确定性DDIM):**
    *   随机性部分为0。
    *   给定一个初始噪声 `x_T`，整个从 `x_T` 到 `x_0` 的生成路径是**完全固定**的。每次运行都会得到一模一样的结果。
    *   这就像从山顶沿着一条固定的、最陡峭的路径下山，路径上没有任何随机扰动。
    *   **表现：** 图像清晰、稳定，但可能缺乏一些“生气”或多样性。

*   **当 `σ > 0` (η>0, 随机DDIM):**
    *   在每一步，除了沿着模型预测的方向前进外，还会加上一个由 `σ` 控制强度的随机“推力” (`σ·ε'`)。
    *   即使从同一个 `x_T` 开始，每次运行生成的最终图像也会有所不同，因为每一步的 `ε'` 都是新采样的。
    *   这就像下山时，你每走一步，都会有一个随机的风把你往旁边吹一下。`σ` 越大，风力越强。
    *   **表现：**
        *   **优点（小`σ`）:** 可能会发现更好的下山路径（纠正模型错误），最终到达的山谷风景（生成的图像）更多样。
        *   **缺点（大`σ`）:** 如果风力太强，你可能会被吹得偏离主路太远，导致最终结果质量下降（图像模糊、结构混乱）。

### 总结

| 特性 | `σ = 0` (η=0, 确定性DDIM) | `σ > 0` (η>0, 随机DDIM) |
| :--- | :--- | :--- |
| **采样路径** | 给定`x_T`后完全固定 | 随机，每次运行都不同 |
| **图像保真度** | 通常更高，更清晰 | 可能略低，取决于`σ`大小 |
| **样本多样性** | 较低（多样性仅来自不同的`x_T`） | 更高（`x_T`和每步的随机性） |
| **纠错能力** | 无 | 有，随机噪声可能帮助跳出局部最优 |
| **常见用途** | 追求高质量、高稳定性的图像生成 | 追求样本多样性，或作为一种正则化手段 |

希望这个详细的解释能帮助您更深入地理解DDIM的加速采样机制和`σ`（或`η`）参数的作用！

###  两个问题：

辅助geok回答：

ddim的eta和ddpm的随机性相同时，eta为1，但这个时候迭代的步骤要小很多，因此也可以认为其在ddpm推理等效的基础上性能有较大的提升，可以这么理解吗？请给与必要的补充说明。
在图像生成领域随机性在迭代的早期更多，是否可以在时间步骤的跨度上在前面的步骤中跨度小一点，后面跨度大一点？

等效性是统计意义上的，即 DDIM（$\eta = 1$）生成的样本分布与 DDPM 的样本分布在理论上相同，但具体实现上可能略有差异（例如，DDIM 使用非马尔可夫过程，允许跳跃时间步）。
在实际中，由于数值精度、模型训练差异或采样步数的不同，DDIM（$\eta = 1$）的生成结果可能与 DDPM 略有偏差，但总体上非常接近。

在 $\eta = 1$ 时，DDIM 的随机性与 DDPM 等价，但如果步数过少，随机噪声的累积可能不足以充分探索数据流形，导致生成的样本多样性略低于 DDPM（尽管 FID 分数可能接近）。
在人脸数据集上，DDIM 的性能提升尤为明显，因为人脸流形相对结构化，模型容易学习到高质量的去噪映射。对于更复杂的数据集（如高分辨率自然图像），DDIM 可能需要更多步数以保证质量。

非均匀采样的理论依据
非均匀采样的合理性可以从噪声调度和去噪过程的特性推导：

噪声调度 $\beta_t$ 和 $\alpha_t$： 在扩散模型中，噪声调度 $\beta_t$ 通常随时间 $t$ 递增（例如，线性调度或余弦调度），导致早期时间步的累积方差 $\alpha_t = \prod_{s=1}^t (1 - \beta_s)$ 下降较快，后期下降较慢。这意味着早期时间步对信号的保留影响更大，需要更精细的控制。
随机噪声的影响： 早期时间步的 $\sigma_t$（或 DDPM 的 $\tilde{\beta}_t$）通常较大，随机噪声对去噪路径的扰动更强，因此更密集的采样可以确保去噪过程更稳定地接近数据流形。
后期细节调整： 后期时间步的 $\sigma_t$ 较小，随机噪声的影响主要体现在细节上，较稀疏的采样足以捕捉这些变化，而不会显著影响生成 predegenerate

## Notes
https://gitee.com/kindlytree/ai-learning/blob/master/difffusion-models/annotated_native_ddpm_impl.ipynb
https://huggingface.co/blog/annotated-diffusion


## 环境设置

- 在colab里的终端下安装miniconda，并运行相关环境条件

```
bash ./Miniconda3-latest-Linux-x86_64.sh

echo 'export PATH=$PATH:/root/minianaconda3/bin' >> ~/.bashrc

source ~/.bashrc

conda env create -f environment-validated.yaml

conda activate ldm
```

## 推理示例

```
cd Stable-Diffusion
mkdir -p models/ldm/cin256-v2/
wget -O models/ldm/cin256-v2/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt
python ldm_inference.py
```
