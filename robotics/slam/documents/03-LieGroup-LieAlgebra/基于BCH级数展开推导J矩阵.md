# 基于级数展开推导J矩阵

以下回答内容基于大模型反馈结果

基于指数展开能让我们深刻理解BCH公式、指数映射和雅可比矩阵之间的内在联系。我们将完全基于级数展开来推导左雅可比 $\boldsymbol{J}_l(\boldsymbol{\phi})$

这里的参数我们用 $\boldsymbol{\phi}$ 表示，以与旋转角 $\theta$ 区分。

### 目标

我们的目标是推导出 **左雅可比 $\boldsymbol{J}_l(\boldsymbol{\phi})$** 的级数形式。这个雅可比矩阵的定义来自于以下关系：
一个在李代数参数空间的微小变化 $\delta\boldsymbol{\phi}$，通过指数映射，可以等效于在当前姿态 $\exp(\boldsymbol{\phi}^\wedge)$ 上左乘一个微小的物理旋转 $\delta\boldsymbol{\epsilon}$。
$$
\underbrace{\exp((\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge)}_{\text{参数空间中的新姿态}} = \underbrace{\exp(\delta\boldsymbol{\epsilon}^\wedge)}_{\text{物理上的微小扰动}} \underbrace{\exp(\boldsymbol{\phi}^\wedge)}_{\text{旧姿态}}
$$
左雅可比 $\boldsymbol{J}_l(\boldsymbol{\phi})$ 正是连接这两个微小量 $\delta\boldsymbol{\phi}$ 和 $\delta\boldsymbol{\epsilon}$ 的桥梁：
$$
\delta\boldsymbol{\epsilon} = \boldsymbol{J}_l(\boldsymbol{\phi}) \delta\boldsymbol{\phi}
$$
我们的任务就是找出这个 $\boldsymbol{J}_l(\boldsymbol{\phi})$ 的表达式。

### 推导过程

推导的关键是**对同一个量用两种方式展开，然后对比系数**。这个量就是新姿态 $\exp((\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge)$。

#### 步骤 1: 展开右侧 $\exp(\delta\boldsymbol{\epsilon}^\wedge) \exp(\boldsymbol{\phi}^\wedge)$

我们直接使用BCH公式，其中 $X = \delta\boldsymbol{\epsilon}^\wedge$，$Y = \boldsymbol{\phi}^\wedge$。
$$
\log(\exp(\delta\boldsymbol{\epsilon}^\wedge) \exp(\boldsymbol{\phi}^\wedge)) = \delta\boldsymbol{\epsilon}^\wedge + \boldsymbol{\phi}^\wedge + \frac{1}{2}[\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge]  + \frac{1}{12}[\delta\boldsymbol{\epsilon}^\wedge, [\delta\boldsymbol{\epsilon}^\wedge,\boldsymbol{\phi}^\wedge]] + \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\boldsymbol{\phi}^\wedge,\delta\boldsymbol{\epsilon}^\wedge]] + \dots
$$
我们只关心 $\delta\boldsymbol{\epsilon}$ 的一阶项，其中 $\frac{1}{12}[\delta\boldsymbol{\epsilon}^\wedge, [\delta\boldsymbol{\epsilon}^\wedge,\boldsymbol{\phi}^\wedge]]$ 为 $\delta\boldsymbol{\epsilon}^\wedge $ 的二阶项，所以右侧可以近似写成：
$$
\exp(\delta\boldsymbol{\epsilon}^\wedge) \exp(\boldsymbol{\phi}^\wedge) \approx \exp\left( \boldsymbol{\phi}^\wedge + \delta\boldsymbol{\epsilon}^\wedge + \frac{1}{2}[\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge]  + \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\boldsymbol{\phi}^\wedge, \delta\boldsymbol{\epsilon}^\wedge]] + \dots \right)
$$

#### 步骤 2: 展开左侧 $\exp((\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge)$

左侧的展开相对简单。我们假设 $\delta\boldsymbol{\phi}$ 是一个微小量，直接进行泰勒展开。令 $f(\boldsymbol{\phi}) = \exp(\boldsymbol{\phi}^\wedge)$，我们想求 $f(\boldsymbol{\phi}+\delta\boldsymbol{\phi})$。
这实际上就是指数映射的微分。其标准形式为：
$$
\exp((\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge) = \exp(\boldsymbol{\phi}^\wedge) + \text{DerivativeTerm}
$$
一个更直接的方法是，我们认为左侧的李代数就是 $\boldsymbol{\phi}^\wedge + \delta\boldsymbol{\phi}^\wedge$。

#### 步骤 3: 建立等式并求解

现在我们让左右两侧指数函数内部的李代数相等：
$$
(\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge = \boldsymbol{\phi}^\wedge + \delta\boldsymbol{\phi}^\wedge \approx \boldsymbol{\phi}^\wedge + \delta\boldsymbol{\epsilon}^\wedge + \frac{1}{2}[\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge] - \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge]] + \dots
$$

消去 (后面直接用等号来表示左右近似相等关系) $\boldsymbol{\phi}^\wedge$：

$$
\delta\boldsymbol{\phi}^\wedge = \delta\boldsymbol{\epsilon}^\wedge + \frac{1}{2}[\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge] - \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge]] + \dots
$$
为了方便，我们使用李代数的**伴随表示 (Adjoint representation)** $\text{ad}(\cdot)$。对于 $\mathfrak{so}(3)$，我们有：
*   $[\boldsymbol{A}, \boldsymbol{B}] = \text{ad}(\boldsymbol{A})\boldsymbol{B}$
*   $[\boldsymbol{a}^\wedge, \boldsymbol{b}^\wedge] = (\boldsymbol{a} \times \boldsymbol{b})^\wedge$
*   $\text{ad}(\boldsymbol{a}^\wedge)\boldsymbol{b}^\wedge = (\boldsymbol{a} \times \boldsymbol{b})^\wedge$
所以，$\text{ad}(\boldsymbol{\phi}^\wedge)$ 作用在一个反对称矩阵 $\boldsymbol{v}^\wedge$ 上，等效于 $\boldsymbol{\phi}^\wedge$ 作用在向量 $\boldsymbol{v}$ 上。

将上式转换为伴随表示的形式：
$$
\delta\boldsymbol{\phi}^\wedge = \delta\boldsymbol{\epsilon}^\wedge - \frac{1}{2}[\boldsymbol{\phi}^\wedge, \delta\boldsymbol{\epsilon}^\wedge] + \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\boldsymbol{\phi}^\wedge, \delta\boldsymbol{\epsilon}^\wedge]] + \dots
$$
$$
\delta\boldsymbol{\phi}^\wedge = \left( \boldsymbol{I} - \frac{1}{2}\text{ad}(\boldsymbol{\phi}^\wedge) + \frac{1}{12}\text{ad}(\boldsymbol{\phi}^\wedge)^2 - \dots \right) \delta\boldsymbol{\epsilon}^\wedge
$$
去掉两边的 `^` 算子，我们就得到了参数空间和物理空间微小量的关系：
$$
\delta\boldsymbol{\phi} = \left( \boldsymbol{I} - \frac{1}{2}\text{ad}(\boldsymbol{\phi}) + \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2 - \dots \right) \delta\boldsymbol{\epsilon}
$$

#### 步骤 4: 求逆以得到 $\boldsymbol{J}_l(\boldsymbol{\phi})$

回顾我们的目标：$\delta\boldsymbol{\epsilon} = \boldsymbol{J}_l(\boldsymbol{\phi}) \delta\boldsymbol{\phi}$。
而我们刚刚推导出的是 $\delta\boldsymbol{\phi}$ 关于 $\delta\boldsymbol{\epsilon}$ 的表达式。所以，我们需要求上面那个大矩阵的**逆**。

令 $\boldsymbol{A} = \boldsymbol{I} - \frac{1}{2}\text{ad}(\boldsymbol{\phi}) + \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2 - \dots$
则 $\boldsymbol{J}_l(\boldsymbol{\phi}) = \boldsymbol{A}^{-1}$。

我们使用一个经典的级数求逆公式：对于矩阵 $\boldsymbol{X}$，如果其范数小于1，则 $(\boldsymbol{I}-\boldsymbol{X})^{-1} = \boldsymbol{I} + \boldsymbol{X} + \boldsymbol{X}^2 + \dots$。
在这里，令 $\boldsymbol{X} = \frac{1}{2}\text{ad}(\boldsymbol{\phi}) - \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2 + \dots$
$$
\boldsymbol{J}_l(\boldsymbol{\phi}) = (\boldsymbol{I} - \boldsymbol{X})^{-1} = \boldsymbol{I} + \boldsymbol{X} + \boldsymbol{X}^2 + \dots
$$
我们来计算级数的前几项：
*   **$\boldsymbol{I}$**: 就是 $\boldsymbol{I}$。
*   **$\boldsymbol{X}$**: 就是 $\frac{1}{2}\text{ad}(\boldsymbol{\phi}) - \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2 + \dots$
*   **$\boldsymbol{X}^2$**: 我们只关心最低阶项，即 $(\frac{1}{2}\text{ad}(\boldsymbol{\phi}))^2 = \frac{1}{4}\text{ad}(\boldsymbol{\phi})^2$。

将它们相加：
$$
\begin{aligned}
\boldsymbol{J}_l(\boldsymbol{\phi}) &= \boldsymbol{I} + \left(\frac{1}{2}\text{ad}(\boldsymbol{\phi}) - \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2\right) + \frac{1}{4}\text{ad}(\boldsymbol{\phi})^2 + \dots \\
&= \boldsymbol{I} + \frac{1}{2}\text{ad}(\boldsymbol{\phi}) + \left(-\frac{1}{12} + \frac{1}{4}\right)\text{ad}(\boldsymbol{\phi})^2 + \dots \\
&= \boldsymbol{I} + \frac{1}{2}\text{ad}(\boldsymbol{\phi}) + \left(-\frac{1}{12} + \frac{3}{12}\right)\text{ad}(\boldsymbol{\phi})^2 + \dots \\
&= \boldsymbol{I} + \frac{1}{2}\text{ad}(\boldsymbol{\phi}) + \frac{2}{12}\text{ad}(\boldsymbol{\phi})^2 + \dots \\
&= \boldsymbol{I} + \frac{1}{2}\text{ad}(\boldsymbol{\phi}) + \frac{1}{6}\text{ad}(\boldsymbol{\phi})^2 + \dots
\end{aligned}
$$

### 结论

我们成功地**仅通过BCH级数展开和级数求逆**，推导出了左雅可比 $\boldsymbol{J}_l(\boldsymbol{\phi})$ 的级数形式：
$$
\boldsymbol{J}_l(\boldsymbol{\phi}) = \boldsymbol{I} + \frac{1}{2}\text{ad}(\boldsymbol{\phi}) + \frac{1}{6}\text{ad}(\boldsymbol{\phi})^2 + \frac{1}{24}\text{ad}(\boldsymbol{\phi})^3 + \dots
$$
这个级数可以被证明等于：
$$
\boldsymbol{J}_l(\boldsymbol{\phi}) = \sum_{n=0}^{\infty} \frac{1}{(n+1)!} \text{ad}(\boldsymbol{\phi})^n
$$
这正是左雅可比的标准定义之一。

最后，将 $\text{ad}(\boldsymbol{\phi})$ 替换回我们更熟悉的反对称矩阵形式 $\boldsymbol{\phi}^\wedge$，就得到了在机器人学和计算机视觉中常见的形式：
$$
\boldsymbol{J}_l(\boldsymbol{\phi}) = \boldsymbol{I} + \frac{1}{2}\boldsymbol{\phi}^\wedge + \frac{1}{6}(\boldsymbol{\phi}^\wedge)^2 + \frac{1}{24}(\boldsymbol{\phi}^\wedge)^3 + \dots
$$
这个过程清晰地展示了BCH公式是如何成为连接参数空间和物理运动空间的基石，而雅可比矩阵正是这一连接关系的线性化体现。


##  问题1：从BCH得出雅可比的逆的级数表示的过程

辅助大模型回答并加以验证和修订：

当然可以。我们来详细地推导BCH一阶近似如何导出这两个雅可比表达式。这个过程是理解李群与李代数在优化问题中应用的核心。

我们的目标是分析表达式 $ln(exp(φ₁^\wedge) exp(φ₂^\wedge))$，并在线性化假设下（即其中一个旋转是微小量），找到它与 $φ₁$ 和 $φ₂$ 的关系。

### 基础：BCH公式和伴随表示

**BCH公式**给出了两个李代数元素指数映射乘积的对数：

$$
\ln(e^X e^Y) = X + Y + \frac{1}{2}[X, Y] + \frac{1}{12}([X, [X, Y]] + [Y, [Y, X]]) + \dots
$$
其中 $[X, Y] = XY - YX$ 是李括号

对于 $so(3)$，李括号对应于叉乘： $[φ₁^\wedge, φ₂^\wedge] = {(φ₁ × φ₂)}^\wedge$ 

我们还需要**伴随表示（Adjoint Representation）**的李代数形式，记为 $ad$。对于 $X, Y ∈ so(3)$，$ad(X)Y = [X, Y]$。
如果 $X = φ^\wedge$ 且 $Y = ψ^\wedge$，那么 $ad(φ^\wedge)ψ^\wedge = [φ^\wedge, ψ^\wedge] = (φ × ψ)^\wedge$。

---

### 推导一：左扰动模型（$φ₁$ 为小量）

我们分析 $ln(exp(φ₁^\wedge) exp(φ₂^\wedge))$，其中 $φ₁$ 是一个微小的扰动量，我们称之为 $δφ$。$φ₂$ 是一个已知的、大小不限的旋转，我们称之为 $φ$。

所以，我们要分析的是： $ln(exp(δφ^\wedge) exp(φ^\wedge))$，其中 $δφ$ 是小量。

我们将BCH公式的 $X$ 替换为 $δφ^\wedge$，$Y$ 替换为 $φ^\wedge$。由于我们只关心 $δφ$ 的**一阶线性项**，我们保留所有包含一个 $δφ^\wedge$ 的项，并忽略包含 $(δφ^\wedge)^2$ 或更高阶的项。

$ln(exp(δφ^) exp(φ^\wedge)) = δφ^\wedge + φ^\wedge + \frac{1}{2}[δφ^\wedge, φ^\wedge] + \frac{1}{12}([δφ^\wedge, [δφ^\wedge, φ^\wedge]] + [φ^\wedge, [φ^\wedge, δφ^\wedge]]) + \dots$

我们来分析每一项：
-   $φ^\wedge$: 零阶项，是我们的基准。
-   $δφ^\wedge$: 一阶项，与 $δφ$ 线性相关。
-   $\frac{1}{2}[δφ^\wedge, φ^\wedge]$: 一阶项，可以写成 $-\frac{1}{2}[φ^\wedge, δφ^\wedge] = -\frac{1}{2} \text{ad}(\phi^\wedge) \delta\phi^\wedge$。
-   $\frac{1}{12}[δφ^\wedge, [δφ^\wedge, φ^\wedge]]$ : 包含两个 $δφ^\wedge$ ，是二阶项，忽略
-   $\frac{1}{12}[φ^\wedge, [φ^\wedge, δφ^\wedge]]$ : 包含一个 $δφ^\wedge$ ，是一阶项，可以写成 $\frac{1}{12} \text{ad}(\phi^\wedge)^2 \delta\phi^\wedge$。

将所有一阶项收集起来，我们得到：
$$
\ln(e^{\delta\phi^{\wedge}} e^{\phi^{\wedge}}) = \delta\phi^{\wedge} + \phi^{\wedge} +  \frac{1}{2}[\delta\phi^{\wedge}, \phi^{\wedge}] + \frac{1}{12}[\phi^{\wedge}, [\phi^{\wedge}, \delta\phi^{\wedge}]] + \dots
$$
将李括号用 $ad$ 算子表示：
$$
\ln(e^{\delta\phi^{\wedge}} e^{\phi^{\wedge}}) = \phi^{\wedge} + \left( I - \frac{1}{2}\text{ad}(\phi^{\wedge}) + \frac{1}{12}\text{ad}(\phi^{\wedge})^2 - \dots \right) \delta\phi^{\wedge}
$$
这个无穷级数 $(I - \frac{1}{2}\text{ad}(\phi^{\wedge}) + \frac{1}{12}\text{ad}(\phi^{\wedge})^2 - \dots)$ 正是**左雅可比的逆 $$J_l(\phi)^{-1}$$ 的级数展开**。

我们用 $∨$ 算子将 $so(3)$ 矩阵变回 $R³$ 向量：
$$
\ln(e^{\delta\phi^{\wedge}} e^{\phi^{\wedge}})^{\vee} = \phi + J_l(\phi)^{-1} \delta\phi
$$
现在，我们将 $φ$ 替换回 $φ₂$，将 $δφ$ 替换回 $φ₁$：
$$
\ln(e^{\phi_1^{\wedge}} e^{\phi_2^{\wedge}})^{\vee} = \phi_2 + J_l(\phi_2)^{-1} \phi_1
$$
这就证明了您的第一个公式。

**左雅可比的逆 $J_l(φ)⁻¹$ 的闭式形式为**：
$$
J_l(\phi)^{-1} = I - \frac{1}{2}\phi^{\wedge} + \left(\frac{1}{\theta^{\wedge}2} - \frac{1+\cos\theta}{2\theta\sin\theta}\right)(\phi^{\wedge})^2, \quad \theta = ||\phi||
$$
当 $φ$ 很小时，$J_l(φ)⁻¹ ≈ I - 1/2 φ^\wedge$，这与我们BCH展开的前两项吻合。

---

### 推导二：右扰动模型（$φ₂$ 为小量）

现在我们分析 $ln(exp(φ₁^\wedge) exp(φ₂^\wedge))$，其中 $φ₂$ 是一个微小的扰动量 $δφ$，$φ₁$ 是基准旋转 $φ$。

所以，我们要分析的是： $ln(exp(φ^\wedge) exp(δφ^\wedge))$。

这里我们介绍一个非常巧妙的技巧，它利用了我们刚刚推导出的左扰动结果，而无需重新展开BCH。
考虑任意两个旋转矩阵 $R₁ = exp(φ^\wedge)$ 和 $R₂ = exp(δφ^\wedge)$。
我们想计算 $ln(R₁R₂)$。
利用矩阵乘积和逆的性质：
$ln(R₁R₂) = ln((R₂⁻¹ R₁⁻¹)⁻¹) = -ln(R₂⁻¹ R₁⁻¹)$

因为 $R₁ = exp(φ^\wedge)$，所以 $R₁⁻¹ = exp(-φ^\wedge)$。
因为 $R₂ = exp(δφ^\wedge)$，所以 $R₂⁻¹ = exp(-δφ^\wedge)$。

代入上式：
$$
\ln(e^{\phi^{\wedge}} e^{\delta\phi^{\wedge}})^{\vee} = - \ln(e^{-\delta\phi^{\wedge}} e^{-\phi^{\wedge}})^{\vee}
$$
观察右边的 $-ln(...)$ 部分。$exp(-δφ^{\wedge}) exp(-φ^{\wedge})$ 正是一个**左扰动**形式！其中扰动量是 $-δφ$，基准旋转是 $-φ$。

我们可以直接套用第一部分的推导结果：
$ln(exp(扰动) exp(基准))∨ ≈ 基准 + J_l(基准)⁻¹ (扰动)$

将 $扰动 = -δφ$ 和 $基准 = -φ$ 代入：
$$
\ln(e^{-\delta\phi^{\wedge}} e^{-\phi^{\wedge}})^{\vee} = -\phi + J_l(-\phi)^{-1} (-\delta\phi)
$$
现在把它代回我们的原式：
$$
\ln(e^{\phi^{\wedge}} e^{\delta\phi^{\wedge}})^{\vee} = - \left( -\phi - J_l(-\phi)^{-1} \delta\phi \right) = \phi + J_l(-\phi)^{-1} \delta\phi
$$
这里，我们引入一个重要的雅可比恒等式：**右雅可比等于负参数的左雅可比**。
$$
J_r(\phi) = J_l(-\phi)
$$
因此，$J_r(φ)⁻¹ = J_l(-φ)⁻¹$。

将这个恒等式代入我们的推导结果：
$$
\ln(e^{\phi^{\wedge}} e^{\delta\phi^{\wedge}})^{\vee} = \phi + J_r(\phi)^{-1} \delta\phi
$$
现在，我们将 $φ$ 替换回 $φ₁$，将 $δφ$ 替换回 $φ₂$：
$$
\ln(e^{\phi_1^{\wedge}} e^{\phi_2^{\wedge}})^{\vee} = \phi_1 + J_r(\phi_1)^{-1} \phi_2
$$
这就证明了您的第二个公式。

### 总结

-   **左扰动**：当一个微小旋转 $exp(φ₁^{\wedge})$ **左乘**到一个已有旋转 $exp(φ₂^{\wedge})$ 上时，其在李代数上的增量由**左雅可比的逆 $J_l(φ₂)⁻¹$** 线性映射。
-   **右扰动**：当一个微小旋转 $exp(φ₂^{\wedge})$ **右乘**到一个已有旋转 $exp(φ₁^{\wedge})$ 上时，其在李代数上的增量由**右雅可比的逆 $J_r(φ₁)⁻¹$** 线性映射。

这些推导精确地展示了BCH公式是如何作为理论基石，导出在实际优化（如SLAM中的位姿图优化）中至关重要的线性化模型，即扰动模型中使用的雅可比矩阵。