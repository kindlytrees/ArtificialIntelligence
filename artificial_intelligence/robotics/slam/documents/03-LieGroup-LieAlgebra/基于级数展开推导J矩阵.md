# 基于级数级数展开推导J矩阵

以下回答内容基于大模型反馈结果

基于指数展开能让我们深刻理解BCH公式、指数映射和雅可比矩阵之间的内在联系。我们将完全基于级数展开来推导左雅可比 $\boldsymbol{J}_l(\boldsymbol{\phi})$。

这里的参数我们用 $\boldsymbol{\phi}$ 表示，以与旋转角 $\theta$ 区分。

### 目标

我们的目标是推导出 **左雅可比 $\boldsymbol{J}_l(\boldsymbol{\phi})$** 的级数形式。这个雅可比矩阵的定义来自于以下关系：
一个在参数空间的微小变化 $\delta\boldsymbol{\phi}$，通过指数映射，可以等效于在当前姿态 $\exp(\boldsymbol{\phi}^\wedge)$ 上左乘一个微小的物理旋转 $\delta\boldsymbol{\epsilon}$。
$$
\underbrace{\exp((\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge)}_{\text{参数空间中的新姿态}} \approx \underbrace{\exp(\delta\boldsymbol{\epsilon}^\wedge)}_{\text{物理上的微小扰动}} \underbrace{\exp(\boldsymbol{\phi}^\wedge)}_{\text{旧姿态}}
$$
左雅可比 $\boldsymbol{J}_l(\boldsymbol{\phi})$ 正是连接这两个微小量 $\delta\boldsymbol{\phi}$ 和 $\delta\boldsymbol{\epsilon}$ 的桥梁：
$$
\delta\boldsymbol{\epsilon} = \boldsymbol{J}_l(\boldsymbol{\phi}) \delta\boldsymbol{\phi}
$$
我们的任务就是找出这个 $\boldsymbol{J}_l(\boldsymbol{\phi})$ 的表达式。

### 推导过程

推导的关键是**对同一个量用两种方式展开，然后对比系数**。这个量就是新姿态 $\exp((\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge)$。

#### 步骤 1: 展开右侧 $\exp(\delta\boldsymbol{\epsilon}^\wedge) \exp(\boldsymbol{\phi}^\wedge)$

我们直接使用BCH公式，其中 $X = \delta\boldsymbol{\epsilon}^\wedge$，$Y = \boldsymbol{\phi}^\wedge$。我们只关心 $\delta\boldsymbol{\epsilon}$ 的一阶项。
$$
\log(\exp(\delta\boldsymbol{\epsilon}^\wedge) \exp(\boldsymbol{\phi}^\wedge)) = \delta\boldsymbol{\epsilon}^\wedge + \boldsymbol{\phi}^\wedge + \frac{1}{2}[\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge] - \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge]] + \dots
$$
所以，右侧可以近似写成：
$$
\exp(\delta\boldsymbol{\epsilon}^\wedge) \exp(\boldsymbol{\phi}^\wedge) \approx \exp\left( \boldsymbol{\phi}^\wedge + \delta\boldsymbol{\epsilon}^\wedge + \frac{1}{2}[\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge] - \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge]] + \dots \right)
$$

#### 步骤 2: 展开左侧 $\exp((\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge)$

左侧的展开相对简单。我们假设 $\delta\boldsymbol{\phi}$ 是一个微小量，直接进行泰勒展开。令 $f(\boldsymbol{\phi}) = \exp(\boldsymbol{\phi}^\wedge)$，我们想求 $f(\boldsymbol{\phi}+\delta\boldsymbol{\phi})$。
这实际上就是指数映射的微分。其标准形式为：
$$
\exp((\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge) \approx \exp(\boldsymbol{\phi}^\wedge) + \text{DerivativeTerm}
$$
一个更直接的方法是，我们认为左侧的李代数就是 $\boldsymbol{\phi}^\wedge + \delta\boldsymbol{\phi}^\wedge$。

#### 步骤 3: 建立等式并求解

现在我们让左右两侧指数函数内部的李代数相等：
$$
(\boldsymbol{\phi} + \delta\boldsymbol{\phi})^\wedge = \boldsymbol{\phi}^\wedge + \delta\boldsymbol{\phi}^\wedge \approx \boldsymbol{\phi}^\wedge + \delta\boldsymbol{\epsilon}^\wedge + \frac{1}{2}[\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge] - \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge]] + \dots
$$
消去 $\boldsymbol{\phi}^\wedge$：
$$
\delta\boldsymbol{\phi}^\wedge \approx \delta\boldsymbol{\epsilon}^\wedge + \frac{1}{2}[\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge] - \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\delta\boldsymbol{\epsilon}^\wedge, \boldsymbol{\phi}^\wedge]] + \dots
$$
为了方便，我们使用李代数的**伴随表示 (Adjoint representation)** $\text{ad}(\cdot)$。对于 $\mathfrak{so}(3)$，我们有：
*   $[\boldsymbol{A}, \boldsymbol{B}] = \text{ad}(\boldsymbol{A})\boldsymbol{B}$
*   $[\boldsymbol{a}^\wedge, \boldsymbol{b}^\wedge] = (\boldsymbol{a} \times \boldsymbol{b})^\wedge$
*   $\text{ad}(\boldsymbol{a}^\wedge)\boldsymbol{b}^\wedge = (\boldsymbol{a} \times \boldsymbol{b})^\wedge$
所以，$\text{ad}(\boldsymbol{\phi}^\wedge)$ 作用在一个反对称矩阵 $\boldsymbol{v}^\wedge$ 上，等效于 $\boldsymbol{\phi}^\wedge$ 作用在向量 $\boldsymbol{v}$ 上。

将上式转换为伴随表示的形式：
$$
\delta\boldsymbol{\phi}^\wedge \approx \delta\boldsymbol{\epsilon}^\wedge - \frac{1}{2}[\boldsymbol{\phi}^\wedge, \delta\boldsymbol{\epsilon}^\wedge] - \frac{1}{12}[\boldsymbol{\phi}^\wedge, [\boldsymbol{\phi}^\wedge, \delta\boldsymbol{\epsilon}^\wedge]] + \dots
$$
$$
\delta\boldsymbol{\phi}^\wedge \approx \left( \boldsymbol{I} - \frac{1}{2}\text{ad}(\boldsymbol{\phi}^\wedge) + \frac{1}{12}\text{ad}(\boldsymbol{\phi}^\wedge)^2 - \dots \right) \delta\boldsymbol{\epsilon}^\wedge
$$
去掉两边的 `^` 算子，我们就得到了参数空间和物理空间微小量的关系：
$$
\delta\boldsymbol{\phi} \approx \left( \boldsymbol{I} - \frac{1}{2}\text{ad}(\boldsymbol{\phi}) + \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2 - \dots \right) \delta\boldsymbol{\epsilon}
$$

#### 步骤 4: 求逆以得到 $\boldsymbol{J}_l(\boldsymbol{\phi})$

回顾我们的目标：$\delta\boldsymbol{\epsilon} = \boldsymbol{J}_l(\boldsymbol{\phi}) \delta\boldsymbol{\phi}$。
而我们刚刚推导出的是 $\delta\boldsymbol{\phi}$ 关于 $\delta\boldsymbol{\epsilon}$ 的表达式。所以，我们需要求上面那个大矩阵的**逆**。

令 $\boldsymbol{A} = \boldsymbol{I} - \frac{1}{2}\text{ad}(\boldsymbol{\phi}) + \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2 - \dots$
则 $\boldsymbol{J}_l(\boldsymbol{\phi}) = \boldsymbol{A}^{-1}$。

我们使用一个经典的级数求逆公式：对于矩阵 $\boldsymbol{X}$，如果其范数小于1，则 $(\boldsymbol{I}-\boldsymbol{X})^{-1} = \boldsymbol{I} + \boldsymbol{X} + \boldsymbol{X}^2 + \dots$。
在这里，令 $\boldsymbol{X} = \frac{1}{2}\text{ad}(\boldsymbol{\phi}) - \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2 + \dots$
$$
\boldsymbol{J}_l(\boldsymbol{\phi}) = (\boldsymbol{I} - \boldsymbol{X})^{-1} \approx \boldsymbol{I} + \boldsymbol{X} + \boldsymbol{X}^2 + \dots
$$
我们来计算级数的前几项：
*   **$\boldsymbol{I}$**: 就是 $\boldsymbol{I}$。
*   **$\boldsymbol{X}$**: 就是 $\frac{1}{2}\text{ad}(\boldsymbol{\phi}) - \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2 + \dots$
*   **$\boldsymbol{X}^2$**: 我们只关心最低阶项，即 $(\frac{1}{2}\text{ad}(\boldsymbol{\phi}))^2 = \frac{1}{4}\text{ad}(\boldsymbol{\phi})^2$。

将它们相加：
$$
\begin{aligned}
\boldsymbol{J}_l(\boldsymbol{\phi}) &\approx \boldsymbol{I} + \left(\frac{1}{2}\text{ad}(\boldsymbol{\phi}) - \frac{1}{12}\text{ad}(\boldsymbol{\phi})^2\right) + \frac{1}{4}\text{ad}(\boldsymbol{\phi})^2 + \dots \\
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