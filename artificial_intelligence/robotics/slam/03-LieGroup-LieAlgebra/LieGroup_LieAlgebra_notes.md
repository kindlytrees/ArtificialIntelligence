# Lie Group and Lie Algebra notes

- 雅可比的理解:SO(3)的左雅可比。这个公式正是描述了在李代数中，一个小的增量 δφ 如何通过左乘的方式疊加到现有向量 φ 上。

$$
\exp \left((\phi+\Delta \phi)^{\wedge}\right)=\exp \left(\left(J_l \Delta \phi\right)^{\wedge}\right) \exp \left(\phi^{\wedge}\right)=\exp \left(\phi^{\wedge}\right) \exp \left(\left(J_r \Delta \phi\right)^{\wedge}\right)
$$

- J的计算，亦即SO(3)的左雅可比的计算

$$
\begin{aligned}
\sum_{n=0}^{\infty} \frac{1}{(n+1)!}\left(\phi^{\wedge}\right)^n & =\boldsymbol{I}+\frac{1}{2!} \theta \boldsymbol{a}^{\wedge}+\frac{1}{3!} \theta^2\left(\boldsymbol{a}^{\wedge}\right)^2+\frac{1}{4!} \theta^3\left(\boldsymbol{a}^{\wedge}\right)^3+\frac{1}{5!} \theta^4\left(\boldsymbol{a}^{\wedge}\right)^4 \cdots \\
& =\frac{1}{\theta}\left(\frac{1}{2!} \theta^2-\frac{1}{4!} \theta^4+\cdots\right)\left(\boldsymbol{a}^{\wedge}\right)+\frac{1}{\theta}\left(\frac{1}{3!} \theta^3-\frac{1}{5} \theta^5+\cdots\right)\left(\boldsymbol{a}^{\wedge}\right)^2+\boldsymbol{I} \\
& =\frac{1}{\theta}(1-\cos \theta)\left(\boldsymbol{a}^{\wedge}\right)+\frac{\theta-\sin \theta}{\theta}\left(\boldsymbol{a} \boldsymbol{a}^T-\boldsymbol{I}\right)+\boldsymbol{I} \\
& =\frac{\sin \theta}{\theta} \boldsymbol{I}+\left(1-\frac{\sin \theta}{\theta}\right) \boldsymbol{a} \boldsymbol{a}^T+\frac{1-\cos \theta}{\theta} \boldsymbol{a}^{\wedge} \triangleq \boldsymbol{J}
\end{aligned}
$$

- Sim(3)变换群的相关理解和应用
主要用在单目场景下，解决尺度缩放的问题，基于回环检测的结果进行矫正，orbslam3代码中有实现

## Outline
- 关键知识点
- 相关问题
    - 指数映射的展开
    - 雅可比矩阵的一些相关计算及理解
    - BCH的推导
    - 李代数的导数和扰动模型的区别的理解
    - 请给出从BCH一阶近似得出雅可比积分公式的详细推导

### 问题

#### 问题1：自然指数的泰勒展开得出自然指数的幂级数形式，请对se(3)的李代数的指数映射进行泰勒展开的推导，并对上述的断言给出更多的解释说明。

$$
\begin{aligned}
\exp \left(\phi^{\wedge}\right) & =\exp \left(\theta \boldsymbol{a}^{\wedge}\right)=\sum_{n=0}^{\infty} \frac{1}{n!}\left(\theta \boldsymbol{a}^{\wedge}\right)^n \\
& =\boldsymbol{I}+\theta \boldsymbol{a}^{\wedge}+\frac{1}{2!} \theta^2 \boldsymbol{a}^{\wedge} \boldsymbol{a}^{\wedge}+\frac{1}{3!} \theta^3 \boldsymbol{a}^{\wedge} \boldsymbol{a}^{\wedge} \boldsymbol{a}^{\wedge}+\frac{1}{4!} \theta^4\left(\boldsymbol{a}^{\wedge}\right)^4+\ldots \\
& =\boldsymbol{a} \boldsymbol{a}^{\mathrm{T}}-\boldsymbol{a}^{\wedge} \boldsymbol{a}^{\wedge}+\theta \boldsymbol{a}^{\wedge}+\frac{1}{2!} \theta^2 \boldsymbol{a}^{\wedge} \boldsymbol{a}^{\wedge}-\frac{1}{3!} \theta^3 \boldsymbol{a}^{\wedge}-\frac{1}{4!} \theta^4\left(\boldsymbol{a}^{\wedge}\right)^2+\ldots \\
& =\boldsymbol{a} \boldsymbol{a}^{\mathrm{T}}+\underbrace{\left(\theta-\frac{1}{3!} \theta^3+\frac{1}{5!} \theta^5-\ldots\right)}_{\sin \theta} \boldsymbol{a}^{\wedge}-\underbrace{\left(1-\frac{1}{2!} \theta^2+\frac{1}{4!} \theta^4-\ldots\right)}_{\cos \theta} \boldsymbol{a}^{\wedge} \boldsymbol{a}^{\wedge} \\
& =\boldsymbol{a}^{\wedge} \boldsymbol{a}^{\wedge}+\boldsymbol{I}+\sin \theta \boldsymbol{a}^{\wedge}-\cos \theta \boldsymbol{a}^{\wedge} \boldsymbol{a}^{\wedge} \\
& =(1-\cos \theta) \boldsymbol{a}^{\wedge} \boldsymbol{a}^{\wedge}+\boldsymbol{I}+\sin \theta \boldsymbol{a}^{\wedge} \\
& =\cos \theta \boldsymbol{I}+(1-\cos \theta) \boldsymbol{a} \boldsymbol{a}^{\mathrm{T}}+\sin \theta \boldsymbol{a}^{\wedge}
\end{aligned}
$$

#### 问题2：雅可比矩阵的理解

是的，您提出的这个公式**完美地、极其精准地**阐述了李群雅可比的几何意义和其作为“一阶近似”的本质。这组等式是理解李群与李代数在运动、优化和控制中如何应用的核心。

我们可以从三个层面来解读这个公式：**几何含义**、**数学近似**以及**计算推导**。

---

### 1. 几何含义：连接李代数空间的“加法”与李群空间的“乘法”

这个公式揭示了一个深刻的几何事实：

*   **李代数空间 (`so(3)` 或 `se(3)`) 是一个向量空间（欧氏空间）。** 在这个空间里，我们可以像普通向量一样进行加法运算，例如 `φ + Δφ`。这代表了“在原有旋转向量 `φ` 的基础上，增加一个小的旋转向量 `Δφ`”。

*   **李群空间 (`SO(3)` 或 `SE(3)`) 是一个弯曲的流形。** 在这个空间里，没有全局的“加法”。基本运算是**乘法**，即旋转的复合。`exp(φ₁^) exp(φ₂^)` 代表了“先执行旋转 `φ₂`，再执行旋转 `φ₁`”。

您给出的公式：

$$
\exp \left((\phi+\Delta \phi)^{\wedge}\right) \approx \exp \left(\left(J_l(\phi) \Delta \phi\right)^{\wedge}\right) \exp \left(\phi^{\wedge}\right)
$$

其几何意义是：
> 在李代数中对 `φ` 进行一个微小的线性增量 `Δφ`，其效果（通过指数映射`exp`）近似等于在李群中，对现有的旋转 `exp(φ^)` **左乘**上一个由 `Δφ` 变换而来的微小旋转。

这个“变换”就是**左雅可比 `J_l(φ)`**。它扮演了一个“校正因子”或“度量变换”的角色，将李代数空间中一个“直”的微小位移 `Δφ`，转换成李群流形上对应位置的一个“弯”的微小位移。

同样地，对于右雅可比：

$$
\exp \left((\phi+\Delta \phi)^{\wedge}\right) \approx \exp \left(\phi^{\wedge}\right) \exp \left(\left(J_r(\phi) \Delta \phi\right)^{\wedge}\right)
$$

其几何意义是：
> 在李代-数中对 `φ` 进行一个微小的线性增量 `Δφ`，其效果近似等于在李群中，对现有的旋转 `exp(φ^)` **右乘**上一个由 `Δφ` 变换而来的微小旋转。这个变换就是**右雅可比 `J_r(φ)`**。

**一言以蔽之：雅可比矩阵 `J_l` 和 `J_r` 是将李代数空间的线性加法 `+` 映射到李群空间的乘法 `*` 的“局部线性化桥梁”。**

---

### 2. 数学近似：这是一阶泰勒展开

您在公式中使用的约等号 `≈` (或者在严谨推导中应写作一阶近似) 是非常关键的。这个公式本身就是**指数映射 `exp` 在点 `φ` 处的一阶泰勒展开**。

让我们考虑一个函数 `f(φ) = exp(φ^)`。我们想在 `φ` 点附近对它进行泰勒展开：
`f(φ + Δφ) ≈ f(φ) + Df(φ)[Δφ]`
其中 `Df(φ)` 是 `f` 在 `φ` 处的导数（雅可比）。

这里的困难在于 `f(φ)` 的输出是矩阵，而输入是向量，并且加法和乘法发生在不同空间。上述公式 `exp((φ+Δφ)^) ≈ exp((J_l Δφ)^) exp(φ^)` 正是这个泰勒展开在李群上的正确形式。

让我们用更熟悉的形式重写它。当 `Δφ` 是无穷小时 `dφ`：
`exp((φ + dφ)^) = exp( (J_l(φ) dφ)^ ) exp(φ^)`
两边同时右乘 `exp(φ^)⁻¹ = exp(-φ^)`：
`exp((φ + dφ)^) exp(-φ^) = exp( (J_l(φ) dφ)^ )`

当 `dφ` 无穷小时，`exp((J_l(φ)dφ)^) ≈ I + (J_l(φ)dφ)^`。
所以，这个公式本质上是在描述指数映射的**微分（tangent map）**。它定义了李代数 `so(3)`（即 `SO(3)` 在单位元处的切空间）与 `SO(3)` 在任意点 `exp(φ^)` 处的切空间之间的映射关系。

---

### 3. 计算推导：通过级数展开得出雅可比

是的，这个近似关系式正是推导雅可比 `J_l` 和 `J_r` 闭式解的出发点之一。下面我们简述如何利用这个关系来找到 `J_l` 的级数。

我们的目标是找到一个矩阵 `J_l(φ)` 使得下式在一阶近似下成立：
$$
\exp((\phi + \Delta\phi)^\wedge) \approx (I + (J_l(\phi)\Delta\phi)^\wedge) \exp(\phi^\wedge)
$$
(这里我们对 `exp((J_l Δφ)^)` 也做了一阶近似 `I + ...`)

这个过程比较复杂，涉及到所谓的 **Duhamel's formula** 或 **variation of parameters formula** for matrix exponentials。一个更直接的方法是回到我们之前讨论的积分形式。

但我们可以从另一个角度，即**对数映射的微分**，来反向验证这个公式。
我们之前推导了BCH近似：
$$
\ln(e^{\delta\phi^\wedge} e^{\phi^\wedge})^\vee \approx \phi + J_l(\phi)^{-1} \delta\phi
$$
令 `φ_new = ln(exp(δφ^) exp(φ^))∨`。这个 `φ_new` 对应于您公式中的 `φ + Δφ`，但它们并不完全相等，而是近似相等。
`φ_new ≈ φ + Δφ`
这里的 `δφ` 对应于 `J_l(φ)Δφ`。
代入BCH近似公式：
`φ_new ≈ φ + J_l(φ)⁻¹ (J_l(φ)Δφ) = φ + Δφ`
这表明了整个体系的自洽性。

**如何从这个近似关系式直接推导 `J_l` 的级数？**
这个过程通常出现在更高级的李群教材中，例如通过求解一个关于 `t` 的微分方程：
令 `R(t) = exp(tφ^)`。我们想知道 `d/dt R(t)`。
`d/dt exp(tφ^) = lim_{h->0} (exp((t+h)φ^) - exp(tφ^)) / h`
`= lim (exp(tφ^) exp(hφ^) - exp(tφ^)) / h` (使用BCH一阶近似)
`= exp(tφ^) lim (exp(hφ^) - I) / h`
`= exp(tφ^) φ^`

这个结果是右乘形式。要得到左雅可比，需要更复杂的 **Magnus expansion** 或 **Wei-Norman method**，它们最终都会导向我们之前看到的积分形式：
$$
J_l(\phi) = \int_0^1 \exp(s \cdot \text{ad}(\phi^\wedge)) ds = \sum_{n=0}^\infty \frac{1}{(n+1)!} \text{ad}(\phi^\wedge)^n
$$
将 `ad(φ^)` 的矩阵表示 `φ^` 代入，就得到了您之前给出的级数：
$$
J_l(\phi) = \sum_{n=0}^\infty \frac{1}{(n+1)!} (\phi^\wedge)^n
$$

### 结论

您提出的公式 `exp((φ+Δφ)^) ≈ exp((J_lΔφ)^)exp(φ^)` 是对雅可比作用最核心、最直观的描述：

1.  **几何上**，它解释了 `J_l` 和 `J_r` 是如何将李代数中的“平直”加法，转化为李群流形上的“弯曲”乘法。
2.  **数学上**，它就是指数映射在李群设定下的一阶泰勒展开，是所有扰动模型和基于优化的机器人算法（如SLAM，标定，控制）的理论基石。
3.  **计算上**，这个关系式以及其微分形式，是推导雅可比级数展开和最终闭式解的出发点。通过对这个关系进行严格的数学处理（如积分、解微分方程），我们最终能得到雅可比的具体计算公式。


#### 问题3：bch完整的公式是不是在指数的泰勒展开的多项式乘法展开中省略了二阶以上的对称项，请对BCH的公式进行推导并进行详细说明。

您提出了一个非常深刻的问题，并且您的直觉方向是正确的，但结论需要更精确的阐述。BCH公式的推导并非简单地“省略二阶以上的对称项”，而是系统地处理 **非交换乘法** 下泰勒级数展开和合并的结果。

我们来详细推导BCH公式的前几项，您会清晰地看到李括号 $[X, Y]$ 是如何从非交换性中自然产生的。

---

### 1. 核心思想与工具

**核心思想**：我们想要找到一个 $Z$，使得 $e^Z = e^X e^Y$。通过对数函数，我们可以得到 $Z = \log(e^X e^Y)$。我们的目标就是将这个表达式展开，并用 $X$ 和 $Y$ 的组合来表示 $Z$。

**所需工具**：矩阵指数函数和对数函数的泰勒级数展开。
1.  **指数函数展开**： $e^A = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \dots$
2.  **对数函数展开** (对于靠近单位矩阵的 $I+B$)： $\log(I+B) = B - \frac{B^2}{2} + \frac{B^3}{3} - \dots$

**关键前提**： $X$ 和 $Y$ 是李代数中的元素（例如，$4\times4$ 的矩阵），它们的乘法是**非交换的**，即 $XY \neq YX$。这是整个推导的核心。

---

#### 问题3：BCH 公式的推导（前几项）

我们的目标是计算 $Z = \log(e^X e^Y)$。

**第一步：展开 $e^X e^Y$**

我们先将 $e^X$ 和 $e^Y$ 分别进行泰勒展开，然后相乘。为了推导出包含第一个李括号的项，我们至少需要展开到二阶。

 $e^X \approx I + X + \frac{X^2}{2}$
 $e^Y \approx I + Y + \frac{Y^2}{2}$

现在，将这两个多项式相乘，**并严格保持乘法顺序**：
 $e^X e^Y \approx (I + X + \frac{X^2}{2}) (I + Y + \frac{Y^2}{2})$
 $= I(I + Y + \frac{Y^2}{2}) + X(I + Y + \frac{Y^2}{2}) + \frac{X^2}{2}(I + Y + \frac{Y^2}{2})$
 $= (I + Y + \frac{Y^2}{2}) + (X + XY + \frac{XY^2}{2}) + (\frac{X^2}{2} + \frac{X^2Y}{2} + \frac{X^2Y^2}{4})$

现在，我们按阶数（$X$ 和 $Y$ 的幂次之和）来收集项，并忽略三阶及以上的项：
*   **零阶**： $I$
*   **一阶**： $X + Y$
*   **二阶**： $\frac{X^2}{2} + XY + \frac{Y^2}{2}$

所以，我们得到 $e^X e^Y$ 的二阶近似： $e^X e^Y \approx I + \underbrace{(X+Y) + \frac{1}{2}(X^2 + 2XY + Y^2)}_{\text{记为 } B}$

**第二步：代入对数函数 $\log(I+B)$**

现在我们计算 $Z = \log(e^X e^Y) \approx \log(I+B)$。使用对数函数的泰勒展开 $\log(I+B) = B - \frac{B^2}{2} + \dots$

代入 $B$： $Z \approx \left[ (X+Y) + \frac{1}{2}(X^2 + 2XY + Y^2) \right] - \frac{1}{2} \left[ (X+Y) + \dots \right]^2 + \dots$

我们只需要计算到二阶，所以第二项中的 $B^2$ 只需要计算其最低阶部分，即 $(X+Y)^2$。
 $(X+Y)^2 = (X+Y)(X+Y) = X^2 + XY + YX + Y^2$  **(注意这里的 $XY$ 和 $YX$ 不能合并！)**

现在将这个结果代回 $Z$ 的表达式： $Z\approx \left[ (X+Y) + \frac{1}{2}X^2 + XY + \frac{1}{2}Y^2 \right] - \frac{1}{2} [X^2 + XY + YX + Y^2]$

**第三步：合并同类项，见证奇迹**

我们来仔细合并所有二阶项：
*   $X^2$ 项:  $(\frac{1}{2}X^2) - (\frac{1}{2}X^2) = 0$
*   $Y^2$ 项:  $(\frac{1}{2}Y^2) - (\frac{1}{2}Y^2) = 0$
*   $XY$ 项:  $(XY) - (\frac{1}{2}XY) = \frac{1}{2}XY$
*   $YX$ 项:  $0 - (\frac{1}{2}YX) = -\frac{1}{2}YX$

将所有项合并起来，我们得到 $Z$ 的最终近似表达式：
$ Z \approx (X+Y) + (\frac{1}{2}XY - \frac{1}{2}YX) $
$ Z \approx X + Y + \frac{1}{2}(XY - YX) $

这个 $(XY - YX)$ 正是**李括号 $[X, Y]$** 的定义！

所以，我们推导出了BCH公式的前三项： $\log(e^X e^Y) \approx X + Y + \frac{1}{2}[X, Y]$

---

##### 3. 详细说明与回答您的问题

1.  **为什么不是“省略对称项”？**
    您可以看到，在推导过程中，像 $X^2$ 和 $Y^2$ 这样“对称形式”的项，以及 $(X+Y)$ 展开中的部分对称项，是**自然抵消**掉了。但最终留下的项 $\frac{1}{2}(XY-YX)$ 是一个**反对称**的项。
    所以，更准确的说法是：**BCH公式揭示了李群乘法与李代数加法的差异，这个差异完全由一系列反对称的李括号项来描述。**
    如果 $X$ 和 $Y$ 是可交换的（例如普通数字），那么 $XY=YX$，李括号为零，所有修正项都消失，BCH公式退化为 $Z = X+Y$，这符合我们的直觉 ($e^x e^y = e^{x+y}$)。

2.  **高阶项的来源**
    如果我们想得到更高阶的项，比如 $\frac{1}{12}[X,[X,Y]]$，我们需要：
    *   将 $e^X, e^Y$ 的泰勒展开保留到三阶。
    *   将 $\log(I+B)$ 的泰勒展开也保留到三阶。
    *   进行极其繁琐的代数运算，合并所有同阶项。
    最终会发现，所有剩余的修正项都可以被巧妙地组织成嵌套的李括号形式。这是一个非常深刻且优美的数学结果。

##### 结论

BCH公式的推导是一个严谨的代数过程，它精确地量化了“指数的乘积”与“指数的和的指数”之间的差异。这个差异的根源是**非交换性**。

推导过程的核心是：
1.  **展开**：将 $e^X$, $e^Y$ 和 $\log(\cdot)$ 进行泰勒展开。
2.  **相乘**：计算 $e^X e^Y$ 的展开式，严格保持乘法顺序。
3.  **合并**：将结果代入 $\log$ 的展开式，并按阶数重新组合。
4.  **识别**：发现所有修正项都可以表示为李括号 $[X,Y]=XY-YX$ 及其嵌套组合。

这个公式为我们提供了从李群的非线性乘法空间，到李代数的线性加法空间的桥梁，并清楚地表明，只有当扰动非常小（高阶李括号项可以忽略）时，我们才能安全地使用线性近似（例如 $T' \approx T + \widehat{\delta\boldsymbol{\xi}}T$），这正是扰动模型计算简洁性的数学基础。