# NonLinear Optimization notes

## Outline
- 线性最小二乘如线性回归可以有解析解，线性回归问题中定义的函数为线性函数，参数变量空间和数据空间为线性关系，
    - 线性回归问题可以有解析解
    - 也可以基于随机梯度下降进行求解
- 非线性最小二乘中问题优化目标定义为非线性函数，如李群李代数表示的运动与观测方程中，观测结果如图像中像素点与数据空间的关系为非线性的关系，表现为
    - 一，成像模型中要除以深度z为非线性操作
    - 二，欧式变换为约束高的SE(3)参数变量空间，其中的旋转变换约束为正交基且行列式为1的李群流形，或者其对应为的李代数的指数映射，也可以看成是对应到罗德里格斯公式去理解，其中的theta角度为李代数的模长，因此也引入了非线性）

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