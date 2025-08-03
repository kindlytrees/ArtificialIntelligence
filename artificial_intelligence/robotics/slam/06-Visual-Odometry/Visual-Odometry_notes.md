# Visual Odometry notes

## 对极几何的更详细的介绍
- 对极几何在ORB-SLAM3中用于初始化时两帧图像之间的关键点之间的匹配，来进一步初始化相机位姿

$$
\begin{aligned}
&\begin{aligned}
& p_1 \simeq K P, \quad p_2 \simeq K(R P+t) \\
& x_1=K^{-1} p_1, \quad x_2=K^{-1} p_2 . \\
& x_2 \simeq R x_1+t .
\end{aligned}\\
&\boldsymbol{t}^{\wedge} \boldsymbol{x}_2 \simeq \boldsymbol{t}^{\wedge} \boldsymbol{R} \boldsymbol{x}_1\\
&\boldsymbol{x}_2^{\mathrm{T}} \boldsymbol{t}^{\wedge} \boldsymbol{x}_2 \simeq \boldsymbol{x}_2^{\mathrm{T}} \boldsymbol{t}^{\wedge} \boldsymbol{R} \boldsymbol{x}_1 \cdot \boldsymbol{t}^{\wedge} \boldsymbol{x}_2 \text { 是一个与 } \boldsymbol{t} \text { 和 } \boldsymbol{x}_2 \text { 都垂直的向量 }\\
&\boldsymbol{x}_2^{\mathrm{T}} \boldsymbol{t}^{\wedge} \boldsymbol{R} \boldsymbol{x}_1=0 \longrightarrow \boldsymbol{p}_2^{\mathrm{T}} \boldsymbol{K}^{-\mathrm{T}} \boldsymbol{t}^{\wedge} \boldsymbol{R} \boldsymbol{K}^{-1} \boldsymbol{p}_1=0 .
\end{aligned}
$$


辅助大模型回答：

好的，我们来一步步详细拆解这些对极几何的公式，并解释其背后的几何原理。这是一个从相机成像模型到最终**对极约束**方程的完整推导过程。

### 0. 场景设定

我们有两个相机，相机1和相机2，它们从不同的位置和角度观察同一个三维空间点 $P$。

*   **世界坐标系**: $P$ 在世界坐标系下的坐标是 $\mathbf{P}_w$。为了简化，我们通常将**相机1的坐标系**设为世界坐标系。所以，三维点 $P$ 在相机1坐标系下的坐标就是 $\mathbf{P}$。
*   **相机位姿**:
    *   相机1的位姿是单位矩阵 $[I | \mathbf{0}]$。
    *   相机2相对于相机1的位姿由旋转矩阵 $R$ 和平移向量 $t$ 描述。一个在相机1坐标系下的点 $\mathbf{P}$，在相机2坐标系下的坐标是 $\mathbf{P}' = R\mathbf{P} + t$。
*   **相机内参**: 我们假设两个相机的内参矩阵 $K$ 是相同的（或分别用 $K_1, K_2$）。$K$ 将相机坐标系下的3D点投影到像素平面。
*   **像素坐标**: $p_1$ 和 $p_2$ 分别是点 $P$ 在相机1和相机2图像上的**像素坐标**（齐次坐标，3x1向量）。
*   **归一化平面坐标**: $x_1$ 和 $x_2$ 是点 $P$ 在各自相机**归一化平面**（焦距为1的平面）上的坐标（齐次坐标，3x1向量）。

---

### 1. 第一组公式：从3D点到像素点，再到归一化平面

$ \begin{aligned}
p_1 &\simeq K \mathbf{P} \\
p_2 &\simeq K(R \mathbf{P}+t)
\end{aligned} $

**原理说明**:
*   这是**相机投影模型**的基本公式。
*   **第一行**: $p_1 \simeq K \mathbf{P}$。点 $P$ 在相机1坐标系下的坐标是 $\mathbf{P}$。相机内参 $K$ 将这个3D点投影到相机1的像素平面上，得到像素坐标 $p_1$。符号 $\simeq$ 表示等式在齐次坐标意义下成立，即左右两侧相差一个非零尺度因子（这个尺度因子就是点的深度）。
*   **第二行**: $p_2 \simeq K(R \mathbf{P}+t)$。首先，将点 $P$ 从相机1坐标系变换到相机2坐标系，得到 $\mathbf{P}' = R\mathbf{P}+t$。然后，相机2的内参 $K$ 再将这个点投影到相机2的像素平面上，得到像素坐标 $p_2$。

$ \begin{aligned}
x_1 &= K^{-1} p_1 \\
x_2 &= K^{-1} p_2
\end{aligned} $

**原理说明**:
*   这是从**像素坐标**反向计算**归一化平面坐标**的过程。
*   归一化平面是一个虚拟的、焦距为1的成像平面。将一个3D点投影到这个平面上，相当于去除了相机内参（焦距、主点偏移）的影响，只保留了纯粹的几何方向信息。
*   从几何上看，$x_1$ 和 $x_2$ 是从各自相机光心出发，指向3D点 $P$ 的**方向向量**。$x_1 = [X/Z, Y/Z, 1]^T$（在相机1坐标系下），$x_2 = [X'/Z', Y'/Z', 1]^T$（在相机2坐标系下）。
*   通过左乘 $K^{-1}$，我们“撤销”了内参 $K$ 的作用，得到了这个方向向量。

$ x_2 \simeq R x_1 + t $

**原理说明**:
*   这个公式是**核心几何关系**在归一化平面上的体现。
*   我们从 $p_1 \simeq K\mathbf{P}$ 和 $p_2 \simeq K(R\mathbf{P}+t)$ 出发。令 $p_1 = s_1 K \mathbf{P}$ 和 $p_2 = s_2 K (R\mathbf{P}+t)$，其中 $s_1, s_2$ 是深度。
*   那么，$\mathbf{P} = \frac{1}{s_1} K^{-1}p_1 = \frac{1}{s_1}x_1$。
*   代入第二个式子：$p_2 \simeq K(R(\frac{1}{s_1}x_1) + t)$。
*   两边左乘 $K^{-1}$：$K^{-1}p_2 \simeq \frac{1}{s_1}Rx_1 + t$。
*   即 $x_2 \simeq \frac{1}{s_1}Rx_1 + t$。由于齐次坐标的尺度不变性，我们可以写成 $s_2 x_2 = s_1(R x_1) + t$（这里的$s_1,s_2$是新的尺度）。但这仍然意味着 $x_2, Rx_1, t$ 三个向量是**共面**的。这个关系可以近似写为 $x_2 \simeq Rx_1 + t$。

---

### 2. 第二组公式：推导对极约束

我们的目标是消除未知的3D点 $\mathbf{P}$（或其深度 $s_1, s_2$），只建立 $x_1, x_2, R, t$ 之间的关系。

$ \boldsymbol{t}^{\wedge} \boldsymbol{x}_2 \simeq \boldsymbol{t}^{\wedge} \boldsymbol{R} \boldsymbol{x}_1 $

**原理说明**:
*   **几何意义**: 我们知道 $x_2, Rx_1, t$ 三个向量是共面的。这三个向量的几何关系是：相机光心 $O_1$、光心 $O_2$ 和三维点 $P$ 构成了**对极平面 (Epipolar Plane)**。向量 $x_1$（在相机1坐标系）、$x_2$（在相机2坐标系）、$t$（连接 $O_1, O_2$）都位于这个平面内。$Rx_1$ 是将 $x_1$ 旋转到相机2坐标系下，它也必然在这个平面内。
*   **代数操作**: 为了消除平移向量 $t$，一个非常巧妙的方法是两边同时**左乘 $t$ 的反对称矩阵 $t^{\wedge}$**。
*   回忆叉乘的性质：$a \times b = a^\wedge b$。
*   我们从 $x_2 = R x_1 + t$ （忽略尺度因子）出发，两边左叉乘 $t$：
    $ t \times x_2 = t \times (R x_1 + t) $
    $ t \times x_2 = t \times (R x_1) + t \times t $
*   因为任何向量与自身的叉乘为零 ($t \times t = \mathbf{0}$)，所以：
    $ t \times x_2 = t \times R x_1 $
*   将其写成反对称矩阵形式，就得到了 $t^{\wedge} x_2 = t^{\wedge} R x_1$。

---

### 3. 第三组公式：得到最终的对极约束方程

$ \boldsymbol{x}_2^{\mathrm{T}} \boldsymbol{t}^{\wedge} \boldsymbol{x}_2 \simeq \boldsymbol{x}_2^{\mathrm{T}} \boldsymbol{t}^{\wedge} \boldsymbol{R} \boldsymbol{x}_1 $

**原理说明**:
*   这是一个纯粹的代数步骤，目的是进一步简化关系，得到一个标量方程。
*   我们将上一步得到的向量等式 $t^{\wedge} x_2 = t^{\wedge} R x_1$ 两边同时**左点乘向量 $x_2^T$**。

$ \boldsymbol{t}^{\wedge} \boldsymbol{x}_2 \text { 是一个与 } \boldsymbol{t} \text { 和 } \boldsymbol{x}_2 \text { 都垂直的向量 } $

**原理说明**:
*   $t^{\wedge} x_2$ 就是叉积 $t \times x_2$。
*   根据叉积的定义，其结果向量必然同时垂直于参与运算的两个向量，即 $t \times x_2$ 垂直于 $t$ 并且垂直于 $x_2$。
*   因此，向量 $x_2$ 与向量 $t \times x_2$ 的点积必然为零。
    $ x_2^T (t^{\wedge} x_2) = x_2 \cdot (t \times x_2) = 0 $
*   这是一种**混合积 (Scalar Triple Product)** 的形式 $a \cdot (b \times c)$。当其中两个向量相同时（如 $x_2 \cdot (t \times x_2)$），这个混合积代表的平行六面体体积为零，所以结果为零。

$ \boldsymbol{x}_2^{\mathrm{T}} \boldsymbol{t}^{\wedge} \boldsymbol{R} \boldsymbol{x}_1 = 0 $

**原理说明**:
*   这是整个推导的**最终结果**，被称为**对极约束 (Epipolar Constraint)**。
*   我们将上一步的结论 $x_2^T (t^{\wedge} x_2) = 0$ 代入到方程 $x_2^{\mathrm{T}} t^{\wedge} x_2 = x_2^{\mathrm{T}} t^{\wedge} R x_1$ 的左边。
*   于是，方程的右边也必须为零，即 $x_2^{\mathrm{T}} t^{\wedge} R x_1 = 0$。
*   这个方程只包含了**归一化平面坐标** $x_1, x_2$ 和相机间的**相对位姿** $R, t$，完全消除了对3D点 $\mathbf{P}$ 及其深度的依赖。

**几何解释**:
$x_2^T (t^\wedge R x_1) = 0$ 实际上是混合积 $x_2 \cdot (t \times Rx_1) = 0$。
*   $Rx_1$ 是点 $P$ 在相机1坐标系下的方向向量，旋转到相机2坐标系下。
*   $t$ 是从相机1光心指向相机2光心的向量。
*   $x_2$ 是点 $P$ 在相机2坐标系下的方向向量。
*   混合积为零的几何意义是：**这三个向量 $x_2, t, Rx_1$ 是共面的**。这完美地印证了我们最初的几何设定：$O_1, O_2, P$ 三点定义了一个对极平面。

---

### 4. 从本质矩阵到基础矩阵

$ \boldsymbol{p}_2^{\mathrm{T}} \boldsymbol{K}^{-\mathrm{T}} \boldsymbol{t}^{\wedge} \boldsymbol{R} \boldsymbol{K}^{-1} \boldsymbol{p}_1 = 0 $

**原理说明**:
*   在实际应用中，我们通常只有像素坐标 $p_1, p_2$ 和相机内参 $K$。
*   我们将 $x_1 = K^{-1}p_1$ 和 $x_2 = K^{-1}p_2$ 代入对极约束方程 $x_2^T t^\wedge R x_1 = 0$：
    $ (K^{-1}p_2)^T (t^\wedge R) (K^{-1}p_1) = 0 $
*   利用转置的性质 $(AB)^T = B^T A^T$：
    $ p_2^T (K^{-1})^T t^\wedge R K^{-1} p_1 = 0 $
    $ (K^{-1})^T$ 通常简写为 $K^{-T}$。
*   这就得到了最终可以用像素坐标直接计算的约束方程。

**定义两个重要的矩阵**:
*   **本质矩阵 (Essential Matrix)** $E = t^\wedge R$。它包含了相机间的旋转和平移信息（外参）。约束写为 $x_2^T E x_1 = 0$。
*   **基础矩阵 (Fundamental Matrix)** $F = K^{-T} t^\wedge R K^{-1} = K^{-T} E K^{-1}$。它不仅包含了外参，还包含了相机内参。约束写为 $p_2^T F p_1 = 0$。

在计算机视觉中，我们通过匹配多对点 $(p_1, p_2)$，利用 $p_2^T F p_1 = 0$ 这个约束来求解基础矩阵 $F$，进而分解出相机的内外参数和相对位姿。


### p3p方法求解pnp是否可以用在rgb-d相机场景下，3个三维点坐标和3的对应的像素点然后得出相机的位姿？请对PnP算法做更多的补充说明。

Perspective-n-point(透视点法)，p3p 主要用在连续帧的位姿估计中，3维坐标点为世界坐标系下的坐标，二维像素点为当前帧图像上的像素点，基于P3P去估计位姿，RANSAC（随机样本一致性）框架内使用P3P


### 为什么3d-3d位姿估计，比如连续点云帧的数据，可以基于svd的分解来计算位姿变化？3d为各自帧的的lidar局部坐标系下的坐标就可以？

 3D-3D位姿估计 或 点云配准 (Point Cloud Registration) 问题，而基于SVD的解法是其最经典和基础的方法。