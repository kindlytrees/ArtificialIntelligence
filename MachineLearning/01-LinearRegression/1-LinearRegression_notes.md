# 线性回归

## 参数更新公式推导

- 迭代法求解过程

$$
\begin{aligned}
J(\theta) & =\frac{1}{2} \sum_{i=1}^n\left(y^{(i)}-\theta^{\top} x^{(i)}\right)^2 \\
\frac{\partial}{\partial \theta_j} J(\theta) & =\frac{1}{2} \cdot 2 \cdot \sum_{i=1}^n\left(y^{(i)}-\theta^{\top} x^{(i)}\right) \frac{\partial\left(y^{(i)}-\theta^{\top} x^{(i)}\right)}{\partial \theta_j} \\
& =\sum_{i=1}^n\left(y^{(i)}-\theta^{\top} x^{(i)}\right) \cdot\left(-x_j^{(i)}\right) \\
& =-\sum_{i=1}^n\left(y^{(i)}-\theta^{\top} x^{(i)}\right) x_j^{(i)} \\
\hat{\theta}_j & =\theta_j-\alpha \frac{\partial}{\partial \theta_j} J(\theta)=\theta_j+\alpha \sum_{i=1}^n\left(y^{(i)}-\theta^{\top} x^{(i)}\right) x_j^{(i)}
\end{aligned}
$$

- 解析解求解过程

$$
\begin{gathered}
X=\left[\begin{array}{l}
-\chi^{(1) \top}- \\
-\chi^{(2) \top}- \\
-\chi^{(n)^{\top}}-
\end{array}\right] \quad J(\theta)=\frac{1}{2}(\vec{y}-X \theta)^{\top}(\vec{y}-X \theta) \\
\nabla_\theta J(\theta)=\frac{1}{2} \nabla_\theta\left(\vec{y}^{\top}-\theta^{\top} X^{\top}\right)(\vec{y}-X \theta) \\
=\frac{1}{2} \nabla_\theta\left(\vec{y}^{\top} \cdot \vec{y}-\vec{y}^{\top} X \theta-\theta^{\top} X^{\top} \vec{y}+\theta^{\top}\left(X^{\top} X\right) \theta\right) \\
=\frac{1}{2} \nabla_\theta\left(-2 X^{\top} \vec{y}+\theta^{\top}\left(X^{\top} X\right) \theta\right) \\
=-X^{\top} \vec{y}+X^{\top} X \theta=0 \\
X^{\top} X \theta=\vec{y}^{\top} X \\
\theta=\left(X^{\top} X\right)^{-1} X^{\top} \vec{y}
\end{gathered}
$$

- 基于高斯概率分布的log似然函数

$$
\begin{aligned}
& l(\theta)=\log L(\theta) \\
= & \log \prod_{i=1}^n \frac{1}{2\pi^{1 / 2} \sigma} e^{-\frac{\left(y(i)-\theta^{\top} x^{(i)}\right)^2}{2 \sigma^2}} \\
= & n \log \frac{1}{2\pi^{1 / 2} \sigma}+\log e^{-\frac{\left(y^{(i)}-\theta^{\top} x^{(i)}\right)^2}{2 \sigma^2}} \\
= & n \log \frac{1}{\sqrt{2 \pi} \sigma}-\frac{1}{\sigma^2} \cdot \frac{1}{2} \sum_{i=1}^n\left(y^{(i)}-\theta^{\top} x^{(i)}\right)^2
\end{aligned}
$$
