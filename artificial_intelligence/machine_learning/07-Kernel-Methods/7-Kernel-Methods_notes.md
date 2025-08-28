# 核方法

- LMS with kernel trick

$$
\begin{aligned}
\theta= & \theta+\alpha \sum_{i=1}^n\left(y^{(i)}-\theta^{\top} x^{(i)}\right) x^{(i)} \\
\Rightarrow \quad \theta= & \theta+\alpha \sum_{i=1}^n \left(y^{(i)}-\theta^{\top} \phi\left(x^{(i)}\right)\right) \phi\left(x^{(i)}\right). \\
& \theta=\sum_{i=1}^n \beta_i \phi\left(x^{(i)}\right) \\
\Rightarrow \quad \theta= & \sum_{i=1}^n \beta_i \phi\left(x^{(i)}\right)+\alpha \sum_{i=1}^n\left(y^{(i)}-\theta^{\top} \phi\left(x^{(i)}\right)\right) \phi\left(x^{(i)}\right) \\
= & \sum_{i=1}^n\left(\beta_i+\alpha\left(y^{(i)}-\theta^{\top} \phi\left(x^{(i)}\right)\right) \phi\left(x^{(i)}\right)\right. \\
& \\
& \quad \beta_i=\beta_i+\alpha\left(y^{(i)}-\sum_{j=1}^n \beta_j \phi\left(x^{(j)}\right)^{\top} \phi\left(x^{(i)}\right)\right) \\
& \left.\phi\left(x^{(j)}\right)\right)^{\top} \phi\left(x^{(i)}\right)=\left\langle\phi\left(x^{(j)}\right), \phi\left(x^{(i)}\right)\right\rangle
\end{aligned}
$$

- 线性函数的参数可以看成为训练样本的线性组合的解释

$$
\begin{aligned}
\theta & =\left(X^{\top} X\right)^{-1} X^{\top} y \\
& =X^{\top}\left(X X^{\top}\right)^{-1} y \\
& =X^{\top} \alpha \\
\left(X^{\top} X\right)^{-1} X^{\top} \cdot X X^{\top} & =\left(X^{\top} X\right)^{-1} \cdot\left(X^{\top} \cdot X\right) \cdot X^{\top}=X^{\top} \\
X^{\top}\left(X X^{\top}\right)^{-1} \cdot X X^{\top} & =X^{\top}\left(X X^{\top}\right)^{-1} \cdot\left(X X^{\top}\right)=X^{\top}
\end{aligned}
$$

## 基于核方法的解析解求解

$$
\begin{aligned}
& \theta  =\sum_{i=1}^n \alpha_i \phi\left(x^{(i)}\right) \\
\hat{y}(j) & =\theta^{\top} \phi \left(x^{(j)}\right) \\
& =\sum_{i=1}^n \alpha_i \phi\left(x^{(i)}\right)^{\top} \cdot \phi\left(x^{(j)}\right) \\
& =\sum_{i=1}^n \alpha_i K\left(x^{(i)}, x^{(j)}\right) \\
& \vec{\hat y} ={\left[\begin{array}{l}
\sum_{i=1}^n \alpha_i K\left(x^{(i)}, x^{(i)}\right) \\
...... \\
\sum_{i=1}^n \alpha_i K\left(x^{(i)}, x^{(2)}\right) \\
\sum_{i=1}^n \alpha_i K\left(x^{(i)}, x^{(n)}\right)
\end{array}\right] }  \\
& J(\alpha)=\frac{1}{2}(\vec{\hat y}-\vec{y})^{\top}(\vec{\hat y}-\vec{y})  \\
& =\frac{1}{2}\left(K^{\top} \alpha^{\top}-\vec{y}^{\top}\right)^T \left(K^{\top} \alpha^{\top}-\vec{y}\right)
\end{aligned}
$$

## 常见的几个核函数及其展开分析

- Can be inner product in infinite dimensional space Assume $x \in R^1$ and $\gamma>0$.
且自然常数的指数的级数表示为：

$$
e^x=\sum_{n=0}^{\infty} \frac{x^n}{n!}
$$

- 一元变量高斯核函数映射

$$
\begin{aligned}
& \quad e^{-\gamma \left(x-z\right)^2}=e^{-\gamma\left(x-z\right)^2}=e^{-\gamma x^2+2 \gamma x z-\gamma z^2} \\
& =e^{-\gamma x^2-\gamma z^2}\left(1+\frac{2 \gamma x_i x_j}{1!}+\frac{\left(2 \gamma x_i x_j\right)^2}{2!}+\frac{\left(2 \gamma x_i x_j\right)^3}{3!}+\cdots\right) \\
& =e^{-\gamma x^2-\gamma z^2}\left(1 \cdot 1+\sqrt{\frac{2 \gamma}{1!}} x \cdot \sqrt{\frac{2 \gamma}{1!}} z+\sqrt{\frac{(2 \gamma)^2}{2!}} x^2 \cdot \sqrt{\frac{(2 \gamma)^2}{2!}} z^2+\sqrt{\frac{(2 \gamma)^3}{3!}} x^3 \cdot \sqrt{\frac{(2 \gamma)^3}{3!} z^3}+\cdots\right) \\
& =\phi\left(x\right)^T \phi\left(z\right) \\
\end{aligned}
$$ 

因此映射函数可以表示为

$$
\phi(x)=e^{-\gamma x^2}\left[1, \sqrt{\frac{2 \gamma}{1!}} x, \sqrt{\frac{(2 \gamma)^2}{2!}} x^2, \sqrt{\frac{(2 \gamma)^3}{3!}} x^3, \cdots\right]^{\top}
$$


- 当属性为多元变量的高斯核函数

$$
\begin{aligned}
& \phi(x, z)=\exp \left(-r \| x-z \|^2\right) \\
& =e^{-r\sum_{i}^{d}\left(x_i - z_i\right)^2}  \\
& \begin{array}{l}
=e^{-r \sum_{i}^{d}\left(x_i^2 + z_i^2 \right) }  \sum_{i}^{d} \left(1+\frac{2 r x_i z_i}{1!}+\frac{\left(2 r x_i z_i\right)^2}{2!}+\cdots\right) \\
\end{array} \\
& =e^{-r \sum_{i}^{d}\left(x_i^2 + z_i^2 \right) } \sum_{i}^{d}\left(1+\sqrt \frac{{2 r}}{1!} x_i \cdot \sqrt\frac{{2 r}}{1!} z_i+ \sqrt\frac{{(2 r)}^2 }{2 !} x_i^2 \cdot \sqrt\frac{{(2 r)}^2 }{2 !} z_i^2+\cdots\right) \\
& \phi\left(x_i\right)=e^{-r x_i^2}\left[1, \sqrt\frac{{2} r}{{1!}} x_i, \sqrt\frac{{(2 r)}^2}{{2!}} x_i^2, \sqrt\frac{(2 r)^3}{{3!}} x_i^3, \cdots\right] (i=1,2,...d) \\
& \phi\left(x\right) = \begin{bmatrix} \phi\left(x_1\right) & \phi\left(x_2\right) & \phi\left(x_3\right) & \cdots & \phi\left(x_d\right) \end{bmatrix}
\end{aligned}
$$

## 核函数矩阵为半正定矩阵的证明

$$
\begin{aligned}
z^T K z & =\sum_i \sum_j z_i K_{i j} z_j \\
& =\sum_i \sum_j z_i \phi\left(x^{(i)}\right)^T \phi\left(x^{(j)}\right) z_j \\
& =\sum_i \sum_j z_i \sum_k \phi_k\left(x^{(i)}\right) \phi_k\left(x^{(j)}\right) z_j \\
& =\sum_k \sum_i \sum_j z_i \phi_k\left(x^{(i)}\right) \phi_k\left(x^{(j)}\right) z_j \\
& =\sum_k\left(\sum_i z_i \phi_k\left(x^{(i)}\right)\right)^2 \\
& \geq 0 .
\end{aligned}
$$

二次型(Quadratic Form)是一个关于多个变量的齐次多项式，其中每个项的次数都是 2。
利用二次型和定义来证明一个特定形式的矩阵（Gram 矩阵/核矩阵）是半正定的。它通过代数变形将复杂的矩阵-向量乘积转换为一系列实数的平方和，而平方和总是非负的，从而直接满足了半正定的定义。