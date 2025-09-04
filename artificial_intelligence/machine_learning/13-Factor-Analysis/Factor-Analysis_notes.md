# Factor Analysis Note

- 因素分析中，x和z通过关系式 $x = \mu +  \Lambda z + \epsilon$ 进行表示，z为均值为0，方差为 $\Psi$的隐向量分布
- 因素分析中隐向量z通过lambda矩阵变换到观察空间，观察空间是一个维度更高的空间对吗? 把z也看成短向量的属性？构建条件概率得出z和x的变换关系，在实际使用时z未知，通过反变换求出z吗？
- 因素分析是否可以看成是vae的mini或解析解法版本?其解释性更强，而vae的由于采用了神经网络深度学习的模型，其能量更大?

gemini辅助回答摘要：

因子分析（Factor Analysis）完全可以被看作是VAE(线性高斯世界的VAE)的一个线性的、有解析解的“迷你版本”

观察空间是一个维度更高的空间。 因素分析的核心目标就是降维（dimensionality reduction）和结构发现（structure discovery）
隐向量 (z): 是模型假设的、导致观测变量之间相关性的潜在因素。它的维度通常远低于观测变量的维度。
因素分析模型假设观测变量x是由隐向量Z线性变换（通过因子载荷矩阵Λ）并加上一个误差项ϵ产生的。其基本形式为：

我们要做的是在已知X和已估计的 Λ,μ,Ψ的情况下，推断最可能的Z值。这个过程被称为**因子得分（Factor Scores）**的估计。

$$
\begin{aligned}
z & \sim \mathcal{N}(0, I) \\
x \mid z & \sim \mathcal{N}(\mu+\Lambda z, \Psi)
\end{aligned}
$$

$$
\begin{aligned}
z & \sim \mathcal{N}(0, I) \\
\epsilon & \sim \mathcal{N}(0, \Psi) \\
x & =\mu+\Lambda z+\epsilon
\end{aligned}
$$


- 多元高斯分布及条件分布边缘分布的概率分布特性

$$
\begin{aligned}
& p\left(x_2\right)=\frac{1}{\sqrt{(2 \pi)^{n_2}\left|\Sigma_{22}\right|}} \exp \left(-\left(x_2-u_2\right)^{\top} \Sigma_{22}^{-1}\left(x_2-u_2\right)\right) \\
& E\left(\left[\begin{array}{l}
x_1 \\
x_2
\end{array}\right]\right)=\binom{u_1}{u_2} \quad \sum\binom{x_1}{x_2}=E\left(\left[\begin{array}{l}
x_1 \\
x_2
\end{array}\right]-E\left[\begin{array}{l}
x_1 \\
x_2
\end{array}\right]\right)\left(\left[\begin{array}{l}
x_1 \\
x_2
\end{array}\right]-E\left[\begin{array}{l}
x_1 \\
x_2
\end{array}\right]^{\top}\right) \\
& \left.=E\left(\left[\begin{array}{l}
x_1 \\
x_2
\end{array}\right]-\left[\begin{array}{l}
u_1 \\
u_2
\end{array}\right]\right)\left(\left[\begin{array}{l}
x_1 \\
x_2
\end{array}\right]-\left[\begin{array}{l}
u_1 \\
u_2
\end{array}\right]\right)^{\top}\right) \\
& =E\left(\binom{x_1-u_1}{x_2-u_2}\binom{x_1-u_1}{x_2-u_2}^{\top}\right) \\
& =E\left[\binom{x_1-u_1}{x_2-u_2}\left(\left(x_1-u_1\right)^{\top}\left(x_2-u_2\right)^{\top}\right)\right] \\
& =\left[\begin{array}{ll}
E\left(x_1-u_1\right)\left(x_1-u_1\right)^{\top} & E\left(x_1-u_1\right)\left(x_2-u_2\right)^{\top} \\
E\left(x_2-u_2\right)\left(x_1-u_1\right)^{\top} & E\left(x_2-u_2\right)\left(x_2-u_2\right)^{\top}
\end{array}\right] \\
& =\left[\begin{array}{ll}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22}
\end{array}\right]
\end{aligned}
$$

$$
p(x)=\frac{1}{\sqrt{(2 \pi)^D|\Sigma|}} \exp \left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right)
$$

$$
\begin{aligned}
&\Lambda_{11}=\left(\Sigma_{11}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}\right)^{-1}\\
&\Lambda_{22}=\left(\Sigma_{22}-\Sigma_{21} \Sigma_{11}^{-1} \Sigma_{12}\right)^{-1}\\
&\Lambda_{12}=-\Lambda_{11} \Sigma_{12} \Sigma_{22}^{-1}\\
&\Lambda_{21}=-\Lambda_{22} \Sigma_{21} \Sigma_{11}^{-1}=\Lambda_{12}^T
\end{aligned}
$$

$$
\begin{aligned}
&p\left(x_2\right)=\mathcal{N}\left(\mu_2, \Sigma_{22}\right)=\frac{1}{\sqrt{(2 \pi)^{D_2}\left|\Sigma_{22}\right|}} \exp \left(-\frac{1}{2}\left(x_2-\mu_2\right)^T \Sigma_{22}^{-1}\left(x_2-\mu_2\right)\right)\\
&p\left(x_1 \mid x_2\right) \propto \exp \left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right) \exp \left(+\frac{1}{2}\left(x_2-\mu_2\right)^T \Sigma_{22}^{-1}\left(x_2-\mu_2\right)\right)
\end{aligned}
$$

$$
\begin{aligned}
&\Lambda_{11}^{-1}=\Sigma_{11}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}\\
&\Lambda_{12}=-\Lambda_{11} \Sigma_{12} \Sigma_{22}^{-1}\\
&\mu_{1 \mid 2}=\mu_1-\Lambda_{11}^{-1}\left(-\Lambda_{11} \Sigma_{12} \Sigma_{22}^{-1}\right)\left(x_2-\mu_2\right)\\
&\mu_{1 \mid 2}=\mu_1+\Sigma_{12} \Sigma_{22}^{-1}\left(x_2-\mu_2\right)\\
&\Sigma_{1 \mid 2}=\Lambda_{11}^{-1}=\Sigma_{11}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}\\
&\Sigma_{1 \mid 2}=\Sigma_{11}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{12}^T
\end{aligned}
$$

$$
\begin{aligned}
& {\left[\begin{array}{l}
z \\
x
\end{array}\right] \sim \mathcal{N}\left(\left[\begin{array}{l}
\overrightarrow{0} \\
\mu
\end{array}\right],\left[\begin{array}{cc}
I & \Lambda^T \\
\Lambda & \Lambda \Lambda^T+\Psi
\end{array}\right]\right)} \\
& \mathcal{N}\left(\mu, \Lambda \Lambda^T+\Psi\right)
\end{aligned}
$$

- 回顾EM算法

$$
\begin{aligned}
& \log \prod_{i=1}^{n}\left(\sum_z p(x, z ; \theta)\right) \\
&= \sum_{i=1}^n \log \left(\sum_z p(x, z; \theta)\right) \\
&= \sum_{i=1}^n \log \sum_z \frac{Q\left(x^{(i)} z\right)}{Q\left(x^{(i)}, z\right)} p(x, z ; \theta) \\
&= \sum_{i=1}^n \log \sum_z Q\left(x^{(i)}, z\right) \frac{p(x, z ; \theta)}{Q\left(x^{(i)}, z\right)} \\
& \geq \sum_{i=1}^n \sum_z Q\left(x^{(i)}, z\right) \log \frac{p\left(x^{(i)}, z ; \theta\right)}{Q\left(x^{(i)}, z\right)} \\
& Q\left(x^{(i)}, z\right) \propto p\left(x^{(i)}, z ; \theta\right) \\
& \sum_z Q\left(x^{(i)}, z\right)=1 \\
& \frac{c*p\left(x^{(i)}, z ; \theta\right)}{c*\sum_z p\left(x^{(i)}, z ; \theta\right)}=\frac{p\left(x^{(i)}, z ; \theta\right)}{p\left(x^{(i)} ; \theta\right)}=p\left(z \mid x^{(i)} ; \theta\right)
\end{aligned}
$$

公示的推导说明

$$
\begin{aligned}
& \left.\nabla_{\Lambda} \sum_{i=1}^n-E\left[\frac{1}{2}\left(x^{(i)}-u-\Lambda z^{(i)}\right)\right]^{\top} \Psi^{-1}\left(x^{(i)}-u-\Lambda z^{(i)}\right)\right] \\
& =\sum_{i=1}^n \nabla_{\Lambda} E\left[\operatorname{tr}\left[\frac{1}{2}\left(\left(x^{(i)}-u\right)-\Lambda z^{(i)}\right)^{\top} \Psi^{-1}\left(\left(x^{(i)}-u\right)-\Lambda z^{(i)}\right)\right]\right. \\
& =\sum_{i=1}^n \nabla_{\Lambda} E\left[-\frac{1}{2} \operatorname{tr}\left[-\left[\left(x^{(i)}-u\right)^{\top} \Psi^{-1} \Lambda z^{(i)}-\left(\Lambda z^{(i)}\right)^{\top} \Psi^{-1}\left(x^{(i)}-u\right)+\left(\Lambda z^{(i)}\right)^{\top} \Psi^{-1}\left(\Lambda z^{(i)}\right)\right.\right.\right. \\
& \left.=\sum_{i=1}^n \nabla_{\Lambda} E\left[-\frac{1}{2} \operatorname{tr}\left[z^{(i)^T} \Lambda^{\top} \psi^{-1} \Lambda z^{(i)}\right)-2 \operatorname{tr} z^{(i)^T} \Lambda^{\top} \Psi^{-1}\left(x^{(i)}-u\right)\right]\right] \\
& =\sum_{i=1}^n \nabla_{\Lambda} E\left[-\operatorname{tr} \frac{1}{2} \Lambda^{\top} \Psi^{-1} \Lambda z^{(i)} z^{(i) \top}+\operatorname{tr} \Lambda^{\top} \Psi^{-1}\left(x^{(i)}-u\right) z^{(i) \top}\right] \\
& \nabla_ A \operatorname{tr} A B A^{\top} C=C A B+C^{\top} A B^{\top} \operatorname{tr} A B=\operatorname{tr} B A \quad \operatorname{tr} A B C=\operatorname{tr} B C A \\
& \operatorname{tr} \frac{\Lambda^{\top} \Psi^{-1}}{A} \frac{\Lambda}{B} \frac{z^{(i)} z^{(i)}{ }^T}{C}=\operatorname{tr} \frac{\Lambda}{B} \frac{z^{(i)} z^{(i)}{ }^T}{C} \frac{\Lambda^{\top} \Psi^{-1}}{A}=\operatorname{tr} \frac{\Lambda}{A} \frac{z^{(i)}\left(z^{(i)}\right)^T}{B} \frac{\Lambda^T}{A^T} \frac{\Psi^{-1}}{C} \\
& =\frac{\Psi^{-1}}{C} \frac{\Lambda}{A} \frac{z^{(i)} z^{(i)}}{B}+\frac{\Psi^{-1} \Lambda\left(z^{(i)}\left(z^{(i)}\right)^{\top}\right)^{\top}}{C^{\top} A B^{\top}} \\
& =2 \Psi^{-1} \Lambda z^{(i)} z^{(i) T}
\end{aligned}
$$