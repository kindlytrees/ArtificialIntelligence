# 逻辑回归

Notice that g(z) tends towards 1 as z → ∞, and g(z) tends towards 0 as
z → −∞. Moreover, g(z), and hence also h(x), is always bounded between
0 and 1. As before, we are keeping the convention of letting x0 = 1, so that

$\theta^Tx=\theta _0 +\sum_{j=1}^d θ_jx_j$

For now, let’s take the choice of g as given. Other functions that smoothly
increase from 0 to 1 can also be used, but for a couple of reasons that we’ll see
later (when we talk about GLMs, and when we talk about generative learning
algorithms), the choice of the logistic function is a fairly natural one. Before
moving on, here’s a useful property of the derivative of the sigmoid function,
which we write as g′:

$g'(z)=\frac{d}{dz}(\frac{1}{1+e^{-z}})= \frac{1}{(1+e^{-z})^2}e^{-z}= \frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}})= g(z)(1-g(z)$

Assuming that the n training examples were generated independently, we
can then write down the likelihood of the parameters as

$p(y=1|x;\theta)=h_\theta(x)$

$p(y=0|x;\theta)=1-h_\theta(x)$

$p(y|x;\theta)=h_\theta(x)^y(1-h_\theta(x))^{(1-y)}$

$p(\vec y|X;\theta )=\prod p(y|x;\theta)=\prod h_\theta(x)^y(1-h_\theta(x))^{(1-y)}$

As before, it will be easier to maximize the log likelihood:

$l(\theta)=logL(\theta)=\sum_{i=1}^{m}y^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))$

$\theta=\theta+\alpha\bigtriangledown _\theta l(\theta)$

$\frac{\partial }{\partial \theta_j}l(\theta)=\sum_{i=1}^{m}(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}$

$$
\begin{aligned}
& \frac{\partial }{\partial \theta_j}l(\theta)=-\sum_{i=1}^n (y_i \frac {1} {\sigma( \theta ^T x_i)} -(1-y_i) \frac {1}{1-\sigma(\theta ^Tx_i)})\frac {\partial} {\partial \theta_j}g(\theta ^T x_i) \\
= & -\sum_{i=1}^n (y \frac {1} {\sigma( \theta ^T x_i)} -(1-y) \frac {1}{1-\sigma(\theta ^Tx_i)})\sigma(\theta ^T x_i)(1-\sigma(\theta ^Tx_i))\frac {\partial}{\partial \theta_j} \theta ^Tx_i \\
= & -\sum_{i=1}^n (y_i(1-\sigma(\theta ^Tx_i))-(1-y_i)\sigma(\theta ^T x_i))x_i^{(j)} \\
= & -\sum_{i=1}^n (y_i-\sigma(\theta ^Tx_i))x_i^{(j)} \\
\end{aligned}
$$
