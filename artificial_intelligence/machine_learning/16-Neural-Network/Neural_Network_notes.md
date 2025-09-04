# Neural Network

## 激活函数的种类

- logistic funciton

$$
f(z)=\frac{1}{1+\exp (-z)}
$$

- 双曲正切函数

$$
f(z)=\tanh (z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
$$

- ReLU

$$
f(z)=\max (0, x)
$$


## 神经网络的函数表示

[!Nenual Network](./nn.png)

$$
\begin{gathered}
a_1^{(2)}=f\left(W_{11}^{(1)} x_1+W_{12}^{(1)} x_2+W_{13}^{(1)} x_3+b_1^{(1)}\right) \\
a_2^{(2)}=f\left(W_{21}^{(1)} x_1+W_{22}^{(1)} x_2+W_{23}^{(1)} x_3+b_2^{(1)}\right) \\
a_3^{(2)}=f\left(W_{31}^{(1)} x_1+W_{32}^{(1)} x_2+W_{33}^{(1)} x_3+b_3^{(1)}\right) \\
h_{W, b}(x)=a_1^{(3)}=f\left(W_{11}^{(2)} a_1^{(2)}+W_{12}^{(2)} a_2^{(2)}+W_{13}^{(2)} a_3^{(2)}+b_1^{(2)}\right) \\
z_i^{(2)}=\sum_{j=1}^n W_{i j}^{(1)} x_j+b_i^{(1)} \quad a_i^{(l)}=f\left(z_i^{(l)}\right) \\
z^{(2)}=W^{(1)} x+b^{(1)} \\
a^{(2)}=f\left(z^{(2)}\right) \\
z^{(3)}=W^{(2)} a^{(2)}+b^{(2)} \\
h_{W, b}(x)=a^{(3)}=f\left(z^{(3)}\right)
\end{gathered}
$$

- 单个样本的损失函数

$$
J(W, b ; x, y)=\frac{1}{2}\left\|h_{W, b}(x)-y\right\|^2
$$

- 所有样本损失函数定义及权重衰减正则化

$$
\begin{aligned}
J(W, b) & =\left[\frac{1}{m} \sum_{i=1}^m J\left(W, b ; x^{(i)}, y^{(i)}\right)\right]+\frac{\lambda}{2} \sum_{l=1}^{n_l-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}}\left(W_{j i}^{(l)}\right)^2 \\
& =\left[\frac{1}{m} \sum_{i=1}^m\left(\frac{1}{2}\left\|h_{W, b}\left(x^{(i)}\right)-y^{(i)}\right\|^2\right)\right]+\frac{\lambda}{2} \sum_{l=1}^{n_l-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}}\left(W_{j i}^{(l)}\right)^2
\end{aligned}
$$

$$
\begin{aligned}
\delta_i^{(l)}=\left(\sum_{j=1}^{s_{l+1}} W_{j i}^{(l)} \delta_j^{(l+1)}\right) f^{\prime}\left(z_i^{(l)}\right) \\
\delta^{(l)}=\left(\left(W^{(l)}\right)^T \delta^{(l+1)}\right) \bullet f^{\prime}\left(z^{(l)}\right) \\
\frac{\partial}{\partial W_{i j}^{(l)}} J(W, b ; x, y) & =a_j^{(l)} \delta_i^{(l+1)} \\
\frac{\partial}{\partial b_i^{(l)}} J(W, b ; x, y) & =\delta_i^{(l+1)}
\end{aligned}
$$

$$
\begin{aligned}
\nabla_{W^{(l)}} J(W, b ; x, y) & =\delta^{(l+1)}\left(a^{(l)}\right)^T \\
\nabla_{b^{(l)}} J(W, b ; x, y) & =\delta^{(l+1)} .
\end{aligned}
$$

- 反向传播链式法则的公式说明

$$
\begin{gathered}
\delta_i^{(2)}=\frac{\partial J}{\partial z_i^{(2)}}=\frac{\partial J}{\partial a_i^{(2)}} \cdot \frac{\partial a_i^{(2)}}{\partial z_i^{(2)}}=\left(\frac{\partial J}{\partial z_1^{(3)}} \cdot \frac{\partial z_1^{(3)}}{\partial a_i^{(2)}}+\frac{\partial J}{\partial z_2^{(3)}}+\frac{\partial z_2^{(3)}}{\partial a_i^{(2)}}+\cdots+\frac{\partial J}{\partial z_k^{(3)}} \cdot \frac{\partial z_k^{(3)}}{\partial a_i^{(2)}}\right) \frac{\partial a_i^{(2)}}{\partial z_i^{(2)}} \\
\delta_i^{(2)}=\sum_k\left[\delta_k^{(3)} \cdot w_{k i}^{(2)}\right] \cdot f^{\prime}\left(\delta_i^{(2)}\right)=\left[w_{\cdots i}^{(2)} \cdot \delta^{(3)}\right] f^{\prime}\left(\delta_i^{(2)}\right) \\
\delta^{(2)}=\left[w^{(2) T} \cdot \delta^{(3)}\right] \cdot f^{\prime}\left(\delta^{(2)}\right) \quad \delta^{(1)}=\left[w^{(l) T} \cdot f^{(l+1)}\right] \cdot f^{\prime}\left(\delta^{(l)}\right)
\end{gathered}
$$

## 实验说明

实验代码中的dz4值得推导过程

$$
\begin{aligned}
& p(x, \hat{y})={\hat{y}}^{y}(1-\hat y)^{(1-y)} \\
& \log p(x, \hat{y})  =y \log \hat{y}+(1-y) \log (1-\hat{y}) \\
& J = (y \log \hat{y}+(1-y) \log (1-\hat{y})) \\
& = \frac{\partial J}{\partial z_{4}} =\frac{\partial J}{\partial a_4} \cdot \frac{\partial a_4}{\partial z_4} \\
& =-\left[y \frac{1}{\hat y}+(1-y) \frac{1}{(1-\hat{y})} \times(-1)\right] (\hat{y})(1-\hat{y}) \\
& =\left(-y \frac{1}{\hat{y}}+(1-y) \frac{1}{(1-\hat{y})}\right)(\hat{y})(1-\hat{y}) \\
& =-y(1-\hat{y})+(1-y) \hat{y} \\
& =-y+y \hat{y}+\hat{y}-y \hat{y} \\
& =\hat{y}-y
\end{aligned}
$$

## 参考资源

- [神经网络可视化](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
- [函数拟合的视角理解神经网络](http://staff.ustc.edu.cn/~lgliu/Resources/DL/What_is_DeepLearning.html)