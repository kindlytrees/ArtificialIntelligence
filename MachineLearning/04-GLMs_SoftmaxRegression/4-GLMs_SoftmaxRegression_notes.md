# Softmax回归

##  几种典型概率分布的描述
- 伯努利分布（PMF）
  - 二项式分布
  - 几何分布
  - 负几何分布 
- 范畴分布（PMF）
  - 多项式分布
- 高斯分布(PDF)
- 线性回归的最小二乘问题和逻辑回归问题如何用GLM来表示

最后一个公式的推导思路如下：

$$
\begin{align}
J(\theta) = - \left[ \sum_{i=1}^{m} \sum_{k=1}^{K}  1\left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}\right]
\end{align}
$$

以上损失函数对参数向量k求导时可以写成如下的组合：

$$
\begin{aligned}
& \nabla_{\theta^{(k)}} J(\theta)=\nabla_{\theta^{(k)}}\left(-\sum_{i=1}^m 1\left\{y^{(i)}=k\right\} \log \frac{\exp \left(\theta^{(k) T} T^{(i)}\right)}{\sum_{j=1}^k \exp \left(\theta^{(j)} T^{(i)} x^{(i)}\right)}  +\left(-\sum_{i=1}^m 1\left(y^{(i)} \neq k\right) \log \frac{\exp \left(\theta^{\left(y^{(i)}\right) \top} \chi^{(i)}\right.}{\sum_{j=1}^k \exp \left(\theta^{(j)^{\top}} \chi^{(i)}\right)}\right)\right)
\end{aligned}
$$

当样本的真值为k时的对参数k向量梯度求导：

$$
\begin{aligned}
& \nabla_{\theta^{(k)}} \log \frac{\exp \left(\theta^{(k) T} x^{(i)}\right)}{\sum_{j=1}^k \exp \left(\theta^{(j) T} x^{(i)}\right)} \\
= & \nabla_{\theta^{(k)}}\left(\log \exp \left(\theta^{(k) T} x^{(i)}\right) -\log\left(\sum_{j=1}^k \exp \left(\theta^{(j) T} x^{(i)}\right)\right)\right. \\
= & \nabla_{\theta^{(k)}}\left(\theta^{(k) T} x^{(i)}\right)-\nabla_{\theta^{(k)}} \log \left(\sum_{j=1}^k \exp \left(\theta^{(j) T} x^{(i)}\right)\right) \\
= & x^i-\frac{\exp \left(\theta^{(k) T} x^{(i)})\right.}{\sum_{j=1}^k \exp \left(\theta^{(i) T} x^{(i)}\right)} x^{(i)}\\
= & x^i\left(1-P\left(y^{(i)}=k \mid x^{(i)} ; \theta\right)\right)
\end{aligned}
$$

当样本的真值为非k时对参数k向量梯度求导：

$$
\begin{aligned}
& \nabla_{\theta(k)} \log \frac{\exp \left(\theta^{\left(y^{(i)}\right)^{\top}} x^{(i)}\right)}{\sum_{j=1}^k \exp \left(\theta^{(j) T} x^{(i)}\right)}\left(y^{(i)} \neq k\right) \\
& =-\frac{\theta^{(k)^{\top} x(i)}}{\sum_{j=1}^k \exp \left(\theta^{(j)^{\top} x(i)}\right)} \cdot x_i \\
& =-x_i \cdot P\left(y^{(i)}=k \mid x^{(i)} ; \theta\right)
\end{aligned}
$$

可以直观的理解为当样本真值分类为k时增大为k类的概率的参数更新，不为k时减小为k类概率的参数更新

综合起来的写法：

$$
\begin{align}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right]  }
\end{align}
$$

Softmax的损失函数是真实标签分布 (one-hot) 与模型预测概率分布之间的交叉熵。最小化这个损失，就是让模型的预测尽可能地匹配真实情况。
Softmax回归的损失函数确实可以从两个等价但视角不同的角度来理解：最大似然估计 (Maximum Likelihood Estimation, MLE) 和 信息论中的交叉熵 (Cross-Entropy)