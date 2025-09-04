# Ensemble Learing 1: Random Forest and AdaBoost

Bagging（Bootstrap Aggregating）, Boosting为集成学习算法系列的两种主要主要框架思路方法分类

将AdaBoost中的样本权重看作是**“样本量的多少”或者“虚拟的样本重复次数”**，是一个非常有效的理解方式。

- AdaBoost的集成方法定义和损失函数

$$
\begin{aligned}
&f(x)=\sum_{m=1}^M \alpha_m G_m(x)\\
&L(y, f(x))=\exp (-y f(x))
\end{aligned}
$$

- 样本错误率定义
  
$$
\epsilon_t=\frac{\sum_{i=1}^n w_t(i) \cdot \mathbb{1}\left(h_t\left(x_i\right) \neq y_i\right)}{\sum_{i=1}^n w_t(i)}
$$

- 分类器权重，于错误率有关，错误率越小，模型性能表现更好，权重就越高
  
$$
\alpha_t=\frac{1}{2} \ln \left(\frac{1-\epsilon_t}{\epsilon_t}\right)
$$

- 更新样本的权重，然后进行归一化处理

$$
w_{t+1}(i)=w_t(i) \cdot \exp \left(-\alpha_t y_i h_t\left(x_i\right)\right) \quad w_{t+1}(i)=\frac{w_{t+1}(i)}{\sum_{j=1}^n w_{t+1}(j)}
$$

- 最终分类器的表示

$$
H(x)=\operatorname{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)
$$


- 第m个弱分类器的目标函数定义
  
$$
\left(\alpha_m, G_m(x)\right)=\arg \min _{\alpha, G} \sum_{i=1}^N \exp \left(-y_i\left(f_{m-1}\left(x_i\right)+\alpha G\left(x_i\right)\right)\right)
$$

- 损失函数的求导过程

$$
\begin{aligned}
& \left.\sum_{i=1}^N \exp \left(-y_i\left(f_{m-1}\left(x_i\right)+\alpha G\left(x_i\right)\right)\right)=\sum_{i=1}^N \varpi_{m i} \exp \left(-y_i \alpha G\left(x_i\right)\right)\right) \\
& =\sum_{y_i=G m\left(x_i\right)} \varpi_{m i} e^{-\alpha}+\sum_{y_i \neq G m\left(x_i\right)} \varpi_{m i} e^\alpha=e^{-\alpha} \sum_{y_i=G m\left(x_i\right)} \varpi_{m i}+e^\alpha \sum_{y_i \neq G m\left(x_i\right)} \varpi_{m i} \\
& =e^{-\alpha} \sum_{i=1}^N \varpi_{m i} I\left(y_i=G_m\left(x_i\right)\right)+e^\alpha \sum_{i=1}^N \varpi_{m i} I\left(y_i \neq G m\left(x_i\right)\right) \\
& =e^{-\alpha} \sum_{i=1}^N \varpi_{m i}+\left(e^\alpha-e^{-\alpha}\right) \sum_{i=1}^N \varpi_{m i} I\left(y_i \neq G_m\left(x_i\right)\right)
\end{aligned}
$$

其中 $\varpi_{m i}=\exp \left(-y_i f_{m-1}\left(x_i\right)\right)$

$$
\begin{aligned}
& g(\alpha)=e^{-\alpha} \sum_{i=1}^N \varpi_{m i}+\left(e^\alpha-e^{-\alpha}\right) \sum_{i=1}^N \varpi_{m i} I\left(y_i \neq G_m\left(x_i\right)\right) \\
& \frac{\partial g(\alpha)}{\partial \alpha}=\frac{\partial}{\partial \alpha}\left[e^{-\alpha} \sum_{i=1}^N \varpi_{m i}+\left(e^\alpha-e^{-\alpha}\right) \sum_{i=1}^N \varpi_{m i} I\left(y_i \neq G_m\left(x_i\right)\right)\right] \\
& \left.=-e^{-\alpha} \sum_{i=1}^N \varpi_{m i}+\left(e^\alpha+e^{-\alpha}\right) \sum_{i=1}^N \varpi_{m i} I\left(y_i \neq G_m\left(x_i\right)\right)\right)
\end{aligned}
$$

- 损失函数最小化参数求解过程推导
  
$$
\begin{aligned}
& -e^{-\alpha} \sum_{i=1}^N \varpi_{m i}+\left(e^\alpha+e^{-\alpha}\right) \sum_{i=1}^N \varpi_{m i} I\left(y_i \neq G_m\left(x_i\right)\right)=0 \\
& \therefore\left(e^\alpha+e^{-\alpha}\right) \sum_{i=1}^N \varpi_{m i} I\left(y_i \neq G_m\left(x_i\right)\right)=e^{-\alpha} \sum_{i=1}^N \varpi_{m i}  \\
& 归一化表示：\frac{\left(e^\alpha+e^{-\alpha}\right) \sum_{i=1}^N \varpi_{m i} I\left(y_i \neq G_m\left(x_i\right)\right)}{\sum_{j=1}^N \varpi_{m j}}=\frac{e^{-\alpha} \sum_{i=1}^N \varpi_{m i}}{\sum_{j=1}^N \varpi_{m j}} \\
& \therefore\left(e^\alpha+e^{-\alpha}\right) e_m=e^{-\alpha} \\
& \therefore e^{2 \alpha}=\frac{1-e_m}{e_m} \\
& \therefore \alpha=\frac{1}{2} \operatorname{ln} \frac{1-e_m}{e_m}
\end{aligned}
$$

- 样本权重更新,与其损失函数的大小正相关
  
$$
\begin{aligned}
& \varpi_{m+1, i}=\exp \left(-y_i f_m\left(x_i\right)\right)=\exp \left[-y_i\left(f_{m-1}(x)+\alpha_m G_m(x)\right)\right] \\
& =\exp \left[-y_i\left(f_{m-1}(x)+\alpha_m G_m(x)\right)\right] \\
& =\varpi_{m, i} \exp \left(-y_i \alpha_m G_m\left(x_i\right)\right)
\end{aligned}
$$

- 进行归一化：

$$
\begin{aligned}
&w_{m+1, i}=\frac{\varpi_{m+1, i}}{Z_m} \exp \left(-y_i \alpha_m G_m\left(x_i\right)\right)\\
& \text { 其中规范化因子为：} \mathrm{Z}_m=\sum_{i=1}^N \varpi_{m+1, i} \exp \left(-y_i \alpha_m G_m\left(x_i\right)\right)
\end{aligned}
$$