# Decision Tree

- 基尼系数定义
  
$$
\operatorname{Gini}(p)=\sum_{k=1}^K p_k\left(1-p_k\right)=1-\sum_{k=1}^K p_k^2
$$

- 加权基尼系数

$$
G(D, A)=\sum_{v \in \text { values }(A)} \frac{\left|D_v\right|}{|D|} G\left(D_v\right)
$$

- 和方差度量

$$
\underbrace{\min }_{A, s}[\underbrace{\min }_{c_1} \sum_{x_i \in D_1(A, s)}\left(y_i-c_1\right)^2+\underbrace{\min }_{c_2} \sum_{x_i \in D_2(A, s)}\left(y_i-c_2\right)^2]
$$