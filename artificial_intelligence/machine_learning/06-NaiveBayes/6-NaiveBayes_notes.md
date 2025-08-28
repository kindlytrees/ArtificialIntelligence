# NaiveBayes

## 公式说明

$$
\begin{aligned}
& p\left(x_j=1 \mid y=1\right)=\phi_j \mid y=1 \\
& p\left(x_j=0 \mid y=1\right)=1-\phi_j \mid y=1 \\
& p\left(x_j=1 \mid y=0\right)=\phi_j \mid y=0 \\
& p\left(x_j=0 \mid y=0\right)=1-\phi_j \mid y=0
\end{aligned}
$$

## 公式推导

- 参数phi的估计

$$
\begin{aligned}
& L\left(\phi_y, \phi_j\left|y=0, \phi_j\right| y=1\right) \quad\left(\phi_j \mid y=1\right)=P\left(x_j=1 \mid y=1\right) \\
& L= \prod_{i=1}^n P\left(x^{(i)}, y^{(i)}\right)=\prod_{i=1}^n p\left(y^{(i)}\right) P\left(x^{(i)} \mid y^{(i)}\right) \\
& l=\sum_{i=1}^n \log \left[p\left(y^{(i)}\right) p\left(x^{(i)} \mid y^{(i)}\right)\right]=\sum_{i=1}^n\left[\log p\left(y^{(i)}\right)+\log p\left[x^{(i)} \mid y^{(i)}\right)\right] \\
& \frac{\partial l}{\phi_y}=\frac{\partial \Sigma \log p\left(y^{(i)}\right)}{\partial \phi_y}=\frac{\partial \Sigma \log \left[\phi_y^{y_i} \cdot\left(1-\phi_y\right)^{\left(1-y_i\right)}\right]}{\partial \phi_y}=\frac{\partial \Sigma \left[y_i \log \phi_y+\left(1-y_i\right) \log \left(1-\phi_y\right)\right]}{\partial \phi_y} \\
& =\Sigma\left[y_i \frac{1}{\phi_y}+\left(1-y_i\right) \frac{-1}{1-\phi_y}\right]=\sum_{y_i=1} \frac{1}{\phi_y}+\sum_{y_i=0} \frac{-1}{1-\phi_y}=0 \\
& \frac{s}{\phi_y}=\frac{n-s}{1-\phi_y} \Rightarrow \quad s-s \phi_y=n \phi_y-s \phi_y \\
& \quad \phi_y=\frac{s}{n}
\end{aligned}
$$

- 条件概率分布的估计

$$
\begin{aligned}
& l=\sum_{i=1}^n\left[\log p(y(i))+\log p\left(x^{(i)} \mid y(i)\right)\right] \\
& =\frac{\partial l}{\partial \phi_j \mid y=1} \\
& =\sum_{i=1}^n 1(y(i)=1)\left[\log p(y(i))+\log p\left(x^{(i)} \mid y^{(i)}\right)\right] \\
& =\sum_{i=1}^n 1(y(i)=1) \log \prod _{k} p\left(x_k^{(i)} \mid y(i)\right) \\
& =\sum_{i=1}^n 1\left(y^{(i)}=1\right) \sum_k \log p\left(x_k^{(i)} \mid y(i)\right) \\
& =\sum_{i=1}^n 1\left(y^{(i)}=1\right)\left(\sum_{x_j=1} \log p\left(x_j^{(i)} \mid y(i)\right)+\sum_{x_j=0} \log p\left(x_j^{(i)} \mid y(i)\right)\right) \\
& =\sum_{i=1}^n 1\left(y^{(i)}=1\right)\left(\sum_{x_j=1} \frac{1}{\phi_j \mid y=1}+\sum_{x_j=0} \frac{-1}{1-\phi_j \mid y=1}\right)=0 \\
& \frac{s}{\phi_j \mid y=1}=\frac{t}{1-\phi_j \mid y=1} \\
& {\phi_j \mid y=1} = \frac{s}{s+t} = \frac{\sum_{i=1}^n 1\left(y^{(i)}=1\right) \cdot 1\left(x_j^{(i)}=1\right)}{\sum_{i=1}^n 1\left(y^{(i)}=1\right)}.
\end{aligned}
$$

## 多项式分布的相关描述

$$
\begin{aligned}
p= & p\left(p_1, p_2, \cdots p_k\right), \sum_{i=1}^k p_i=1 \\
& \sum_{i=1}^k x_i=n \\
p\left(x=\left(x_1, x_2, \cdots x_k\right)\right)= & \frac{n!} 
{x_{1}!x_{2}!\cdots x_{k}!}  p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}
\end{aligned}
$$

## 实验代码分析

一些numpy和pandas库的使用方法和技巧

- 列出表格中的所有属性离别列
```
categorical = [var for var in df.columns if df[var].dtype=='O']

['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']
```

- 列出某个属性列中属性可能的集合
```
df.workclass.unique()

array(['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
       'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
      dtype=object)
```

- 处理属性列中的奇异值
```
df['workclass'].replace('?', np.NaN, inplace=True)
```

- 统计属性列的数量特性
```
df.workclass.value_counts()

workclass
Private             22696
Self-emp-not-inc     2541
Local-gov            2093
?                    1836
State-gov            1298
Self-emp-inc         1116
Federal-gov           960
Without-pay            14
Never-worked            7
Name: count, dtype: int64
```

```
X_train[categorical].isnull().mean()
```

## 机器学习中的评估度量指标及含义

AP
MAP
ROC
F1-Score
Confusion Matrix