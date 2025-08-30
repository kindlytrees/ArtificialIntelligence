# 支持向量机

## 几何间隔定义的推导

<p align="center">
  <img src="./geometry_margin.png" width="300"/>
</p>

- 如图所示直线上方的几何距离的计算公式如下（此时y(i)=1，下方的原点类似，统一成最后一行公式）

$$
\begin{array}{r}
w^{\top}\left(x^{(i)}-\frac{w}{\|w\|} \gamma ^{(i)}\right)+b=0 \\
w^{\top} x^{(i)}-\frac{w^{\top} w}{\|w\|} \gamma^{(i)}+b=0 \\
w^{\top} x^{(i)}+b=\frac{w^{\top} w}{\|w\|} \gamma^{(i)} \\
\gamma^{(i)}=\frac{w^{\top} x^{(i)}+b}{\|w\|} \\
=\frac{w^{\top} x^{(i)}}{\|w\|}+\frac{b}{\|w\|} \\
\gamma^{(i)}=y^{(i)}\left(\left(\frac{w}{\| w \|}\right)^{\top} x^{(i)}+\frac{b}{\|w\|}\right)
\end{array}
$$

## 最优间隔分类器的定义及问题等价变换

- 几何间隔最小值的最大化

$$  
\begin{aligned}
&\max _{\gamma, \omega, b} \gamma\\
&\begin{array}{ll}
\text { s.t. } & y^{(i)}\left(\omega^{\top} x^{(i)}+b\right) \geqslant \gamma \\
& \|\omega\|=1
\end{array}
\end{aligned}
$$

- 解除w的模为1的限制，变为最小函数间隔约束

$$
\begin{aligned}
& \max _{\hat{\gamma}, w, b} \frac{\hat{\gamma}}{\|w \|} \\
& \text { s.t. } \quad y^{(i)}\left(w^{\top} x^{(i)}+b\right) \geqslant \hat{r} \quad\{i=1,2, \cdots, n\} \\
\end{aligned}
$$

- w的模的倒数不是凸函数说明：从几何上看，连接凸函数图上任意两点的线段，总是在函数图的上方或与函数图重合。如w为一元参数时，连接 (1,1)和 (−1,1)的线段是一条水平线y=1。而函数f(x)=1/∣x∣在x=0处无限高，这意味着这条线段在某些点上位于函数图的下方（例如在x=0处），因此它不是凸函数。

- 可以将最小函数间隔根据w进行缩放后设置为常数1，同时将w的模的倒数最大化等价为模的平方的最小化，转换后的优化目标和约束函数以及朗格朗日乘数子函数为  

$$
\begin{aligned}
& \quad \min _{w, b} \frac{1}{2}\|w\|^2 \\
& \quad \text { s.t. } \quad y^{(i)}\left(w^{\top} x^{(i)}+b\right) \geqslant 1\{i=1,2, \cdots n\} \\
& L(\alpha, w, b)=\frac{1}{2}\|w\|^i-\sum_{i=1}^n \alpha_i\left[g^{(i)}\left(w^{\top} x^{(i)}+b\right)-1\right] 
\end{aligned}
$$

- 从几何上看，凸函数连接函数图上任意两点的线段，总是在函数图的上方或与函数图重合。


## 约束优化问题的代数几何含义解释
- 梯度方向是垂直于切线方向，和等值线(面)法向量保持一致
这是理解朗格朗日乘数法的关键。直观地，梯度向量指向函数值增长最快的方向。对于一个等值面 F(x,y,z)=c，如果你沿着等值面移动，函数值F不变，这意味着你沿着任何切线方向移动时的变化率为零。如果梯度与某个方向的点积为零，那么梯度就垂直于那个方向。因此，梯度与等值面上的所有切线方向都垂直，从而梯度就是等值面的法向量。
- 函数的极值点切线方向为局部的等值线，其导数为0，目标函数看成等值线族，如果目标函数的极值位置和约束等值线(面)的法线方向不一致(切线重合不平行），则沿着约束等值线方向变量值的小的改变在目标函数上会有梯度的分量不为0，亦即此处的交点不是目标函数的极值点。可以在约束线上沿着交点向某个方向移动，从而进入一个f值更高（或更低）的等值线区域

## 最优间隔分类器的对偶问题定义及求解

$$
\begin{aligned}
& L(\omega, b, \alpha)=\frac{1}{2} \left ||\omega|\right|^2-\sum_{i=1}^n \alpha_i\left[y^{(i)}\left(\omega^{\top} x^{(i)}+b\right)-1\right] \\
& \begin{aligned}
& \nabla_\omega L= \nabla_\omega\left(\frac{1}{2} \omega^{\top} \omega-\sum_{i=1}^n \alpha_i y^{(i)} \omega^{\top} x^{(i)}\right) \\
&=\omega-\sum_{i=1}^n \alpha_i y^{(i)} x^{(i)}=0 \\
& \omega= \sum_{i=1}^n \alpha_i y^{(i)} x^{(i)} \\
& \frac{\partial L}{\partial b}=-\sum_{i=1}^n \alpha_i y^{(i)}=0 \\
&  L(\omega, b, \alpha)=\frac{1}{2}\left\|\sum_{i=1}^n \alpha_i y^{(i)} x^{(i)}\right\|^2-\sum_{i=1}^n \alpha_i y^{(i)} \left(\sum_{j=1}^n \alpha_j y^{(j)} x^{(j)} \right)^{T} \cdot x^{(i)} \\
&=\frac{1}{2} \sum_{i=1}^n \alpha_i y^{(i)} \left(x^{(i)}\right)^{T} \cdot \sum_{j=1}^n \alpha_j y^{(i)} x^{(j)}-\sum_{i=1}^n \alpha_i y^{(i)} \left(x^{(i)}\right)^{T} \sum_{j=1}^n \alpha_j y^{(j)} x^{(j)}+\sum_{i=1}^n \alpha_i \\
&=\sum_{i=1}^n \alpha_i-\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y^{(i)} \left(x^{(i)}\right)^{T} \cdot y^{(j)} x^{(j)} \\
&=\sum_{i=1}^n \alpha_i-\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y^{(i)} y^{(j)}\left\langle x^{(i)}, x^{(j)}\right\rangle
\end{aligned}
\end{aligned}
$$

通过SMO算法求解处alpha后，根据$w=\sum_{i=1}^n \alpha_i y^{(i)} x^{(i)}$求出w，进一步通过如下的推导求出b

$$
\begin{aligned}
w & =\sum_{i=1}^n \alpha_i y^{(i)} x^{(i)} \\
w x+b & =0 \\
\max w x^{(i)}+b & =-1 \quad\left(y^{(i)}=-1\right) \\
\min w x^{(i)}+b & =1 \quad\left(y^{(i)}=1\right) \\
\max_{y^{(i)}=-1} & w^{\top} x^{(i)}+b+\min _{y(i)=1}{w^{\top} x^{(i)}+b}=0 \\
b & =\frac{\max _{y^{(i)}=-1}{w^{\top} x^{(i)}}+\min _{y(i)=1} {w^{\top} x^{(i)}}}{2}
\end{aligned}
$$

## prompts
- 请给出广义拉格朗日函数在不等式约束函数边界核内部约束时函数取得极值的可视化示意图并加以说明。

## 实验部分准备
- 决策边界的可视化
- jupyter notebook文件的执行和解释