# 学习理论

- 在假设空间内假设函数为有限个数的时候，存在假设函数，其训练误差和泛化误差的差异在$\gamma$以上的概率为

$$
\begin{array}{ll} 
& p\left(\exists h \in H,\left|\varepsilon\left(h_i\right)-\hat{\varepsilon}\left(h_i\right)\right|>r\right) \\
& = p\left(A_1 \cup A_2 \cup \ldots \cup A_k\right) \\
& \leqslant \sum_{i=1}^k p\left(A_k\right) \\
& = 2 k \exp \left(-2 r^2 n\right)
\end{array}
$$

- 在假设空间内假设函数为有限个数的时候，所有的假设函数其训练误差和泛化误差的差异在$\gamma$以内的概率为

$$
\begin{array}{ll} 
& p\left(\forall h \in H,\left|\varepsilon\left(h_i\right)-\hat{\varepsilon}\left(h_i\right)\right|<\gamma\right) \\
& =p\left(\not \exists h \in H,\left|\varepsilon\left(h_i\right)-\hat{\varepsilon}\left(h_i\right)\right|>\gamma\right) \\
& =1-p\left(\exists h \in H,\left|\varepsilon\left(h_i\right)-\hat{\varepsilon}\left(h_i\right)\right|>\gamma\right) & \\
& =1-2 k \exp \left(-2 \gamma^2 n\right) & \\
\end{array}
$$

- 给定$\delta$，得出假设空间的所有假设函数的训练无差和泛化误差的差异$\gamma$范围内的样本n的数量满足的关系，和k仅仅只需满足对数关系

$$
\begin{aligned}
& \delta=2 k \exp \left(-2 \gamma^2 n\right) & \\
& \frac{\delta}{2 k}=e^{-2 \gamma^2 n} & \\
& \quad-2 \gamma^2 n=\log \frac{\delta}{2 k} \\
& \quad n=\frac{1}{2 r^2} \log \frac{2 k}{\delta} \\
\end{aligned}
$$

- 假设空间里经验误差最好的假设和泛化误差最好的假设之间的泛化误差差异在一定的范围内

$$
\begin{aligned}
& \hat{h}=\operatorname{argmin}_{h \in H} \hat{\varepsilon}(h) \\
& \varepsilon(\hat{h}) \leqslant \hat{\varepsilon}(\hat{h})+\gamma \\
& \leqslant \hat{\varepsilon}\left(h^*\right)+\gamma \\
& \leqslant \varepsilon\left(h^*\right)+r+\gamma\\
& =\varepsilon\left(h^*\right)+2\gamma \\
\varepsilon(\hat{h}) & \leqslant\left(\min _{h \in H} \varepsilon(h)\right)+2\gamma \\
& \delta =2 k e^{-2 \gamma^2 n} \\
& \gamma =\sqrt{\frac{1}{2n} \log \frac{2k}{\delta}}
\end{aligned}
$$

- vc维度的定义

打散m个点：是指对于这m个点的所有2^m种可能的二分类标签组合，假设空间中都存在一个函数能够完美实现这些分类。
VC维度为d意味着存在这样一组d个点可以被完全打散，但是任何d+1个点，都不能被完全打散（即总有一种标签组合是无法被假设空间中的函数实现的）。
