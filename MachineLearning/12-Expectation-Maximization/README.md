# gmm-em-clustering

高斯混合模型（GMM 聚类）的 EM 算法实现。

Jensen不等式的公式可以从不同的角度来看
- 离散版本
- 连续变量版本

以上jensen不等式在em中运用时，函数f可以是连续的函数定义，比如log函数，但变量可能是离散的变量，也可能是连续的随机变量对吗?上述的变量如果是连续的变量来看的时候，其实E(f(x)]的值在弦上和函数f(E(x)]之间的区域内，而不是在弦上的点

这个重心点 (E[X], E[f(X)]) 位于由函数图像构成的凸区域内，即您所说的“弦上和函数f(E(x))之间的区域内”。这个区域在数学上被称为函数 f 的上境图 (Epigraph) 的凸包。由于 f 是凸函数，这个区域整体都在函数曲线的上方（或线上）。因此，E[f(X)] 的纵坐标值必然大于等于 f(E[X]) 的纵坐标值。

z隐变量有可能是
- 离散的，如混合高斯模型中的类别数目
- 连续的，如因子分析中的隐向量z

Jensen不等式的右边被称为证据下界 (Evidence Lower Bound, ELBO)

EM算法应用示例
- GMM(z为离散随机变量，隐变量)
- Factor analysis(z为连续的随机向量，隐空间)
- HMM

em算法在GMM应用中的M-step的证明思路和GDA思路基本相似，待确认?

摘自大模型回答：

GMM中M-step的推导思路与GDA的参数估计思路，本质上是完全相同的。唯一的区别在于，GDA处理的是“硬标签”（Hard Labels），而GMM的M-step处理的是E-step提供的“软标签”（Soft Labels）。

这个视角完美地揭示了EM算法的本质：它通过迭代的方式，在“猜测标签”（E-step）和“基于猜测的标签进行有监督学习”（M-step）之间循环，直到收敛。这使得一个复杂的无监督问题，被分解成了一系列简单的、类似有监督学习的子问题。

# 相关文章

[高斯混合模型 EM 算法的 Python 实现](http://www.codebelief.com/article/2017/11/gmm-em-algorithm-implementation-by-python/)

# 测试结果

![](http://static.codebelief.com/2017/11/24/gmm.png)

## Reference
- https://www.kaggle.com/code/sidharth178/customer-segmentation-kmeans-dbscan-gmm-agglo
- https://www.kaggle.com/code/bburns/iris-exploration-pca-k-means-and-gmm-clustering
- https://zhuanlan.zhihu.com/p/30483076
