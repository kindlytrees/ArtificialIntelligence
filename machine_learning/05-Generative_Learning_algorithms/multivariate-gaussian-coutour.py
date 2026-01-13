import numpy as np
import matplotlib.pyplot as plt

def multivariate_gaussian(pos, mu, Sigma):
    """计算二维网格点的多元高斯分布概率密度。"""
    n = mu.shape[0]
    # sigma行列式和sigma的逆
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)

    # 多元高斯分布的前面系数
    N = np.sqrt((2 * np.pi)**n * Sigma_det)
    
    '''
    计算指数部分。通过爱因斯坦和的语法定义进行计算 
    np.einsum 的核心原理在于它提供了一种声明式的方式来定义张量操作。
    p.einsum(subscripts, operand1, operand2, ..., [out=None])
    subscripts: 字符串表示张量的索引与缩并规则。
    每个操作数的索引用逗号分隔，对应一个输入张量。-> 后面表示输出张量的索引。
    '...k': 表示 A 是一个最后一维为 k 的任意维数组，... 表示前面可以有任意维度（比如批量维度）。
    'kl': 表示 B 是一个 k × l 的矩阵（比如协方差逆矩阵）。
    '...l': 表示 C 是一个最后一维为 l 的数组，与 A 形状相同，但结尾维度从 k 变为 l。
    输出'...'意味着结果仍然保留所有批次维度，只对 k 和 l 维进行了缩并（求和）。
    多元高斯分布中指数项内部的 马氏距离平方 (squared Mahalanobis distance)
    '''
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
    
    return np.exp(-fac / 2) / N

# 创建网格，以方便多元高斯分布的可视化效果图
X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(X, Y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# 均值为0
mu = np.array([0, 0])

# 定义几种不同的协方差矩阵
covariances = [
    np.array([[1.0, 0.0], [0.0, 1.0]]),  # 无相关性，方差相等
    np.array([[1.0, 0.8], [0.8, 1.0]]),  # 正相关性
    np.array([[1.0, -0.8], [-0.8, 1.0]]),  # 负相关性
    np.array([[0.5, 0.0], [0.0, 2.0]]),  # 方差不等
]

# 设置等高线图的布局
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# 绘制每个协方差矩阵对应的等高线图
for i, Sigma in enumerate(covariances):
    Z = multivariate_gaussian(pos, mu, Sigma)
    
    ax = axs[i//2, i%2]
    ax.contour(X, Y, Z, levels=10, cmap='viridis')
    ax.set_title(f'Covariance Matrix:\n{Sigma}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

plt.tight_layout()
plt.show()
