import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def multivariate_gaussian(pos, mu, Sigma):
    """Calculate the multivariate Gaussian distribution on a 2D grid."""
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi)**n * Sigma_det)
    
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
    
    return np.exp(-fac / 2) / N

# 创建网格
X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(X, Y)
pos = np.empty(X.shape + (2,))
print(f'pos shape is {pos.shape}')
pos[:, :, 0] = X
pos[:, :, 1] = Y

# 均值为 0
mu = np.array([0, 0])

# 可自定义的协方差矩阵
Sigma = np.array([[1.0, 0.8], 
                  [0.8, 1.0]])

# Sigma = np.array([[1.0, 0.0], 
#                   [0.0, 1.0]])

# 计算二维高斯分布
Z = multivariate_gaussian(pos, mu, Sigma)

# 三维绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 设置标签和标题
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Probability Density')
ax.set_title('Bivariate Gaussian Distribution')

plt.show()
