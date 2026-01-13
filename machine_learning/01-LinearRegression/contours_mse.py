import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import copy

#X, Y, Coef = datasets.make_regression(n_samples=600, n_features=1, noise=20, random_state=0, bias=50,coef=True)
# # 损失函数的定义
def cost_function(theta0, theta1, X, y):
    m = len(y)
    predictions = theta0 + theta1 * X
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# 梯度下降的模拟
def gradient_descent(X, y, alpha, iterations):
    theta0, theta1 = 3, -3  # 初始点
    m = len(y)
    history = [(theta0, theta1)]
    
    for _ in range(iterations):
        predictions = theta0 + theta1 * X
        # 不同分量的梯度分量(偏导数计算)
        temp0 = theta0 - alpha * (1 / m) * np.sum(predictions - y)
        temp1 = theta1 - alpha * (1 / m) * np.sum((predictions - y) * X)
        theta0, theta1 = temp0, temp1
        history.append((theta0, theta1))
    
    return history

# 生成示例数据
# X = np.array([1, 2, 3, 4, 5])
# y = np.array([1, 2, 2.5, 4, 5])
X, y, _ = datasets.make_regression(n_samples=20, n_features=1, noise=0, random_state=0,coef=True)
# 自定义斜率 (这里设为 5)
slope = 1

# 添加自定义斜率并加入噪声
noise = np.random.normal(0, 0.002, size=y.shape)  # 生成均值为0，标准差为10的噪声
y = slope * X.flatten() + noise

# 打印生成的斜率数据和噪声
print("X: ", X[:5])  # 前5个数据点的特征
print("y: ", y[:5])  # 前5个数据点的目标值

# 绘制等高线
theta0_vals = np.linspace(-3, 3, 100)
theta1_vals = np.linspace(-3, 3, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# 计算不同参数下的损失函数J
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        J_vals[i, j] = cost_function(theta0_vals[i], theta1_vals[j], X, y)

'''meshgrid生成的结果解释说明
#x = np.array([1, 2, 3])  # 横坐标
#y = np.array([10, 20])   # 纵坐标
#X, Y = np.meshgrid(x, y)
#X[i, j] 和 Y[i, j] 一起组成二维平面上一个点 (x, y)。
#如上述示例的返回结果为

X:
 [[1 2 3]
  [1 2 3]]

Y:
 [[10 10 10]
  [20 20 20]]
'''
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

'''
levels=np.logspace(0, 3, 20)
表示绘制20条等高线,高度从10⁰ = 1 到 10³ = 1000,对数间隔（更适合对数级别变化的函数）
'''
plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(0, 3, 20), cmap='jet') # 数值从 0.1 开始，一直到 1000，并且它们是按照对数比例间隔的（20个间隔而不是线性间隔）
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.title('Cost Function Contours')

# 梯度下降轨迹
history = gradient_descent(X, y, alpha=0.02, iterations=100)
history = np.array(history)
plt.plot(history[:, 0], history[:, 1], 'rx-', markersize=8, linewidth=2)
plt.show()