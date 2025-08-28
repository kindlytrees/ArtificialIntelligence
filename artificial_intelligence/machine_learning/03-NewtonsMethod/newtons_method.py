import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import copy


THETA = np.random.normal(0, 0.1, 3).reshape(3, 1) # learnable parameters
# make_blobs函数是为聚类产生数据集
X, Y = make_blobs(n_samples=600, centers=2, n_features=2, random_state=3)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


X0_train = np.ones([X_train.shape[0],1],dtype=X_train.dtype)
X0_test = np.ones([X_test.shape[0],1], dtype=X_test.dtype)
X_train_original = copy.deepcopy(X_train)
X_train = np.concatenate((X0_train,X_train), axis=1)
X_test = np.concatenate((X0_test, X_test), axis=1)

# 定义 sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 逻辑回归的牛顿法求解
def logistic_regression_newtown(X, y, num_steps=100, tol=1e-6, reg_param=1e-5):
    """
    使用牛顿法优化逻辑回归的参数
    :param X: 特征矩阵 (n_samples, n_features)
    :param y: 标签向量 (n_samples,)
    :param num_steps: 最大迭代次数
    :param tol: 收敛阈值
    :return: 优化后的参数 theta
    """
    # 初始化参数 theta
    theta = np.zeros(X.shape[1])
    
    for step in range(num_steps):
        # 计算预测值 h(X * theta)
        z = np.dot(X, theta)
        h = sigmoid(z)
        
        # 计算梯度
        gradient = np.dot(X.T, h - y)
        
        # 计算 Hessian 矩阵,公式推导参考PPT
        D = np.diag(h * (1 - h))
        H = np.dot(X.T, np.dot(D, X))

        H += reg_param * np.eye(X.shape[1])
        
        # 使用 Newton-Raphson 方法更新 theta
        H_inv = np.linalg.inv(H)  # 计算 Hessian 的逆矩阵
        theta_update = np.dot(H_inv, gradient)
        
        # 更新参数 theta
        theta -= theta_update
        
        # 检查收敛条件
        if np.linalg.norm(theta_update, ord=1) < tol:
            print(f"Converged in {step + 1} steps.")
            break

    return theta

# 创建示例数据
# np.random.seed(0)
# X = np.random.randn(100, 3)  # 100个样本, 3个特征
# y = (np.random.rand(100) > 0.5).astype(int)  # 生成0/1标签

# 调用牛顿法求解逻辑回归的参数
theta_optimized = logistic_regression_newtown(X_train, Y_train)

H_test = np.zeros([Y_test.shape[0], 1], dtype=Y_test.dtype)
i = 0
for x, y in zip(X_test, Y_test):
    H_test[i,0] = np.around(sigmoid(np.dot(x, theta_optimized)))
    i+=1
plt.figure(1)
x = np.linspace(-7, 4, 50)
plt.scatter(X_test[:, 1], X_test[:, 2], c=H_test[:, 0], edgecolors='white', marker='s')
plt.show()

# 输出结果
print("Optimized Theta:", theta_optimized)
