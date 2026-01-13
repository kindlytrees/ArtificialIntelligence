import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 生成二维分类数据
# 这是生成一个特殊的二维数据集，被称为“同心圆”数据集。
# factor=0.5: 内圆的半径是外圆半径的0.5倍。
X, y = datasets.make_circles(n_samples=100, factor=0.5, noise=0.1)

# 使用SVM模型，选择RBF核
# C 值越大，模型越倾向于正确分类所有训练样本，可能会导致更窄的间隔和过拟合。C 值越小，模型允许更多的分类错误，但会获得更宽的间隔，可能更具泛化能力。
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X, y)

# 创建二维网格数据
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 对网格点进行预测
# 这是SVM的一个重要方法。它返回每个样本到决策边界的有符号距离。
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print(Z.min(), Z.max())

# 绘制决策边界和数据点
# ontourf 用于绘制填充的等高线图。它根据 Z 的值在 (xx, yy) 网格上填充颜色。
# 指定了填充的等高线级别。它从 Z 的最小值开始，一直到 0（决策边界），生成7个均匀分布的级别。这意味着它将只填充决策边界以下（即 Z 为负值）的区域，对应一个类别。PuBu 是一个颜色映射，从浅紫蓝色到深蓝色的渐变。
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')  # 决策边界
# contours = plt.contour(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10), linewidths=2, colors='darkred')

# # 为等高线添加数值标签
# plt.clabel(contours, inline=True, fontsize=10)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.spring, edgecolors='k')
plt.title('SVM with RBF kernel')
plt.show()
