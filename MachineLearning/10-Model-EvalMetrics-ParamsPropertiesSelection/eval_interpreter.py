import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# --- 1. 模拟二分类数据 ---
np.random.seed(42) # 为了结果可复现

n_samples = 1000 # 总样本数
n_positive_class = 50 # 正类别样本数 (非常不平衡，只有5%)
n_negative_class = n_samples - n_positive_class

# 真实标签 (y_true)
y_true = np.array([1] * n_positive_class + [0] * n_negative_class)
np.random.shuffle(y_true) # 打乱顺序

# 模型预测的概率分数 (y_scores)
# 假设模型对正类别样本倾向于给出更高的分数，但存在一些错误
y_scores = np.zeros(n_samples)

# 为正类别样本生成分数
# 大部分在0.6-0.9之间，少量低分（假阴性）
y_scores[y_true == 1] = np.random.normal(loc=0.75, scale=0.15, size=n_positive_class)
y_scores[y_true == 1] = np.clip(y_scores[y_true == 1], 0.05, 0.99) # 确保分数在合理范围

# 为负类别样本生成分数
# 大部分在0.1-0.4之间，少量高分（假阳性）
y_scores[y_true == 0] = np.random.normal(loc=0.25, scale=0.15, size=n_negative_class)
y_scores[y_true == 0] = np.clip(y_scores[y_true == 0], 0.01, 0.95) # 确保分数在合理范围

print(f"模拟数据集样本数: {n_samples}")
print(f"正类别样本数: {n_positive_class} ({n_positive_class/n_samples*100:.1f}%)")
print(f"负类别样本数: {n_negative_class} ({n_negative_class/n_samples*100:.1f}%)")
print("-" * 30)

# --- 2. 绘制 ROC 曲线并计算 AUC ---

# 计算 FPR, TPR 和阈值
# roc_curve 函数会返回三个数组：
# - fpr: False Positive Rate (假阳性率)
# - tpr: True Positive Rate (真阳性率 / 召回率)
# - thresholds: 对应的概率阈值
fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)

# 计算 AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) # 1行2列的第一个图
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR) / Recall')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)

# --- 3. 绘制 PR 曲线并计算 AP ---
# 计算精确率 (Precision), 召回率 (Recall) 和阈值
# precision_recall_curve 函数会返回三个数组：
# - precision: 精确率
# - recall: 召回率
# - thresholds: 对应的概率阈值
precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)

# 计算 AP (Average Precision)
# average_precision_score 直接计算 PR 曲线下的面积
average_precision = average_precision_score(y_true, y_scores)

plt.subplot(1, 2, 2) # 1行2列的第二个图
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {average_precision:.2f})')
# 随机分类器的基线通常是正类别比例 (TP / (TP + FP))
# 在不平衡数据中，如果模型随机分类，precision会接近正类别比例
plt.axhline(y=n_positive_class / n_samples, color='green', linestyle='--', label=f'Baseline (Precision = {n_positive_class/n_samples:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc="lower left")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 4. 详细解释计算细节 ---
print("\n--- ROC 曲线和 AUC 计算细节解释 ---")
print("ROC 曲线的绘制原理：")
print("1. 对模型输出的所有概率分数 (y_scores) 进行降序排序。")
print("2. 遍历每一个分数，将它作为一个分类的'阈值' (threshold)。")
print("3. 对于每一个阈值 T：")
print("   - 将所有预测概率 >= T 的样本判定为 '正类'。")
print("   - 将所有预测概率 < T 的样本判定为 '负类'。")
print("4. 基于此判定，计算混淆矩阵中的 TP, FP, TN, FN：")
print("   - TP (True Positives): 真实为正且预测为正的样本数。")
print("   - FP (False Positives): 真实为负但预测为正的样本数。")
print("   - TN (True Negatives): 真实为负且预测为负的样本数。")
print("   - FN (False Negatives): 真实为正但预测为负的样本数。")
print("5. 计算对应阈值下的：")
print("   - 真阳性率 (TPR) / 召回率 (Recall) = TP / (TP + FN)")
print("   - 假阳性率 (FPR) = FP / (FP + TN)")
print("6. 将所有 (FPR, TPR) 点连接起来，就构成了 ROC 曲线。")
print("AUC (Area Under the ROC Curve) 解释：")
print("   - AUC 是 ROC 曲线下方的面积，值介于 0 到 1 之间。")
print("   - AUC 越大，表示模型区分正负样本的能力越强。")
print("   - 0.5 的 AUC 表示随机分类器，1.0 表示完美分类器。")
print(f"本例中，ROC_AUC = {roc_auc:.2f}，表明模型具有较好的区分能力。")

print("\n--- PR 曲线和 AP 计算细节解释 ---")
print("PR 曲线的绘制原理：")
print("1. 同样对模型输出的所有概率分数 (y_scores) 进行降序排序。")
print("2. 遍历每一个分数，将它作为一个分类的'阈值' (threshold)。")
print("3. 对于每一个阈值 T：")
print("   - 同样将所有预测概率 >= T 的样本判定为 '正类'。")
print("   - 计算混淆矩阵中的 TP, FP, TN, FN。")
print("4. 计算对应阈值下的：")
print("   - 精确率 (Precision) = TP / (TP + FP)")
print("   - 召回率 (Recall) = TP / (TP + FN)  (与 ROC 曲线的 TPR 相同)")
print("5. 将所有 (Recall, Precision) 点连接起来，就构成了 PR 曲线。")
print("AP (Average Precision) 解释：")
print("   - AP 是 PR 曲线下方的面积。它衡量了模型在不同召回率水平下的平均精确率。")
print("   - AP 越高，表示模型在召回正样本的同时，能够保持较高的精确率（即误报率低）。")
print("   - 相比于 AUC，AP 对类别不平衡问题更敏感，因为它直接涉及精确率的计算，而精确率的分母 (TP + FP) 受 FP 数量的影响较大。当负样本很多时，少量的 FP 就会显著拉低精确率。")
print(f"本例中，Average Precision (AP) = {average_precision:.2f}。尽管 AUC 看起来不错，但 AP 会更诚实地反映模型在识别少数类时的真实表现，因为它考虑了高召回率下保持精确率的挑战。")
print(f"基线 (随机猜测的精确率) 为正类别比例: {n_positive_class / n_samples:.2f}。")