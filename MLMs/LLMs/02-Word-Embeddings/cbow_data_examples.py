import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
from collections import Counter

# 示例语料库
# corpus = [
#     "we are what we repeatedly do",
#     "excellence then is not an act",
#     "but a habit"
# ]

# # 数据预处理
# def preprocess_corpus(corpus):
#     tokens = [sentence.lower().split() for sentence in corpus]
#     vocab = {word for sentence in tokens for word in sentence}
#     word_to_idx = {word: i for i, word in enumerate(vocab)}
#     idx_to_word = {i: word for word, i in word_to_idx.items()}
#     return tokens, word_to_idx, idx_to_word

corpus = [
    "we are what we repeatedly do",
    "excellence then is not an act",
    "but a habit"
]

# Step 1: 分词并构建词汇表
vocab = set()
sentences = []
for line in corpus:
    words = line.lower().split()
    sentences.append(words)
    vocab.update(words)

# 创建词到索引的映射
word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(vocab)

print("Vocabulary:", word_to_idx)
print()

# Step 2: 构建 CBOW data 和 label
context_window = 2  # 左右各两个词
data = []   # 每个元素是上下文词的索引列表（长度为 4）
labels = [] # 每个元素是中心词的索引（单个整数）

for sentence in sentences:
    n = len(sentence)
    for i in range(context_window, n - context_window):
        # 上下文：i-2, i-1, i+1, i+2
        context_indices = [
            word_to_idx[sentence[i - 2]],
            word_to_idx[sentence[i - 1]],
            word_to_idx[sentence[i + 1]],
            word_to_idx[sentence[i + 2]]
        ]
        target_index = word_to_idx[sentence[i]]

        data.append(context_indices)
        labels.append(target_index)

# Step 3: 打印结果
print("CBOW Data (context word indices):")
for d in data:
    print(d, "->", [idx_to_word[idx] for idx in d])

print("\nLabels (target word indices):")
for l in labels:
    print(l, "->", idx_to_word[l])