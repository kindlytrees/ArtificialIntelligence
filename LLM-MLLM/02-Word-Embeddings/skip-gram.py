import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter

# 示例语料库
corpus = [
    "we are what we repeatedly do",
    "excellence then is not an act",
    "but a habit"
]

# 超参数
context_size = 2  # 上下文窗口大小
embedding_dim = 10  # 词向量维度
epochs = 100
learning_rate = 0.001

# 数据预处理
def preprocess_corpus(corpus):
    tokens = [sentence.lower().split() for sentence in corpus]
    vocab = {word for sentence in tokens for word in sentence}
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    return tokens, word_to_idx, idx_to_word

# 生成Skip-Gram训练样本
def generate_training_data(tokens, word_to_idx, context_size):
    data = []
    for sentence in tokens:
        indices = [word_to_idx[word] for word in sentence]
        for i, target in enumerate(indices):
            context = [
                indices[j] for j in range(max(0, i - context_size), min(len(indices), i + context_size + 1)) if j != i
            ]
            for context_word in context:
                data.append((target, context_word))
    return data

tokens, word_to_idx, idx_to_word = preprocess_corpus(corpus)
training_data = generate_training_data(tokens, word_to_idx, context_size)

# Skip-Gram模型定义
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, target):
        target_embedding = self.embeddings(target)
        output = self.linear(target_embedding)
        return output

# 初始化模型
vocab_size = len(word_to_idx)
model = SkipGramModel(vocab_size, embedding_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    total_loss = 0
    for target, context_word in training_data:
        target_tensor = torch.tensor([target], dtype=torch.long)
        context_word_tensor = torch.tensor([context_word], dtype=torch.long)
        
        # 前向传播
        output = model(target_tensor)
        loss = loss_function(output, context_word_tensor)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {total_loss/len(training_data)}")

# 测试词向量
test_word = "habit"
test_word_idx = word_to_idx[test_word]
print(f"Embedding for '{test_word}': {model.embeddings.weight[test_word_idx].detach().numpy()}")
