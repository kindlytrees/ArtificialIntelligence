import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import random

# 简单的语料库
corpus = [
    "I like deep learning",
    "I like NLP",
    "I enjoy flying",
    "deep learning is amazing",
    "NLP is a field of AI",
    "I like AI"
]

# 构建词汇表
words = [word for sentence in corpus for word in sentence.lower().split()]
vocab = Counter(words)
vocab_size = len(vocab)

word_to_idx = {word: i for i, word in enumerate(vocab.keys())}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# 生成Skip-gram的训练数据
def generate_skipgram_data(corpus, word_to_idx, window_size=2):
    data = []
    for sentence in corpus:
        words = sentence.lower().split()
        for i, target_word in enumerate(words):
            target_word_idx = word_to_idx[target_word]
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_word_idx = word_to_idx[words[j]]
                    data.append((target_word_idx, context_word_idx))
    return data

training_data = generate_skipgram_data(corpus, word_to_idx, window_size=2)


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        # 输入词的词嵌入矩阵（v * h）
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 输出词的词嵌入矩阵（h * v）
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # 初始化
        nn.init.xavier_uniform_(self.target_embeddings.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, target_word_idxs, context_word_idxs):
        # 获取嵌入向量
        target_embedding = self.target_embeddings(target_word_idxs)  # v * h
        context_embedding = self.linear(target_embedding)  # h * v
        
        # 计算相似度分数 (使用矩阵乘法)
        #scores = torch.matmul(target_embedding, context_embedding.T)  # v * h * h * v -> v * v
        return context_embedding


def train_skipgram_model(training_data, vocab_size, embedding_dim=50, batch_size=4, epochs=200, learning_rate=0.01):
    model = SkipGramModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        random.shuffle(training_data)
        total_loss = 0
        for i in range(0, len(training_data), batch_size):
            batch_data = training_data[i:i+batch_size]
            if len(batch_data) == 0:
                continue
            
            target_word_idxs = torch.tensor([pair[0] for pair in batch_data], dtype=torch.long)
            context_word_idxs = torch.tensor([pair[1] for pair in batch_data], dtype=torch.long)
            
            # Forward pass
            optimizer.zero_grad()
            output_scores = model(target_word_idxs, context_word_idxs)
            
            # 计算损失
            loss = loss_fn(output_scores, context_word_idxs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(training_data)}')
    
    return model

skipgram_model = train_skipgram_model(training_data, vocab_size, embedding_dim=50, batch_size=4, epochs=200)
