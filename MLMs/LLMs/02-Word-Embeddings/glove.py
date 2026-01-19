import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import itertools

# 小语料库
corpus = [
    "I like deep learning",
    "I like NLP",
    "I enjoy flying",
    "deep learning is amazing",
    "NLP is a field of AI",
    "I like AI"
]

# 构建词汇表
words = list(itertools.chain(*[sentence.lower().split() for sentence in corpus]))
vocab = Counter(words)
word_to_idx = {word: i for i, word in enumerate(vocab.keys())}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(vocab)


def build_cooccurrence_matrix(corpus, word_to_idx, vocab_size, window_size=2):
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
    
    for sentence in corpus:
        words = sentence.lower().split()
        for i, word in enumerate(words):
            word_idx = word_to_idx[word]
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    distance = abs(i - j)
                
                    # 2. 计算权重 (weight)，距离越远，权重越小
                    weight = 1.0 / distance
                    
                    # 3. 将权重累加到共现矩阵中
                    cooccurrence_matrix[word_idx][word_to_idx[words[j]]] += weight
                    #cooccurrence_matrix[word_idx][word_to_idx[words[j]]] += 1
    return cooccurrence_matrix

cooccurrence_matrix = build_cooccurrence_matrix(corpus, word_to_idx, vocab_size)

class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVe, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_bias = nn.Embedding(vocab_size, 1)
        self.context_bias = nn.Embedding(vocab_size, 1)
        
        # 初始化
        nn.init.xavier_uniform_(self.word_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
        nn.init.zeros_(self.word_bias.weight)
        nn.init.zeros_(self.context_bias.weight)
    
    def forward(self, word_idx, context_idx, cooc_value):
        word_embed = self.word_embeddings(word_idx)
        context_embed = self.context_embeddings(context_idx)
        word_bias = self.word_bias(word_idx).squeeze()
        context_bias = self.context_bias(context_idx).squeeze()
        
        dot_product = torch.sum(word_embed * context_embed, dim=1)
        loss = (dot_product + word_bias + context_bias - torch.log(cooc_value + 1e-10))**2
        return torch.mean(loss)

def train_glove(cooccurrence_matrix, word_to_idx, embedding_dim=50, epochs=100, learning_rate=0.05, x_max=100, alpha=0.75):
    vocab_size = len(word_to_idx)
    model = GloVe(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(vocab_size):
            for j in range(vocab_size):
                cooc_value = cooccurrence_matrix[i][j]
                if cooc_value == 0:
                    continue
                
                weight = (cooc_value / x_max) ** alpha if cooc_value < x_max else 1
                word_idx = torch.tensor([i], dtype=torch.long)
                context_idx = torch.tensor([j], dtype=torch.long)
                cooc_value_tensor = torch.tensor([cooc_value], dtype=torch.float)
                
                optimizer.zero_grad()
                loss = weight * model(word_idx, context_idx, cooc_value_tensor)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        print(f'Epoch: {epoch + 1}, Loss: {total_loss / np.sum(cooccurrence_matrix > 0)}')
    
    return model

glove_model = train_glove(cooccurrence_matrix, word_to_idx, embedding_dim=50, epochs=100)


def get_word_vectors(model):
    return model.word_embeddings.weight.data + model.context_embeddings.weight.data

word_vectors = get_word_vectors(glove_model)

def find_similar(word, word_vectors, word_to_idx, idx_to_word, top_n=5):
    word_idx = word_to_idx[word]
    word_vector = word_vectors[word_idx]
    
    similarities = torch.matmul(word_vectors, word_vector)
    most_similar = torch.argsort(similarities, descending=True)[:top_n + 1]
    
    return [idx_to_word[idx.item()] for idx in most_similar if idx.item() != word_idx]

print(find_similar("ai", word_vectors, word_to_idx, idx_to_word))