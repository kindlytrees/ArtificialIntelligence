# Bert Notes

## 实验一：从零开始实现BERT

实验代码文件: Question_answeringBERT_from_Scratch.ipynb

原始文件参考链接：`https://github.com/ChanCheeKean/DataScience/blob/main/13 - NLP/C04 - BERT (Pytorch Scratch).ipynb`

添加了对于数据集的注释和说明

数据集：电影对话语料库

http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip

1. movie_lines.txt - 所有电影台词
这个文件包含了数据集中每一句台词的详细信息。每一行代表一句台词，并由特定的分隔符 +++$+++ 分隔成5个字段。
格式：
lineID +++$+++ characterID +++$+++ movieID +++$+++ character_name +++$+++ text_of_the_line

2. movie_conversations.txt - 对话的结构
这个文件定义了哪些台词构成了一次完整的对话。每一行代表一次对话，同样由 +++$+++ 分隔成4个字段。
格式：
characterID_1 +++$+++ characterID_2 +++$+++ movieID +++$+++ list_of_lineIDs

对于列表 ['L194', 'L195', 'L196', 'L197']，我们可以构建出以下对话对：
(L194, L195)
(L195, L196)
(L196, L197)

3. movie_titles_metadata.txt - 电影元数据
包含电影的详细信息。
格式：
movieID +++$+++ movie_title +++$+++ movie_year +++$+++ IMDb_rating +++$+++ IMDb_votes +++$+++ list_of_genres

4. movie_characters_metadata.txt - 角色元数据
包含角色的详细信息。
格式：
characterID +++$+++ character_name +++$+++ movieID +++$+++ movie_title +++$+++ gender +++$+++ position_in_credits

embedding包含三部分：token embedding，segment embedding，和positional embedding
masked language model的数据集生成过程在代码中有具体的体现

## 实验二：Transformer编码器架构实现文本的分类问题

实现tweets上的文本内容分类，数据集在本实验根目录下的dataset子目录  
实验的notebook为：`transformers-encoder-from-scratch-dl.ipynb`


## 实验三：BERT实现文本分类

实验代码文件: https://www.kaggle.com/code/ludovicocuoghi/detecting-bullying-tweets-pytorch-lstm-bert

```
self.bert = BertModel.from_pretrained('bert-base-uncased')
outputs = self.bert(input_ids=input_ids,
                    attention_mask=attention_mask)

# Extract the last hidden state of the `[CLS]` token from the BERT output (useful for classification tasks)
# outputs[0] 或 outputs.last_hidden_state,这是BERT模型最后一层输出的所有token的隐层状态向量（hidden state vectors）。
last_hidden_state_cls = outputs[0][:, 0, :]

outputs[1] 或 outputs.pooler_output,它本质上是 last_hidden_state 中第一个token（即 [CLS] token）的表示，然后再经过一个线性层和一个Tanh激活函数的处理。
```

## 实验四：基于SQuAD数据集采用BERT实现抽取式问答任务的模型训练

实验代码文件: Question_answering_BERT_finetuned_Squad.ipynb

qa这种场景下，如果question+context太长，经过分词器处理后会形成多个序列，多个序列同时对模型进行推理，如果一个序列中只有start或end，其损失函数是不进行回传的对吗，具体在预测的时候如何进行推理呢

tokenizer(text_a, text_b, ...) 这种调用方式是专门为需要输入一对文本的任务设计的，如问答（Question-Context）或自然语言推断（Premise-Hypothesis）。
Tokenizer会自动将它们格式化成BERT等模型需要的格式，通常是：
[CLS] question_tokens [SEP] context_tokens [SEP]

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
hugging Face Transformers 库中的 AutoModelForQuestionAnswering。这是一个非常强大且方便的类，专门用于处理**抽取式问答（Extractive Question Answering）**任

1. 什么是抽取式问答 (Extractive QA)？
首先，我们要明确 AutoModelForQuestionAnswering 所解决的问题类型。
输入 (Input): 一个问题 (Question) 和一个包含答案的上下文段落 (Context)。
任务 (Task): 模型需要从给定的上下文中，找出答案所在的文本片段（span）。
输出 (Output): 答案的起始位置 (start position) 和结束位置 (end position) 的索引。
例子：
Context: "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower."
Question: "Who designed the Eiffel Tower?"
Expected Answer Span: "Gustave Eiffel"
Model's Goal: 预测出 "Gustave Eiffel" 在原文中的开始和结束 token 的索引。
这种任务不要求模型“创造”答案，而是要求模型“定位”答案，因此被称为“抽取式”。


3. 模型内部结构
一个典型的 ...ForQuestionAnswering 模型由两部分构成：
a. 基础语言模型 (Base Language Model)
这部分可以是 BERT, RoBERTa, DistilBERT, ALBERT 等任何一个强大的 Transformer 编码器模型。
它的作用是接收拼接后的 [CLS] question [SEP] context [SEP] 输入，并为每一个 token 生成一个富含上下文信息的隐层状态向量 (hidden state vector)。
输出的形状为 (batch_size, sequence_length, hidden_size)。
b. 问答头 (QA Head)
这是嫁接在基础模型之上的一个简单的全连接层（Linear Layer）。
输入: 基础模型输出的所有 token 的隐层状态（形状 (batch_size, sequence_length, hidden_size)）。
输出: 两个 logits 向量。
Start Logits: 形状为 (batch_size, sequence_length)。向量中的第 i 个值，代表了输入序列的第 i 个 token 是答案起始位置的概率得分。
End Logits: 形状为 (batch_size, sequence_length)。向量中的第 i 个值，代表了输入序列的第 i 个 token 是答案结束位置的概率得分。
权重: 这个全连接层的权重 W 的形状是 (hidden_size, 2)。它将每个 token 的 hidden_size 维向量映射成一个2维向量（一个用于start, 一个用于end）。