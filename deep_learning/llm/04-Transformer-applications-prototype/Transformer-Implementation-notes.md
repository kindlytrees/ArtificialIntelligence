# Transformer Implementation

## 实验一：Transformer编码器架构实现文本的分类问题

实现tweets上的文本内容分类，数据集在本实验根目录下的dataset子目录  
实验的notebook为：`transformers-encoder-from-scratch-dl.ipynb`

## 实验二：Transformer解码器架构实现的nanoGPT
实验的脚本为：`transformers_decoders_from_scratch.py`

## 实验三： 编码器解码器架构的Transformer实现语言翻译任务

实验的notebook为：`The_Annotated_Attention_is_All_You_Need.ipynb`
问题： 在做loss = loss_backprop(model.generator, criterion, out[:,:-1,:], trg[:, 1:], batch['valid_token_count'])这样的翻译任务的loss计算时，一般解码器的输入中开始加上cls，结束加上sep，然后输出的真值的第一个会去掉cls token，最后一个sep对应的输出不参与loss的计算，这种基于自回归的真值设置合理吗？请给与分析和补充。

回答（gemini 2.5 pro辅助）：
是最常见的实现：  
trg: 完整的带 `<start>` 和 `<end>` 的目标序列。  
[`<start>`, w1, w2, ..., wn, `<end>`]  
Decoder Input: trg[:, :-1] -> [`<start>`, w1, w2, ..., wn]  
Model Output out: model(decoder_input) -> 得到 out，其序列长度与 decoder_input 相同。  
Target for Loss: trg[:, 1:] -> [w1, w2, ..., wn, `<end>`]  
Loss Calculation: criterion(out, target_for_loss)

在loss的计算中采用了基于有效长度的mask

## References

pytorch实现的MultiHeadAttention:pytorch\torch\nn\modules\activation.py文件中的MultiheadAttention类
https://github.com/facebookresearch/fairseq
https://github.com/joeynmt/joeynmt/blob/main/joeynmt/transformer_layers.py
https://github.com/joeynmt/joeynmt.git 支持训练，测试和翻译的transformer autoencoder的实现
https://colab.research.google.com/drive/1mHUprXsx5VpNiGrrdg0yYLtOP4Y5jRiB?pli=1&authuser=1#scrollTo=zBaKFoTbvC5D
