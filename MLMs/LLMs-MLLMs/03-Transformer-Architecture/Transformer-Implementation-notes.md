# Transformer Implementation

## 实验： 编码器解码器架构的Transformer实现语言翻译任务

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

## 问题1：在批量样本的transformer架构中，如何统一输入序列的长度，以及统一进行self attention的计算？

回答：未达到最大长度的，用padding补齐，并计算对应的mask向量，用于attention matrix的时候进行对相关无效部分进行mask免除相关计算
因此，编码器解码器架构中src有mask，target也有mask，同时target还有causual mask，最终解码器的mask即为target_mask和causual mask的交集计算。

## 问题2：在编码器解码器架构中，解码器中的attention计算的mask如何定义，其机制是什么？？

回答(gemini辅助回答)：每一个docoder block中，下面的causual block因为是输入的单项序列相关，因此mask为目标的输入的mask（padding）和因果mask的交集，在上面的cross attention的计算机制中，采用的是encoder输出的src mask，因为z的输出的计算是完整的序列，因此每一层都要进行mask。

编码器的输出 z 是固定的：编码器一次性处理完整个源句子，生成一个固定的输出张量z。这个z包含了源句子所有token（包括填充token）的表示。
z被重复使用：这个z会作为**每一个解码器模块（Decoder Block）**中Cross-Attention层的Key和Value。
一致性要求：因此，为了保证在解码的每一步、每一层，模型都不会错误地从源句子的填充位获取信息，这个源句子的Padding Mask (src_mask) 必须在每一个解码器模块的Cross-Attention中被应用。

同理解码器block下一层的输出z是整个序列的结果，因此在当前层还要做padding mask和causual mask的交集计算。

## References

pytorch实现的MultiHeadAttention:pytorch\torch\nn\modules\activation.py文件中的MultiheadAttention类
https://github.com/facebookresearch/fairseq
https://github.com/joeynmt/joeynmt/blob/main/joeynmt/transformer_layers.py
https://github.com/joeynmt/joeynmt.git 支持训练，测试和翻译的transformer autoencoder的实现
https://colab.research.google.com/drive/1mHUprXsx5VpNiGrrdg0yYLtOP4Y5jRiB?pli=1&authuser=1#scrollTo=zBaKFoTbvC5D