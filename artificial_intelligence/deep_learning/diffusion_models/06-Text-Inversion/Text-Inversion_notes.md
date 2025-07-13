# Text Inversion notes

## 相关的文字说明

辅助gemini回答：

diffusion models里的text inversion用新的embeddings代表新的视觉概念，请问性能的embedding的token如何设计，是不在当前词汇表里的新词汇吗，这样是不是要修改embedding的参数，还是说和dreambooth类似找出在预训练模型里没有用到但在词汇表里的token来进行指代是不是更加方便点？

Textual Inversion的核心思想是：“我们不改变模型的任何部分（U-Net或Text Encoder），只在文本编码器的词嵌入层（word embedding layer）中，为我们的新概念创造一个全新的、自定义的词向量（embedding）。”

我们设计的不是一个词（token），而是一个占位符（placeholder string）。

修改Embedding参数: 是的，这正是Textual Inversion要做的事情。它会：
在文本编码器的Embedding层（一个巨大的查找表，Token ID -> Vector）中，增加一行。
为这个新行分配一个新的、特殊的Token ID。
将你的占位符（如<sks-dog>）与这个新的Token ID关联起来。
在训练开始时，用随机值初始化这一行对应的词向量。
在训练过程中，只更新这个新添加的词向量，模型的其他所有部分（包括Embedding层的其他所有行）都保持冻结。


这个想法非常聪明，它本质上是在问：“我们能不能用Textual Inversion的方法，但去优化一个已有的稀有token的embedding，而不是创建一个新的？”
理论上是可行的，但实践中不如创建新词条效果好，原因如下：
潜在的语义冲突: 即使是像sks这样的稀有token，它在CLIP的巨大训练数据中也可能带有微弱的、我们未知的“语义偏向”。优化这个已有的向量，可能会受到这种未知偏向的干扰。而从零开始创建一个全新的、随机初始化的向量，就像在一张完全干净的白纸上作画，没有历史包袱。
实现与分享的便利性: 创建一个全新的占位符（如<my-super-dog>）非常直观。用户不需要去“寻找”一个神秘的稀有词。分享时也更容易理解，你只需要告诉别人：“加载这个embedding文件，然后用<my-super-dog>这个词就能生成我的狗了。”这比告诉他们“用sks这个词”要清晰得多。
避免Token冲突: 如果两个人都碰巧选择了sks这个稀有词来训练他们各自的概念（比如一个人训练了狗，另一个人训练了背包），那么他们的embedding文件就会产生冲突。而使用自定义的、独特的占位符（如<johns-dog>和<marys-backpack>）则完全避免了这个问题。

## 为新概念个特殊的名字主要是为了解析方便，脚本先处理找出这个特殊的占位符并赋予新创建的tokenid和其对应，其他的部分再沿用分词器处理的结果，如何进行细节上的处理呢，比如占位符再分词器得到的token序列中的位置等信息？

- 辅助gemini回答：

您好！您这个问题问得非常深入，直击了Textual Inversion在工程实现上的核心细节。您描述的流程完全正确，现在我们来把这个“黑盒”打开，看看内部是如何处理这些细节的。

整个流程可以分为两步：**1. 训练前的准备（扩展Tokenizer和模型）** 和 **2. 训练中的动态处理（修改输入）**。

### 第一步：训练前的准备 (Setup Phase)

在训练脚本开始时，会执行一次性的设置操作。

**1. 识别和添加占位符到Tokenizer**

*   **脚本逻辑**: 训练脚本会接收你指定的占位符字符串，比如 `--placeholder_token="<sks-dog>"`。
*   **添加新Token**: 它会调用Tokenizer的 `add_tokens` 方法。这个方法就是专门用来向现有的Tokenizer词汇表中添加新词的。
    ```python
    # 伪代码示例
    placeholder_token = "<sks-dog>"
    num_added_tokens = tokenizer.add_tokens(placeholder_token)

    if num_added_tokens == 0:
        # 说明这个词已经存在了，可能不是个好选择
        raise ValueError("The placeholder token is already in the tokenizer's vocabulary.")

    # 获取新添加的token的ID
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    ```
    执行后，Tokenizer内部的词汇表就从 `N` 个词变成了 `N+1` 个词。这个新的词 `<sks-dog>` 就有了一个独一无二的、新分配的`token_id`（例如，`50257`，如果原始词汇表大小是50257）。

**2. 扩展文本编码器的Embedding层**

*   **目标**: Tokenizer现在认识新词了，但文本编码器（如CLIP的TextEncoder）的Embedding层还不知道该给这个新ID分配哪个向量。它的Embedding矩阵（一个巨大的查找表）的大小还是 `N x D`（N个词，每个词D维向量）。
*   **调整大小**: 脚本会调用模型的 `resize_token_embeddings` 方法。
    ```python
    # 伪代码示例
    text_encoder.resize_token_embeddings(len(tokenizer))
    ```
    这个方法会做两件事：
    a. 检查Tokenizer的新大小 (`N+1`) 和Embedding层当前的大小 (`N`)。
    b. 发现不匹配后，它会自动将Embedding层的矩阵扩展为 `(N+1) x D`，即在末尾**增加一行**。
*   **初始化新向量**: 新增的这一行向量会被**随机初始化**（通常是小的随机正态分布值）。

**3. 冻结模型参数，解冻新向量**

*   **冻结所有**: 脚本会遍历模型的所有参数（U-Net和Text Encoder），并将它们的 `requires_grad` 属性设置为 `False`。
    ```python
    # 伪代码示例
    for param in text_encoder.parameters():
        param.requires_grad = False

    for param in unet.parameters():
        param.requires_grad = False
    ```
*   **解冻目标**: 唯独我们新添加的那个词向量需要被训练。脚本会通过新`token_id`精确定位到Embedding矩阵中的那一行，并将其 `requires_grad` 设置为 `True`。
    ```python
    # 伪代码示例
    # text_encoder.get_input_embeddings() 获取Embedding层
    embedding_layer = text_encoder.get_input_embeddings()
    # 通过新ID定位到那一行
    embedding_layer.weight.data[placeholder_token_id].requires_grad = True
    ```

至此，准备工作完成。模型现在已经准备好学习这个新词向量了。

---

### 第二步：训练中的动态处理 (Training Loop Phase)

现在我们来看在每个训练步骤中，当一个包含占位符的提示（如 `a photo of <sks-dog>`）传来时，是如何被处理的。

**1. 标准的分词流程**

*   你的提示字符串被送入我们**已经扩展过的Tokenizer**。
    ```python
    # 伪代码示例
    prompt = "a photo of <sks-dog> in a park"
    input_ids = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).input_ids
    ```
*   **Tokenizer的行为**:
    *   它会像处理普通句子一样处理这个提示。
    *   遇到 `a`, `photo`, `of`, `in`, `a`, `park` 这些词，它会转换成它们已有的Token ID。
    *   当它遇到 `<sks-dog>` 时，因为它在第一步已经被添加到了词汇表中，Tokenizer会愉快地将其识别为一个**单独的、完整的token**，并转换成我们新分配的那个 `placeholder_token_id`。
    *   最后，它会根据需要进行填充（padding）和截断（truncation），生成一个固定长度的Token ID序列。

**2. 位置等信息的处理**

您的问题“如何处理位置等信息”非常关键。答案是：**我们不需要做任何特殊的处理，标准流程已经为我们搞定了一切。**

*   **Token序列中的位置**:
    `input_ids` 张量本身就包含了位置信息。例如，`a photo of <sks-dog> in a park` 转换后可能看起来像这样（ID是示意）：
    `[49406, 320, 1125, 539, 50257, 525, 269, 3968, 49407, ...]`
    这里的 `50257` 就是 `<sks-dog>` 的ID，它自然地出现在了序列的第5个位置。

*   **传递给文本编码器**:
    这个`input_ids`张量被送入文本编码器。编码器会：
    a. **查找向量**: 通过`input_ids`中的每个ID，去Embedding层查找对应的词向量。当查到`50257`时，它就会取出我们那个正在被训练的新向量。
    b. **加入位置编码**: 文本编码器（如CLIP的Transformer结构）会自动将**位置编码（Positional Encoding）**与查找到的词向量相加。这使得模型能够理解每个词在句子中的顺序和位置。`50257`这个向量也会被加上代表“第5个位置”的位置编码。
    c. **通过注意力层**: 包含了位置信息的词向量序列会被送入多层Transformer（注意力机制），进行深度的上下文理解。

**3. 反向传播**

*   当计算出最终的损失后，梯度会反向传播。
*   由于我们只解冻了`placeholder_token_id`对应的那个词向量，所以梯度只会流向并更新Embedding矩阵中那一行的值。其他所有参数都因为`requires_grad=False`而不会被更新。

### 总结

细节处理流程如下：

1.  **初始化**:
    *   用`tokenizer.add_tokens()`将占位符字符串注册为一个新词，获得新`token_id`。
    *   用`model.resize_token_embeddings()`扩展Embedding层，为新`token_id`增加一行随机初始化的向量。
    *   冻结模型所有参数，唯独解冻这个新向量。

2.  **训练循环**:
    *   **无需特殊解析**: 将包含占位符的完整提示直接送入扩展后的Tokenizer。
    *   **自动处理**: Tokenizer会将占位符识别为单个token，并输出包含其`token_id`的序列。
    *   **位置信息天生就有**: Token在序列中的位置就是它的位置信息。文本编码器标准的**位置编码机制**会自动处理它。
    *   **梯度只流向目标**: 反向传播时，只有那个被解冻的新向量会被更新。

整个过程设计得非常优雅，它在不破坏原有模型结构和流程的前提下，通过对输入管道的微小扩展，实现了学习全新概念的能力。

## References

https://github.com/huggingface/diffusers
examples目录下三个子目录应用
dreambooth
text_to_image
textual_inversion
https://zhuanlan.zhihu.com/p/627133491 LoRA原理和实践
https://mp.weixin.qq.com/s/NE3Gkr64G3XADVdujtzRXw dreambooth原理和实践
https://huggingface.co/models
https://civitai.com/
https://dreambooth.github.io/ 
https://textual-inversion.github.io/
