# Stable Diffusion LoRA

$$
\begin{aligned}
& W^{\prime}=W+\Delta W=W+\alpha \cdot A \cdot B \\
& \Delta W=A B^T, \text { where } A, \in R^{d \times r}, B \in R^{d \times r}, r \ll d
\end{aligned}
$$


diffusers库安装
pip install git+https://github.com/huggingface/diffusers
用于风格迁移的国画数据集： https://aistudio.baidu.com/datasetdetail/107231

具体的案例可以咨询大模型，比如向大模型提问：
 "请提供一个在google colab里训练text_to_image_lora的具体方法。
 或"请提供一个在baidu ai studio里训练text_to_image_lora的具体方法。"