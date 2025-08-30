# RTX

- in-distribution 的能力主要体现在单一数据集本身的任务能力提升上，RT-1的模型在基于混合数据训练后的RT-1-X模型在体量大的数据集上训练后in-distribution能力下降，说明了RT-1-X出现了欠拟合，模型的表达能力有限
- out-distribution主要体现在emergent skills的评估上

## 参考资料
- rt-1:https://console.cloud.google.com/storage/browser/gresearch/rt-1-data-release
- https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb?pli=1&authuser=1#scrollTo=YrD4_8P9JxBw
- https://huggingface.co/datasets/jxu124/OpenX-Embodiment
- https://say-can.github.io/