# GRPO DeepSeek R1(-Zero)

## 算法要点
- DeepSeek-R1-Zero直接训练强化学习模型
- DeepSeek-R1
    - code start，用few show prompting的方式在DeepSeek-R1-Zero的大模型生成的样本辅以人工核验后的数据集上fine-tune the DeepSeek-V3-Base
    - 在此基础上采用DeepSeek-R1-Zero的训练方法训练一个推理能力较强的模型（为了数据策展的时候，第三步的时候使用）
    - 通过dataset collection and curation，融合推理数据（通过上一个步骤生成且验证为正确的数据）和通用的问答数据集进行finetune DeepSeek-V3-Base（2个epoches）（SFT）
    - 再次用强化学习训练全场景数据
        - 推理数据，用DeepSeek-R1-Zero采用的基于规则的rewards
        - 回答数据（基于人类反馈的），采用reward models奖励模型获得rewards