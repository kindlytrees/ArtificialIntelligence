# GRPO DeepSeek R1(-Zero)

## 算法要点
- DeepSeek-R1-Zero直接训练强化学习模型,主要强调在推理能上的表现，reward为基于规则的计算
- DeepSeek-R1训练的几个阶段
    - code start，用few shot prompting的方式在DeepSeek-R1-Zero的大模型生成的样本(in context learning)，辅以人工核验后的数据集上fine-tune the DeepSeek-V3-Base
    - 在此基础上采用DeepSeek-R1-Zero的训练方法训练一个推理能力较强的模型（为了数据策展的时候使用，第三步的时候使用）
    - 通过dataset collection and curation，融合推理数据（通过上一个步骤生成且验证为正确的数据）和通用的问答数据集进行finetune DeepSeek-V3-Base（2个epoches）（SFT）
    - 再次用强化学习训练全场景数据
        - 推理数据，用DeepSeek-R1-Zero采用的基于规则的rewards
        - 回答数据（基于人类反馈的），采用reward models奖励模型获得rewards

## 问题

- 问题1：如下这个loss，为什么这个定义，和TRPO和PPO中forward kl divergence好像不同？

$$
\mathbb{D}{K L}\left(\pi\theta | \pi_{r e f}\right)=\frac{\pi_{r e f}\left(o_i \mid q\right)}{\pi_\theta\left(o_i \mid q\right)}-\log \frac{\pi_{r e f}\left(o_i \mid q\right)}{\pi_\theta\left(o_i \mid q\right)}-1
$$

- 问题2：DeepSeek-R1-Zero的ref网络是什么预训练模型（也是基于大规模数据集训练的预训练模型对吗)，而且其训练方法其和DeepSeek-R1采用的强化学习算法是相同的GRPO算法对吗？

- 问题3：在rlhf的ppo以及后续的GRPO等算法中，kl散度的计算主要是对照ref网络生成的token的概率，但计算时采用简化版本，可以看成时对应token和非该token的二元概率分布的差异，以为每一个位置的完整kl散度计算则需要基于词汇表空间的概率结果，这样复杂度太高，因此时直接基于对应token位置的kl散度去做计算，这样是不是也就在kl散度损失这一块只会约束对应token处的概率对齐约束？