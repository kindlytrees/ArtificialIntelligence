# RLHF

## 知识要点
- 即时奖励的计算
    - Reward modeling给回答打分，作为整个episode的即时奖励的最后
    - KL相关的计算，作为per token level的中间时间序列的即时奖励的计算
- td_target和之前的actor-critic的思路一致，而GRPO(Critic-Free)算法中没有critic网络，其优势函数主要基于组内的相对得分优势得出
- SFT,Reward Modeling,RLHF三个模型
    - SFT作为参考模型(referece policy)
    - Reward Modeling给回答打分
    - RLHF中策略模型有三个（active policy为训练策略网络，old policy为rollout policy，为active policy的一个近期快照，referece policy为sft模型，和rollout policy的网络计算kl来实现token level的reward计算，rollout policy也同时通过GAE实现优势函数的计算，以及提供old policy的logprobs的计算
    - 代码参考trl库的ppo_trainer.py中的实现