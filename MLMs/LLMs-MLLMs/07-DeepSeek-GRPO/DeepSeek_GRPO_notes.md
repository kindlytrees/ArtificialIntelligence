# DeepSeek GRPO notes

### DeepSeek-R1 算法简介
*   **深度求索第一代推理模型 **(DeepSeek first-generation reasoning models)
*   **DeepSeek-R1-Zero**: 通过大规模强化学习（RL）训练，无需监督微调（SFT）作为初步步骤。
    *   展现出卓越的推理能力。
    *   面临挑战：可读性差、语言混杂。
*   **DeepSeek-R1**: 在 RL 之前引入了多阶段训练和冷启动数据。
    *   提出了多阶段训练策略（`冷启动 -> RL -> SFT -> 全场景 RL`），有效兼顾准确率与可读性。
    *   性能比肩 OpenAI-o1-1217。
*   **开源**: DeepSeek-R1-Zero, DeepSeek-R1, 以及基于 Qwen 和 Llama 从 DeepSeek-R1 蒸馏出的六个稠密模型 (1.5B, 7B, 8B, 14B, 32B, 70B)。
*   **技术报告**: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*

### DeepSeek-R1 实现方法

### DeepSeek-R1-Zero: 基础模型上的强化学习
*   探索大语言模型（LLM）在没有任何监督数据的情况下发展推理能力的潜力。

### GRPO 算法 (Group Relative Policy Optimization, 分组相对策略优化)
*   **核心思想**: 放弃了 PPO 等模型中的 actor-critic 框架中的 critic 模型（其规模通常与策略模型相同）。
*   **基线估计**: 改为从分组得分中估计基线。
*   **优化方法**:
    *   从旧的策略 $\pi_{\theta_{old}}$ 对于 query $q$ 通过 rollout 生成 $G$ 个候选输出 $\{o_1, o_2, \cdots, o_G\}$
    *   通过如下目标函数来优化新策略：
    - 通过如下的优化方法来优化新的策略 $\pi_\theta$
    - 系数 $w_i$ 为重要性采样系数，$A$ 为优势函数，$\sigma$ 和 $\beta$ 为超参数      
$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1 - \varepsilon, 1 + \varepsilon \right) A_i \right) - \beta \mathcal{D}_{KL} (\pi_\theta || \pi_{ref}) \right) \right]
$$

$$
\mathcal{D}_{KL} (\pi_\theta || \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1
$$

$$
A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})}
$$
### 奖励建模 (Reward Modeling)
*   采用**基于规则的奖励系统**，没有使用基于神经网络的奖励模型，以减小复杂度（在大规模强化学习进程中避免出现奖励模型失效（reward hacking）的问题）。
*   **两种主要奖励类型**:
  1.  **准确性奖励 **(Accuracy rewards):
      *   准确性奖励模型评估响应是否正确。
      *   对于具有确定结果的数学问题，要求模型在指定格式内提供最终答案（例如，在方框内），以便进行可靠的基于规则的正确性验证。
      *   对于 LeetCode 问题，可以使用编译器根据预定义的测试用例生成反馈。
  2.  **格式奖励 **(Format rewards):
      *   强制模型将其思考过程放在 `<think>` 和 `</think>` 标签之间。

### 训练模板 (Training Template)
*   一个简单的模板，用于指导基础模型遵循指定的指令。
*   我们有意将约束限制在此结构化格式内，避免任何内容特定的偏差——例如，强制要求反思性推理或推广特定的问题解决策略——以确保我们能够准确地观察模型在 RL 过程中的自然进展。

https://www.kindlyrobot.cn/2025/03/25/llm%e6%8a%80%e6%9c%af%e7%b3%bb%e5%88%97%e4%b9%8b-deepseek-v1-v2-v3%e6%a8%a1%e5%9e%8b%e7%bb%93%e6%9e%84%e7%ae%80%e4%bb%8b/