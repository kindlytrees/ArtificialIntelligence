# RLHF InstructGPT notes

## 问题1：请分析一下在rlhf的实现中，基于每个token的kl散度的计算方式，是基于对应的response的index位置算当前策略网络和参考网络的分布差异，还是基于词汇表的所有位置算分布差异， 从trl的实现看，其采用了更加简洁的计算方式，请对其实现做更加详细的说明。

## 问题2：因此一个生成reponse的过程中的序列中的每个时刻的奖励即为对应位置token生成的kl惩罚的负数和最后位置的reward model给与的评分，对于一个query，会生成多个response对吗？采用unwrap model去生成不同的response，然后采用reward模型打分，数据生成和打分是不是也比较消耗资源，可以考虑离线的策略吗，参考网络即为sft模型的话，在整个训练过程参数不变是吗？经典的ppo中的异策略网络，生成数据的网络参数也会定期更新，请对于以上分析给与补充说明。

## 问题3：rlhf中优势函数用到了non score reward去计算，但该部分中只会以权重和值大小反馈到策略梯度的计算中，而不会有其他的影响，请对上述判断做更加详细的说明。

gemini辅助回答： 
您的判断非常精准，点出了 RLHF 中一个核心且容易被误解的机制。我们可以将您的判断总结为：

**在技术实现上，KL 惩罚项（您所说的 non-score reward）确实是以一个数值的形式，直接加到奖励模型的打分上，然后通过优势函数影响策略梯度的计算。但它的“影响”远不止于一个简单的数值，它从根本上塑造和约束了整个优化过程。**

下面我们来详细地拆解和说明这一点。

---

### 1. 首先，明确“Non-Score Reward”是什么

在 RLHF 的 PPO（Proximal Policy Optimization）阶段，模型在每一步生成的总奖励（Total Reward）由两部分构成：

**Total Reward = `r_θ` - `β * KL`**

*   **`r_θ` (Score Reward)**: 这是来自**奖励模型 (Reward Model, RM)** 的打分。它代表了人类的偏好，是“**我们想要什么**”的信号。这个分数是动态的，模型生成得越好，分数就越高。

*   **- `β * KL` (Non-Score Reward / KL Penalty)**: 这部分是关键。
    *   **KL** 是指 **KL 散度 (Kullback-Leibler Divergence)**，它衡量了当前正在优化的策略模型 `π_φ` 的输出概率分布与初始的、经过监督微调（SFT）的模型 `π_SFT` 的输出概率分布之间的差异。
    *   **`β`** 是一个超参数，控制着这个惩罚项的权重。
    *   这个惩罚项代表了“**我们不希望模型偏离初始状态太远**”的约束。它是一个**正则化项**，其目的是防止模型在追求高奖励分数的过程中“走火入魔”。

---

### 2. 对您判断的详细说明：“只会以权重和值大小反馈到策略梯度的计算中”

您的这个描述在**计算流程**上是完全正确的。我们来看看它是如何发生的：

1.  **生成与评估**:
    *   当前策略模型 `π_φ` 根据一个 prompt 生成一个回答。
    *   在生成每个 token 时，计算 `π_φ` 和 `π_SFT` 在该步的 KL 散度，并乘以 `-β`。这构成了每一步的 KL 惩罚。
    *   当整个回答生成完毕后，奖励模型 `r_θ` 对这个 (prompt, answer) 对进行打分，得到一个总的 RM 分数。

2.  **计算总奖励**:
    *   将 RM 分数和累积的 KL 惩罚相加，得到每一步的“总奖励” `r_t`。
    *   `r_t = r_θ(t) - β * KL(t)` (这里为了简化，假设奖励是每一步都给出的)。

3.  **计算优势函数**:
    *   PPO 算法使用这些 `r_t` 来计算优势函数 `A(s_t, a_t)`。优势函数衡量了在状态 `s_t` 下采取动作 `a_t` 比平均水平好多少。
    *   `A(s_t, a_t)` 的计算（例如使用 GAE）完全依赖于 `r_t` 的数值。此时，`r_t` 只是一个数字，算法本身并不知道这个数字是由 RM 分数和 KL 惩罚两部分组成的。

4.  **计算策略梯度**:
    *   PPO 的目标函数（或策略梯度）大致正比于 `优势函数 * log(策略概率)`，即 `A(s_t, a_t) * log π_φ(a_t | s_t)`。
    *   优势函数 `A` 的大小直接决定了策略更新的方向和幅度。一个大的正优势会鼓励模型增加 `π_φ(a_t | s_t)` 的概率，反之则会抑制。

**结论**：从这个流程可以看出，KL 惩罚项最终被“压缩”成一个数值，并与其他奖励数值合并。从算法的视角看，它与 RM 打分在**计算机制**上没有区别，都是影响优势函数大小的因子。

---

### 3. 补充说明：“而不会有其他的影响”——为什么这个判断需要扩展？

虽然计算机制上如此，但 KL 惩罚项的**实际影响**是深远且多方面的。说它“不会有其他影响”忽略了它作为**正则化器**的根本作用。它的影响体现在以下几个方面：

#### 影响一：防止灾难性遗忘 (Catastrophic Forgetting) 和保持语言质量

*   SFT 模型包含了大量的世界知识、语法结构、语言风格等基础能力。
*   奖励模型 `r_θ` 是一个不完美的、有漏洞的代理。如果单纯追求 RM 的高分，模型可能会发现一些“奖励黑客”（Reward Hacking）的捷径，比如生成一堆重复的、高赞的关键词，或者输出一些语法不通但能骗过奖励模型的句子。
*   KL 惩罚项就像一根**“锚”**，将策略模型 `π_φ` 牢牢地拴在 SFT 模型附近。它告诉模型：“你可以为了更高的奖励分数去探索新的回答方式，但前提是你的语言风格和基础能力不能离你最初学会的样子太远。”
*   **结果**: 模型在学习人类偏好的同时，不会忘记如何说“人话”。

#### 影响二：稳定训练过程，构建信任域

*   RL 的探索空间是巨大的。如果没有约束，策略模型可能会在一次更新中跳到一个非常陌生的区域。在这个区域，奖励模型的表现可能是不可预测和不稳定的，导致训练过程剧烈震荡甚至崩溃。
*   KL 惩罚实际上为 PPO 算法构建了一个隐式的**信任域 (Trust Region)**。它确保了策略的每次更新都是小范围的、可控的，使得训练更加稳定。

#### 影响三：控制对齐的强度和风格

*   超参数 `β` 成为了一个非常重要的**调节旋钮**。
    *   **高 `β` 值**: 惩罚很强。模型会非常保守，紧密地跟随 SFT 模型的风格，对齐效果可能不明显，但输出质量稳定。
    *   **低 `β` 值**: 惩罚很弱。模型会更激进地去迎合奖励模型，对齐效果可能更强，但也更容易产生上述的“奖励黑客”问题和语言质量下降。
*   通过调整 `β`，研究人员可以在“对齐程度”和“基础语言能力”之间做出权衡。

### 最终总结

您的判断是**对了一半，但也是最关键的一半**。

*   **对在哪里**: 您精确地指出了 KL 惩罚在**计算层面**的作用方式——它被量化为一个数值，与 RM 分数合并，通过影响优势函数来参与标准的策略梯度更新。
*   **需要补充在哪里**: 它“没有其他影响”的说法低估了其**功能层面**的重要性。这个看似简单的数值惩罚，是 RLHF 框架的**稳定器、质量保证器和安全阀**。它确保了模型在“学好”的同时，不会“学坏”，是让整个对齐过程得以成功实现的关键设计。

可以这样比喻：在赛车比赛中，引擎的马力（RM 分数）决定了你能跑多快，但刹车和转向系统（KL 惩罚）决定了你是否能安全地跑完整个赛道。在计算中，刹车和加速可能都只是对轮子施加的一个力，但它们的功能和影响是截然不同的。

## 问题3： KL Penalty的的方向性说明
注释： kl penalty以Dkl(pi(ref)||pi(\theta)定义还是以Dkl(pi(\theta)||pi(ref)进行定义，具体计算时直接给出logprob的差，因此先后应该还是有影响的。请给与补充说明。

gemini 辅助回答： 

这个问题非常关键，您指出了KL散度非对称性（asymmetry）在实际应用中的重要影响。虽然在TRL等库的实现中，最终计算形式是简单的 `logprob_pi - logprob_ref`，但其背后的理论根源以及选择哪个方向的KL散度，确实是有区别和讲究的。

---

### 1. KL散度的两个方向及其性质

让我们先回顾一下两个方向KL散度的定义和直观含义。假设我们有两个概率分布 P 和 Q。

#### **方向一：`D_KL(P || Q)` (Forward KL Divergence)**

$$
D_{KL}(P || Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)
$$

*   **直观含义**：这个方向的KL散度试图确保 **“当 P(x) 的概率很高时，Q(x) 的概率也必须很高”**。如果有一个事件 `x`，在P中很常见（`P(x)`大），但在Q中却很罕见（`Q(x)`小），那么 `log(P/Q)` 会是一个很大的正数，导致KL散度爆炸。反之，如果Q覆盖了P没有覆盖的地方，惩罚则较小。
*   **行为**：这种形式会迫使Q“覆盖”P的所有重要模式，因此被称为**“mode-covering”**或**“mean-seeking”**。它倾向于产生一个更“宽泛”、“平均”的分布Q。

#### **方向二：`D_KL(Q || P)` (Reverse KL Divergence)**

$$
D_{KL}(Q || P) = \sum_{x} Q(x) \log\left(\frac{Q(x)}{P(x)}\right)
$$

*   **直观含义**：这个方向的KL散度试图确保 **“当 Q(x) 的概率很高时，P(x) 的概率也必须很高”**。如果Q试图在一个P认为概率很低（`P(x)`小）的地方放置很高的概率（`Q(x)`大），那么 `log(Q/P)` 会是一个很大的正数，导致KL散度爆炸。它不关心Q是否覆盖了P的所有模式。
*   **行为**：这种形式会迫使Q集中在P的高概率区域，避免在P的低概率区域活动。因此被称为**“mode-seeking”**或**“zero-forcing”**。它倾向于产生一个更“尖锐”、“自信”的分布Q，通常会选择P的其中一个主模式（mode）进行拟合。

---

### 2. 在RLHF中的选择与应用

在RLHF的PPO框架中，我们希望策略网络 `π_θ` 不要离参考网络 `π_ref` 太远。我们应该选择哪个方向呢？

**标准的、理论上更常见的选择是 `D_KL(π_θ || π_ref)` (Reverse KL)。**

为什么？
让我们将 `Q` 对应于 `π_θ`（我们要优化的策略），将 `P` 对应于 `π_ref`（固定的参考策略）。

使用 `D_KL(π_θ || π_ref)` 作为惩罚项，意味着我们不希望 `π_θ` 在 `π_ref` 认为“不可能”（概率接近0）的地方，给出很高的概率。这非常符合我们的直觉：**我们允许模型在SFT模型认为合理的范围内，根据奖励信号调整其偏好，但不希望它为了迎合奖励模型而说出一些语法不通、逻辑混乱的“胡话”（这些“胡话”在`π_ref`中的概率极低）。**

这种“mode-seeking”的行为，让 `π_θ` 倾向于在 `π_ref` 已经学到的良好语言模式上进行精调，而不是去探索未知的、可能是坏的语言空间。

---

### 3. 从理论到实践：`logprob` 的差值

现在，我们来看这个理论是如何映射到 `log π_θ(a|s) - log π_ref(a|s)` 这个简单计算上的。

在PPO的框架中，我们不是直接最小化KL散度，而是将其作为奖励的一部分。具体来说，我们是最大化期望奖励：
$$
\mathbb{E}_{(s,a) \sim \pi_{\theta}} [R(s,a) - \beta \cdot D_{KL}(\pi_{\theta}(\cdot|s) || \pi_{ref}(\cdot|s))]
$$
其中 `R(s,a)` 是来自奖励模型的分数。

由于KL散度项本身在每个时间步都计算完整的分布差异是不可行的，所以我们使用蒙特卡洛估计。对于从 `π_θ` 中采样出的一个具体动作 `a`，`D_KL(π_θ || π_ref)` 的**单样本估计 (single-sample estimate)** 就是：
$$
\text{KL_estimate} \approx \log\left(\frac{\pi_{\theta}(a|s)}{\pi_{ref}(a|s)}\right) = \log \pi_{\theta}(a|s) - \log \pi_{ref}(a|s)
$$

这个估计的数学基础是，KL散度的期望可以写作：
$$
\mathbb{E}_{a \sim \pi_{\theta}}[\log \pi_{\theta}(a|s) - \log \pi_{ref}(a|s)]
$$
当我们用从 `π_θ` 采样出的一个样本 `a` 来近似这个期望时，就得到了上面的差值形式。

**所以，`log π_θ - log π_ref` 是对 `D_KL(π_θ || π_ref)` (Reverse KL) 的蒙特卡洛估计。**

### 4. 先后顺序的影响

**您说得完全正确，先后顺序有影响。**

*   **`log π_θ - log π_ref`** 对应 **`D_KL(π_θ || π_ref)` (Reverse KL)**。
    *   **效果**：惩罚 `π_θ` 在 `π_ref` 的低概率区域赋予高概率。这是RLHF中的标准做法。
*   **`log π_ref - log π_θ`** 对应 **`D_KL(π_ref || π_θ)` (Forward KL)**。
    *   **效果**：惩罚 `π_θ` 未能覆盖 `π_ref` 的高概率区域。换句话说，如果SFT模型认为某个词是合理的 (`π_ref`概率高)，但你的新模型 `π_θ` 给了它一个极低的概率，就会受到惩罚。这会鼓励 `π_θ` 变得更“宽泛”，更像 `π_ref` 的平均行为。

虽然两种方式都能起到“拉近”两个分布的作用，但它们的侧重点不同。**Reverse KL (`π_θ || π_ref`) 更侧重于“不犯错”（Don't do stupid things），而Forward KL (`π_ref || π_θ`) 更侧重于“不遗漏”（Don't miss plausible things）。**

在RLHF中，我们的首要目标是防止模型为了追求奖励而“语言崩溃”（generation collapse），产生不连贯的文本。因此，**采用 `D_KL(π_θ || π_ref)` (Reverse KL) 及其对应的 `log π_θ - log π_ref` 惩罚项，是更符合逻辑和目标的选择。**

### 总结

1.  **理论选择**: 在RLHF中，我们通常选择 **`D_KL(π_θ || π_ref)` (Reverse KL)** 作为惩罚项，因为它能有效防止策略网络偏离参考网络太远，避免在低概率区域进行不合理的探索。
2.  **实践计算**: 这个理论上的KL散度在实践中被其**单样本蒙特卡洛估计**所取代，其形式恰好是 `log π_θ(a|s) - log π_ref(a|s)`。
3.  **先后顺序**: `log π_θ - log π_ref` 这个计算形式直接对应于`D_KL(π_θ || π_ref)`。如果反过来写成 `log π_ref - log π_θ`，则对应于`D_KL(π_ref || π_θ)`，其惩罚行为和倾向性会有所不同。
4.  **TRL等库的实现**: 它们采用 `logprob_policy - logprob_ref` 的形式，正是实现了理论上更合理的 **Reverse KL** 惩罚。




## TRL库中PPO_Trainer代码关键处说明注释

```
    # logprob = selective_log_softmax(logits, response)

    response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
    padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
    logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
    ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
    sequence_lengths_p1 = sequence_lengths + 1
    padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
    values = torch.masked_fill(values, padding_mask_p1, 0)

    # 4. compute rewards
    # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
    logr = ref_logprobs - logprobs
    kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr  # Else statement is k3
    non_score_reward = -args.kl_coef * kl
    rewards = non_score_reward.clone()
    actual_start = torch.arange(rewards.size(0), device=rewards.device)
    actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
    rewards[[actual_start, actual_end]] += scores

    # 5. whiten rewards
    if args.whiten_rewards:
        rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
        rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

    # 6. compute advantages and returns
    lastgaelam = 0
    advantages_reversed = []
    gen_length = responses.shape[1]
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
        lastgaelam = delta + args.gamma * args.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    advantages = masked_whiten(advantages, ~padding_mask)
    advantages = torch.masked_fill(advantages, padding_mask, 0)
```

开放的数据集

数据集名称	发布机构	规模	语言	核心特点	许可证
Anthropic/hh-rlhf	Anthropic	约 17 万	英语	经典，有用性 & 无害性双重目标	MIT
Stanford/SHP	斯坦福大学	约 38.5 万	英语	基于 Reddit 投票，真实社区数据	MIT
OpenAssistant/oasst1 & 2	LAION 社区	巨大	多语言 (35+)	众包，对话树结构，多语言首选	Apache 2.0
LMSYS/lmsys-chat-1m	LMSYS	约 100 万	英语为主	真实用户、SOTA 模型间的偏好	CC BY-NC-SA 4.0
PKU-SafeRLHF	北京大学	约 1 万	中/英	专注于安全性对齐	研究用途


## References
- https://github.com/huggingface/transformers.git
- https://www.cnblogs.com/xfuture/p/17809824.html BERT的基本使用和微调方法
- 介绍GPT系列相关技术的博客：http://shiyanjun.cn/archives/2749.html
- 介绍RLHF相关的文章：https://huyenchip.com/2023/05/02/rlhf.html
- 开源RLHF框架
    - https://github.com/huggingface/trl
    - https://github.com/CarperAI/trlx
    - https://github.com/allenai/RL4LMs
    - RLHF博客文章：https://huggingface.co/blog/rlhf
    - 知乎文章如何正确复现Instruct GPT(RLHF): https://zhuanlan.zhihu.com/p/622134699
- Advanced Tricks for Training Large Language Models with Proximal Policy Optimization:
- https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f
- Distributed RLHF on Ray: https://github.com/OpenRLHF/OpenRLHF