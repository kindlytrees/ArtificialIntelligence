# 策略梯度算法

## 知识要点
- 策略梯度基于完整的回合轨迹数据进行训练，不同于时序差分的思想
- 策略梯度原始的算法REINFORCE是只有一个Actor的策略网络训练方法

## 实验代码说明

多次loss.backward()计算的梯度会累积（Gradient Accumulation,循环内的梯度会累积），self.optimizer.step()更新参数，self.optimizer.zero_grad()会将梯度清零。

```
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数（G为从当前步骤开始到结束状态的折扣奖励和），负数最小对应着正数最大（期望的奖励不断提升），对应policy gradient 1.9和1.11部分的介绍，这里不是梯度，而是对数概率，公式里边还有一个梯度算子
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
```

- 关于损失函数按照上述代码进行实现的说明，策略梯度上升算法，其对应的损失函数得采用深度学习框架里现成的梯度下降算法，因此添加了负号，而损失函数也将策略梯度中的\nabla还原成原始对数函数的计算，上述这个过程完美地利用了深度学习框架的现有工具，通过定义一个巧妙的“代理损失函数”。
- 改进方案：可以写成基于batch的算法（batch_size为轨迹的长度），权重通过提前计算好，然后用element wise乘法来实现，最优loss用mean来计算，这样是否也能防止梯度出现计算问题，还可以基于多条轨迹一起作为batch进行训练，能更好的解决高方差的问题，可以参考actor-critic算法的实现思路（基于单个trajectory的批量实现）

$$
\begin{aligned}
\nabla \bar{R}_\theta \approx \frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} G_t^n \nabla \log \pi_\theta\left(a_t^n \mid s_t^n\right) \\
J(\theta) \approx -\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^{T_n} G_t^n  \log \pi_\theta\left(a_t^n \mid s_t^n\right)
\end{aligned}
$$

## 实验的代码实现中，样本的序列长度没有限制，loss的累计回传梯度会不会由较大的情况导致训练参数出现nan？ 
- 如果序列过长，是有可能出现梯度累积导致的梯度爆炸
- 可以通过梯度裁剪，回报标准化（G序列剪均值除以标准差），以及使用优势函数，如后面的A2C（A3C）等算法进行改进