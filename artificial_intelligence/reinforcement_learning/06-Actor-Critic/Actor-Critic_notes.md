# Actor Critic Notes

## 知识要点
- actor为策略网络，critic在此实现为价值网络，其损失函数为mse，目标值为基于值函数的一步时间差分
- 同策略网络，样本数据在线采集只使用一次
- 梯度权重为优势函数(定义为价值函数的增量估计)

## 策略梯度中权重的可能的形式

$g=\mathbb{E}\left[\sum_{t=0}^T \psi_t \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right)\right]$

$$
\begin{aligned}
& \sum_{t^{\prime}=0}^T \gamma^{t^{\prime}} r_{t^{\prime}} //轨迹的总回报;\\
& \sum_{t^{\prime}=t}^T \gamma^{t^{\prime}-t} r_{t^{\prime}} //动作 a_t 之后总回报；\\
& \sum_{t^{\prime}=t}^T \gamma^{t^{\prime}-t} r_{t^{\prime}}-b\left(s_t\right) //基准线版本的改进；\\
& Q^{\pi_\theta}\left(s_t, a_t\right) ：动作价值函数 \\
& A^{\pi_\theta}\left(s_t, a_t\right) ：优势函数 \\
& r_{t}+\gamma V^{\pi_\theta}\left(s_{{t}+1}\right)-V^{\pi_\theta}\left(s_{t}\right)  //时序差分残差  \\
\end{aligned}
$$

- 价值网络的损失函数是当前状态价值网络输出和一阶的回报函数(td_target)的mse

```
td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
```
- 策略梯度的损失函数为REINFORCEMENT策略梯度算法的变种, 其中权重设为时序差分误差。

```
td_delta = td_target - self.critic(states)  # 时序差分误差
actor_loss = torch.mean(-log_probs * td_delta.detach())
```

- 在actor-critic算法中，时序差分误差作为权重的意义解释，是否说明误差表明当前状态价值函数估值差异的大小，越大说明当前低估了当前状态的价值，因此应该在此状态下提升相应的策略梯度？

gemini中的回答和上述观点的一致性：“误差表明当前状态价值函数估值差异的大小，越大说明当前低估了当前状态的价值，因此应该在此状态下提升相应的策略梯度。”


```
def update(self, transition_dict):
    states = torch.tensor(transition_dict['states'],
                            dtype=torch.float).to(self.device)
    actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
        self.device)
    rewards = torch.tensor(transition_dict['rewards'],
                            dtype=torch.float).view(-1, 1).to(self.device)
    next_states = torch.tensor(transition_dict['next_states'],
                                dtype=torch.float).to(self.device)
    dones = torch.tensor(transition_dict['dones'],
                            dtype=torch.float).view(-1, 1).to(self.device)

    # 时序差分目标
    td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                    dones)
    td_delta = td_target - self.critic(states)  # 时序差分误差
    # states 是一个形状为 [B, state_dim] 的张量，其中 B 是批量大小 (batch size)。
    # gather的作用是从第1个维度将actions对应的位置的策略函数的输出值取出，第一个维度为样本索引维度
    # nput.gather(dim, index): 它会根据 index 张量中的索引，在 input 张量的 dim 维度上进行索引。
    log_probs = torch.log(self.actor(states).gather(1, actions))
    # detach的作用是从自动微分计算图中脱离出来对吗？
    # td_delta.detach() 会创建一个与 td_delta 值完全相同的新张量，但这个新张量不包含任何梯度历史。它被从计算图中“分离”了出来。
    actor_loss = torch.mean(-log_probs * td_delta.detach())
    # 均方误差损失函数
    critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
    self.actor_optimizer.zero_grad()
    self.critic_optimizer.zero_grad()
    actor_loss.backward()  # 计算策略网络的梯度
    critic_loss.backward()  # 计算价值网络的梯度
    self.actor_optimizer.step()  # 更新策略网络的参数
    self.critic_optimizer.step()  # 更新价值网络的参数
```