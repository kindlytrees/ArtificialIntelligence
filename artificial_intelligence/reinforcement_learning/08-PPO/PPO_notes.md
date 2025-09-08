# Proximal Policy Optimization

- PPO实验中算法中的优势函数如何定义？
直接根据td_delta进行计算，因为下一个状态的获取采用了 $ \theta' $ 策略对a进行采样的计算。将 $ A^{\theta^{\prime}}\left(s_t, a_t\right) $ 近似成了 $ A^{\theta}\left(s_t, a_t\right) $

- PPO代理目标函数的由来

$$
\begin{array}{ll}
\mathbb{E}_{\left(s_t, a_t\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_\theta\left(s_t, a_t\right)}{p_{\theta^{\prime}}\left(s_t, a_t\right)} A^\theta\left(s_t, a_t\right) \nabla \log p_\theta\left(a_t \mid s_t\right)\right] //基于优势函数作为权重，并采用了异策略进行采样的策略梯度\\
\mathbb{E}_{\left(s_t, a_t\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_\theta\left(a_t \mid s_t\right)}{p_{\theta^{\prime}}\left(a_t \mid s_t\right)} \frac{p_\theta\left(s_t\right)}{p_{\theta^{\prime}}\left(s_t\right)} A^{\theta^{\prime}}\left(s_t, a_t\right) \nabla \log p_\theta\left(a_t \mid s_t\right)\right] //将优势函数近似为异策略下的优势函数计算，并基于时序差分误差的值来定义异策略下的优势函数\\
\mathbb{E}_{\left(s_t, a_t\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_\theta\left(a_t \mid s_t\right)}{p_{\theta^{\prime}}\left(a_t \mid s_t\right)} A^{\theta^{\prime}}\left(s_t, a_t\right) \nabla \log p_\theta\left(a_t \mid s_t\right)\right] //近似的将p_\theta\left(s_t\right) 和p_{\theta^{\prime}}\left(s_t\right)看成相等 \\
\nabla f(x)=f(x) \nabla \log f(x) //f(x)的梯度定义 \\
\nabla p_\theta(\tau)=p_\theta(\tau) \nabla \log p_\theta(\tau) //将上述梯度定义应用到基于\theta的模型参数上，并将和上述的策略梯度的公式进行对照，得出损失函数(原函数)即为下述的J^{\theta^{\prime}}(\theta)\\
J^{\theta^{\prime}}(\theta)=\mathbb{E}_{\left(s_t, a_t\right) \sim \pi_{\theta^{\prime}}}\left[\frac{p_\theta\left(a_t \mid s_t\right)}{p_{\theta^{\prime}}\left(a_t \mid s_t\right)} A^{\theta^{\prime}}\left(s_t, a_t\right)\right]
\end{array}
$$

- PPO中两种代理目标函数的定义

$$
L^{\text {PPO-Penalty }}(\theta)=E\left[r_t(\theta) A_t-\beta D_{\mathrm{KL}}\left(\pi_{\text {old }} \| \pi_\theta\right)\right]
$$

$$
L^{\mathrm{CLIP}}(\theta)=E_t\left[\min \left(r_t(\theta) A_t, \operatorname{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) A_t\right)\right]
$$

- 上述内容在代码中的体现：
    - 在求解两个采样的重要性比重的时候，除法变换为对数的减法，然后用指数函数进行还原
    - 梯度裁剪用到了torch.clamp函数

注：价值网络的训练基本和actor-critic原始算法相同，主要的变化在策略网络的梯度更新上。

```
#均为批量计算，不同时刻的统一处理
td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                dones) #td即为时间差分的迭代去获取Q（动作价值）函数更新
td_delta = td_target - self.critic(states) #更新的delta，td_delta反应了当前状态下的动作相比于平均的动作价值的优势（价值函数为当前动作价值函数的期望值）
advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                        td_delta.cpu()).to(self.device)
old_log_probs = torch.log(self.actor(states).gather(1,
                                                    actions)).detach()

for _ in range(self.epochs): #每个episode的数据训练10次策略网络，数据达到了10次复用
    log_probs = torch.log(self.actor(states).gather(1, actions))
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - self.eps,
                        1 + self.eps) * advantage  # 截断
    actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
    critic_loss = torch.mean(
        F.mse_loss(self.critic(states), td_target.detach())) #当前的和更新后的目标算loss
    self.actor_optimizer.zero_grad()
    self.critic_optimizer.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    self.actor_optimizer.step()
    self.critic_optimizer.step()
```
