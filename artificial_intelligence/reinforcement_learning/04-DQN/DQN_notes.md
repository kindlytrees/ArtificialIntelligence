# DQN 

## 经验回放池的作用和实现
- 问题1： 在dqn中，一般采用经验回放池缓冲在线生成的数据，这样数据也就可能会有多次使用到，但随着训练过程，老的历史数据对于最新的更好的策略的适用性会不会过时，因此经验回放池在实现的使用基于双端队列进行实现，这样采集满了以后最老的数据就会逐渐的从队列中移除，请分析和补充上述的理解。

回答：(采用gemini辅助回答的部分内容)
为什么“过时”数据仍然有价值？即使是“过时”的数据，也并非完全无用。它们的存在有几个重要的正面作用：
- 维持多样性（Diversity）：早期的策略（比如ε-greedy中的随机探索部分）会探索到更广泛的状态空间。这些“罕见”的经验，即使由一个差的策略产生，也能帮助网络理解状态空间的边界情况。例如，一个已经学会“不要掉下悬崖”的智能体，如果完全不用旧数据，可能会慢慢“忘记”悬崖有多危险，因为它的新策略让它根本不会靠近悬崖。旧数据就像一个“警钟”，不断提醒网络某些行为的后果。
- 防止灾难性遗忘（Catastrophic Forgetting）：神经网络在学习新知识时，容易忘记旧的知识。如果只用最新的数据训练，模型可能会过拟合到当前策略的行为模式，从而忘记了在其他状态下的正确行为。经验回放池中的历史数据有助于巩固和泛化已学到的知识。
- 稳定训练过程：完全使用最新的数据进行训练会导致策略的剧烈摆动。想象一下，智能体偶然发现一个高奖励区域，如果只用这些新数据，它的策略可能会迅速向这个区域收敛，而忽略了全局最优解。混合新旧数据可以平滑学习过程，让策略的更新更加稳健。

但随着智能体策略（Policy）的优化，早期的、由较差策略产生的数据，对于训练当前更优的策略来说，其指导意义会下降，甚至可能产生误导。这就是所谓的**数据“过时”（Stale Data）**问题。
经典解决方案：使用一个固定大小的先进先出（FIFO）队列，通常用双端队列（deque）实现。当队列满了之后，新采集的数据会顶掉最老的数据。

## DQN的实现
- 问题1：一般参数更新多少次，会将最新的q网络的参数同步到目标网络
- 两个Q网络，1：参数更新的Q网络；2：目标网络，用于生成训练目标，用于训练真值的生成,算法伪代码如下：

$$
\begin{aligned}
&\text { 用随机的网络参数 } \omega \text { 初始化网络 } Q_\omega(s, a)， { 复制相同的参数 } \omega^{-} \leftarrow \omega \text { 来初始化目标网络 } Q_{\omega^{\prime}},{ 初始化经验回放池 } R \\
&\text { for 序列 } e=1 \rightarrow E \text { do }\\
&\quad \text { 获取环境初始状态 } s_1\\
&\quad \text { for 时间步 } t=1 \rightarrow T \text { do }\\
&\quad \quad \text { 根据当前网络 } Q_\omega(s, a) \text { 以 } \epsilon \text {-贪婪策略选择动作 } a_t\\
&\quad \quad \text { 执行动作 } a_t \text {, 获得回报 } r_t \text {, 环境状态变为 } s_{t+1}, { 将 }\left(s_i, a_i, r_t, s_{t+1}\right) \text { 存储进回放池 } R \text { 中 }\\
&\quad \quad \text { 若 } R \text { 中数据足够, 从 } R \text { 中采样 } N \text { 个数据 }\left\{\left(s_i, a_i, r_i, s_{i+1}\right)\right\}_{i=1, \ldots, N}\\
&\quad \quad \text { 对每个数据, 用目标网络计算 } y_i=r_i+\gamma \max _a Q_{\omega^{-}}\left(s_{i+1}, a\right)\\
&\quad \quad \text { 最小化目标损失 } L=\frac{1}{N} \sum_i\left(y_i-Q_\omega\left(s_i, a_i\right)\right)^2 \text {, 以此更新当前网络 } Q_\omega\\
&\quad \quad \text { 当前策略网络训练参数向前迭代更新到了一定的步骤后将当前策略网络的参数更新到目标网络 }\\
&\quad \text { end for }\\
&\text { end for }
\end{aligned}
$$

## 标准DQN，Double DQN和Dueling DQN
- 标准DQN，所有的DQN都由训练网络和目标网络组成，但是标准DQN中的动作选择和td_target的计算都是在目标网络进行
- Double DQN
    - Q网络在训练初期或面对未知状态时，其输出的Q值是不准确的，充满了噪声(reward是确定的，但是由于神经网络起始时参数的随机初始化，Q函数的输出结果是有随机性和噪声的)。max操作符会倾向于选择那些被偶然高估（due to estimation error）的动作的Q值。这意味着，即使所有动作的真实Q值都差不多，只要其中一个因为随机误差而显得稍高，max就会选中它，导致TD Target偏高（而实际上也许该Q值没有计算的这么高，导致训练不稳定）
    - 将训练网络和目标网络作为两个独立的网络（目标网络是训练网络在过去的某个时间点的“快照”），一个动作在训练网络上被高估的概率，和它同时在目标网络上也被高估的概率，会小很多
    - 通过这种“解耦”（decoupling），Double DQN有效地减少了因为估计误差而导致的Q值过高估计问题，这使得Q值的估计更加准确，学习过程更加稳定，最终通常能获得比标准DQN更好的性能和策略
    - 下面的代码中使用了两种策略（策略一为if，为double dqn的实现，策略二为else，为一般dqn的实现），如果一个Dueling DQN的实现代码中使用了策略二，那么它就是一个“Dueling DQN”，
    ```
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(
                1, max_action)
        else:
            #如果一个Dueling DQN的实现代码中使用了策略二，那么它就是一个“Dueling DQN”，但不是一个“Dueling Double DQN”。
            #而一个使用了策略一的实现，就是一个“Dueling Double DQN”，后者通常性能更好。
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1)
    ```
- Deuling DQN，上述的代码实现中，else部分为dueling dqn的实现，其不是一个“Dueling Double DQN”,而如果使用了策略一(if逻辑的)的实现，就是一个“Dueling Double DQN”，后者通常性能更好

## 对实验代码实现的说明
蒙特卡洛方法必须等待整个回合（episode）完全结束后，才能知道每一步最终带来了多少总回报（Return）。
然后用这个真实、完整的回报来更新这个回合中经历过的所有状态的价值
代码实现中是基于时序差分的在线学习，不用等待采样整个回合完成后才进行训练和更新，因此不属于蒙特卡洛采样方法。
后续的内容策略梯度算法（REINFORCE）就是基于完整回合的蒙特卡洛方法