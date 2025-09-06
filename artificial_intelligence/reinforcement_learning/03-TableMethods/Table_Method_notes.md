# Table Method notes

首先，我们建立一个核心的框架：有模型（Model-Based） vs. 无模型（Model-Free）。  
模型（Model）：在强化学习中，“模型”特指我们是否知道马尔可夫决策过程（MDP）的动态特性，即：  
状态转移概率 P(s′∣s,a)：在状态s执行动作a后，转移到下一个状态 s′的概率是多少。  
奖励函数R(s,a,s′)：在状态s执行动作a转移到s′后，能获得的即时奖励是多少。  
有模型算法：假设我们完全知道P和R。这类算法不需与环境交互，可以直接在已知的模型上进行“规划”（Planning），计算出最优策略。价值迭代和策略迭代属于此类。  
无模型算法：我们不知道P和R。智能体必须像一个新生儿一样，通过与环境的实际交互来“学习”（Learning），从经验（即样本）中估计价值和策略。SARSA和Q-Learning属于此类。  

蒙特卡洛的两种算法都可以看成是策略迭代的广义版本
- 1、基础随机初始的基于贪心策略的蒙特卡洛方法进行最优策略查找，其采样策略基于贪心策略，即当前的q函数的s状态下的最大值，采样后可以用平均值估计q函数，再进行策略更新，循环这个过程
- 2、基于epsilon-greedy的蒙特卡洛采样方法，基于类似的思路，但其策略做了调整

时序差分方法主要基于n（n常为1）步时序差分的计算
Sarsa和QLearning都是基于Q表格更新的最优策略求解算法，这里的实验基于cliffwalking仿真环境进行。  
- Sarsa是同策略的Q表格学习算法，下一个状态的Q函数Q'(s',a')，使用下一个实际会执行的动作 a'（基于同样的epsilon-greedy策略得出的a'） 来计算目标Q值
- QLearning是一种异策略的Q表格学习算法,下一个状态的Q函数是状态 s' 中可能产生的最大Q值来计算目标Q值

同策略和异策略和策略迭代和值迭代算法之间是不是直接相对应？即sarsa是基于策略迭代的表格法，QLearning基于Q的状态动作的值函数迭代？

Q-Learning和SARSA是将DP中的VI和PI思想，应用到无法获得完整环境模型的现实场景中的学习算法。
SARSA可以看作是“采样版本的、模型无关的策略迭代”。 它不是像DP那样完整地进行评估和改进，而是在每一步交互中，都同时进行一小步的评估和一小步的改进。

Sarsa使用下一个实际会执行的动作 a' 来计算目标Q值。  
QLearning用下一个状态 s' 中可能产生的最大Q值来计算目标Q值，而不管实际会执行哪个动作。  

在代码`cliffwalking_sarsa_qlearning.py`的实现中可以看到，一次差分迭代中action要执行两次，  
一次是当前的状态，另一次是下一个状态下再执行一次`next_action = epsilon_greedy_policy(q_table, next_state, epsilon, env)`  
而QLearning中，一次差分迭代中只有一次当前状态下的action执行，下一个状态直接取值`next_max_q = np.max(q_table[next_state])`  

$$
\begin{aligned}
V_{\pi'}\left(s\right) & =\sum_{a \in A} \pi^{\prime}(a \mid s) Q_\pi(s, a) //基于\pi的\epsilon-greedy策略称为\pi'\\
& //有\epsilon的概率采用均匀随机探索，每个动作被选中的概率为\frac{\epsilon}{|A|}，有(1-\epsilon)的概率采用改进的贪婪策略\pi' \\
& =\frac{\varepsilon}{|A|} \sum_{a \in A} Q_\pi(s, a)+(1-\varepsilon) \max _a Q_\pi(s, a)  \\
& //一组数的最大值，一定大于或等于这组数的任意加权平均值, 权重\frac{\pi(a \mid s)-\frac{\varepsilon}{|A|}}{1-\varepsilon}之和为1\\
& \geqslant \frac{\varepsilon}{|A|} \sum_{a \in A} Q_\pi(s, a)+(1-\varepsilon) \sum_{a \in A} \frac{\pi(a \mid s)-\frac{\varepsilon}{|A|}}{1-\varepsilon} Q_\pi(s, a) \\
& =\sum_{a \in A} \pi(a \mid s) Q_\pi(s, a)=V_\pi(s)
\end{aligned}
$$