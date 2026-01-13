# MDP(马尔可夫奖励过程)

## 内容摘要
- 马尔可夫性
- 马尔可夫奖励过程
- 马尔可夫决策过程
- 策略迭代与价值迭代算法（找出最优策略，基于有模型的算法，状态转移矩阵和即时奖励reward已知）

## 知识要点细节

- 回报函数的定义

$$
\begin{aligned}
V^t(s) & =\mathbb{E}_\pi\left[G_t \mid s_t=s\right] \\
& =\mathbb{E}_\pi\left[r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\ldots+\gamma^{T-t-1} r_T \mid s_t=s\right]
\end{aligned}
$$

- 蒙特卡洛法估计马尔可夫奖励过程价值函数

$$
\begin{aligned}
& i \leftarrow 0, G_t \leftarrow 0 \\
& 当 i \neq N 时，执行： \\
& \quad 生成一个回合的轨迹，从状态 s 和时刻 t 开始使用生成的轨迹计算回报 g=\sum_{i=t}^{H-1} \gamma^{i-t} r_{i+1} \\
& \quad G_t \leftarrow G_t+g, i \leftarrow i+1 \\
& 结束循环 \\
& V(s) \leftarrow G_t / N \\
\end{aligned}
$$

## 马尔可夫决策过程

$$
\begin{aligned}
V(s) & =\mathbb{E}\left[G_t \mid s_t=s\right] \\
& =\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^2 r_{t+3}+\ldots \mid s_t=s\right] \\
& =\mathbb{E}\left[r_{t+1} \mid s_t=s\right]+\gamma \mathbb{E}\left[r_{t+2}+\gamma r_{t+3}+\gamma^2 r_{t+4}+\ldots \mid s_t=s\right] \\
& =R(s)+\gamma \mathbb{E}\left[G_{t+1} \mid s_t=s\right] \\
& =R(s)+\gamma \mathbb{E}\left[V\left(s_{t+1}\right) \mid s_t=s\right] \\
& =R(s)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)
\end{aligned}
$$

$$
\begin{aligned}
\mathbb{E}\left[\mathbb{E}\left[G_{t+1} \mid s_{t+1}\right] \mid s_t\right] & =\mathbb{E}\left[\mathbb{E}\left[g^{\prime} \mid s^{\prime}\right] \mid s\right]  \\
& =\mathbb{E}\left[\sum_{g^{\prime}} g^{\prime} p\left(g^{\prime} \mid s^{\prime}\right) \mid s\right] // 期望的定义: \mathbb{E}\left[g^{\prime} \mid s^{\prime}\right]  = \sum_{g^{\prime}} g^{\prime} p\left(g^{\prime} \mid s^{\prime}\right) \\
& =\sum_{s^{\prime}} \sum_{g^{\prime}} g^{\prime} p\left(g^{\prime} \mid s^{\prime}, s\right) p\left(s^{\prime} \mid s\right) // 状态转移到下一个所有可能的s' \\
& =\sum_{s^{\prime}} \sum_{g^{\prime}} \frac{g^{\prime} p\left(g^{\prime} \mid s^{\prime}, s\right) p\left(s^{\prime} \mid s\right) p(s)}{p(s)} //等价变换\\
& =\sum_{s^{\prime}} \sum_{g^{\prime}} \frac{g^{\prime} p\left(g^{\prime} \mid s^{\prime}, s\right) p\left(s^{\prime}, s\right)}{p(s)} //公式分子最右边联合概率分布为条件分布乘以把边缘分布\\
& =\sum_{s^{\prime}} \sum_{g^{\prime}} \frac{g^{\prime} p\left(g^{\prime}, s^{\prime}, s\right)}{p(s)} //公式分子最右边联合概率分布为条件分布乘以把边缘分布 \\
& =\sum_{s^{\prime}} \sum_{g^{\prime}} g^{\prime} p\left(g^{\prime}, s^{\prime} \mid s\right) //条件概率分布的定义 \\
& =\sum_{g^{\prime}} \sum_{s^{\prime}} g^{\prime} p\left(g^{\prime}, s^{\prime} \mid s\right) //求和顺序做一下等价置换\\
& =\sum_{g^{\prime}} g^{\prime} p\left(g^{\prime} \mid s\right) //联合概率分布在某一个变量上的求和得到边缘分布\\
& =\mathbb{E}\left[g^{\prime} \mid s\right]=\mathbb{E}\left[G_{t+1} \mid s_t\right] //期望的定义
\end{aligned}
$$

- 蒙特卡洛法估计马尔可夫决策过程价值函数

$$
\begin{aligned}
&  i \leftarrow 0, G_t \leftarrow 0 \\
&  当 i \neq N 时，执行： \\
&  生成一个回合的轨迹，从状态 s 和时刻 t 开始使用生成的轨迹计算回报 g=\sum_{i=t}^{H-1} \gamma^{i-t} r_{i+1} \\
&  \quad G_t \leftarrow G_t+g, i \leftarrow i+1 \\
&  结束循环 \\
&  V_{\pi}(s) \leftarrow G_t / N \\
\end{aligned}
$$