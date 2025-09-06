# Policy Value Iteration Notes

马尔可夫决策过程的策略迭代算法和价值迭代算法是有模型的算法，状态行为到下一个状态的转移概率矩阵和即时奖励已知。

## 策略迭代总体思想
总体思想，在特定的策略条件下，价值函数通过贝尔曼方程迭代会趋向于收敛，然后选择Q(s,a)中对应的状态s中最大的行为作为策略，实现策略的更新
更新后再次迭代价值函数，再次更新策略，直至最优策略的生成

## 价值迭代总体思想
直接得出最优的值函数（跨策略的）估计，通过Q函数的贝尔曼方程更新Q函数，价值函数直接更新为Q函数里Q(s,a)在s状态下不同a时的最大值
不断这个过程，得出价值函数的稳定收敛(Q函数也将稳定收敛)，最后基于Q值函数直接生成最优策略

## CliffWalking实验设计

- 行为的表示方法, 4个动作，上下左右，第一个元素表示，始时pi的策略为每个方向的概率相同

```
change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
```

- 模型表示,1表示概率，state用位置索引，在边界时的行为如果超出了边界，则保留在边界，reward为-1

```
P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
```

$V^{t+1}(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V^t\left(s^{\prime}\right)\right)$
这里s,a将是下一个状态为全概率为1的事件

```
for res in self.env.P[s][a]: # 长度为1
    p, next_state, r, done = res
    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
```

策略迭代中价值函数的迭代为`new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系`
价值迭代中`new_v[s] = max(qsa_list)`

## 一排5个格子的策略游戏
- 代码见`policy_iter.py`和`value_iter.py`