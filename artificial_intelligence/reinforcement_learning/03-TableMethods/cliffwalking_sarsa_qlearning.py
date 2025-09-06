import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


#在 CliffWalking-v1 环境中：
#env.observation_space 是一个从 0 到 47 的离散整数空间（代表4x12网格中的每个位置）。
#env.action_space.n 是 4（上、下、左、右）。

# --- 1. Epsilon-Greedy 策略 ---
def epsilon_greedy_policy(q_table, state, epsilon, env):
    """
    根据给定的Q表和epsilon，为特定状态选择一个动作。
    
    参数:
    q_table (defaultdict): 存储Q值的字典
    state (int): 当前状态
    epsilon (float): 探索率
    env (gym.Env): aI Gym 环境，用于获取动作空间大小

    返回:
    int: 选择的动作
    """
    if np.random.rand() < epsilon:
        # 探索：随机选择一个动作
        return env.action_space.sample()
    else:
        # 利用：选择Q值最高的动作
        # np.argmax在有多个最大值时会选择第一个，这足够了
        return np.argmax(q_table[state])

# --- 2. SARSA 算法实现 ---
def sarsa_step(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    执行单步SARSA更新。
    
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
    """
    old_value = q_table[state][action]
    next_q_value = q_table[next_state][next_action]
    td_target = reward + gamma * next_q_value
    
    new_value = old_value + alpha * (td_target - old_value)
    q_table[state][action] = new_value

# --- 3. Q-Learning 算法实现 ---
def q_learning_step(q_table, state, action, reward, next_state, alpha, gamma):
    """
    执行单步Q-Learning更新。

    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    old_value = q_table[state][action]
    next_max_q = np.max(q_table[next_state])
    td_target = reward + gamma * next_max_q
    
    new_value = old_value + alpha * (td_target - old_value)
    q_table[state][action] = new_value

# --- 4. 评估函数 ---
def evaluate(q_table, env, num_episodes=100):
    """
    评估策略的性能，不进行探索。
    """
    total_rewards = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not terminated and not truncated:
            # 在评估时，我们总是选择最优动作 (epsilon=0)
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards += episode_reward
        
    return total_rewards / num_episodes

# --- 5. 主训练循环 ---
def train(algorithm='q_learning', num_episodes=3000, alpha=0.1, gamma=0.99, 
          epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_rate=0.995):
    """
    主训练函数，增加了epsilon衰减。
    """
    env = gym.make("CliffWalking-v1")
    # defaultdict 来自Python的 collections 模块，它是一种特殊的字典（dictionary）
    # 尝试访问一个**不存在的键（key）**时，会调用默认工厂函数
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    episode_rewards = []
    
    epsilon = epsilon_start # 初始化epsilon
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_episode_reward = 0
        
        # 使用当前的epsilon选择动作
        action = epsilon_greedy_policy(q_table, state, epsilon, env)
        
        while not terminated and not truncated:
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_episode_reward += reward
            
            if algorithm == 'sarsa':
                next_action = epsilon_greedy_policy(q_table, next_state, epsilon, env)
                sarsa_step(q_table, state, action, reward, next_state, next_action, alpha, gamma)
                state, action = next_state, next_action
            elif algorithm == 'q_learning':
                q_learning_step(q_table, state, action, reward, next_state, alpha, gamma)
                state = next_state
                action = epsilon_greedy_policy(q_table, state, epsilon, env)
        
        # --- Epsilon 衰减 ---
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay_rate
        
        episode_rewards.append(total_episode_reward)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Algorithm: {algorithm.upper()} | Episode: {episode + 1}/{num_episodes} | Epsilon: {epsilon:.3f} | Avg Reward (last 100): {avg_reward:.2f}")

    env.close()
    return q_table, episode_rewards

# --- 6. 运行和可视化 ---
if __name__ == "__main__":
    # 超参数设置
    NUM_EPISODES = 1000
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.99
    EPSILON = 0.1

    sarsa_q_table, sarsa_rewards = train(
        algorithm='sarsa',
        epsilon_start=0.5, 
        epsilon_end=0.01,
        epsilon_decay_rate=0.999 # 使用稍慢的衰减率
    )
    
    print("\n--- Starting Q-Learning Training ---")
    qlearning_q_table, qlearning_rewards = train(
        algorithm='q_learning',
        epsilon_start=0.5, 
        epsilon_end=0.01,
        epsilon_decay_rate=0.999 # 使用稍慢的衰减率
    )

    # 评估模型性能
    print("\n--- Evaluating Models ---")
    sarsa_avg_reward = evaluate(sarsa_q_table, gym.make("CliffWalking-v1"))
    qlearning_avg_reward = evaluate(qlearning_q_table, gym.make("CliffWalking-v1"))
    
    print(f"SARSA Average Evaluation Reward: {sarsa_avg_reward:.2f}")
    print(f"Q-Learning Average Evaluation Reward: {qlearning_avg_reward:.2f}")

    # 可视化训练过程中的奖励变化
    plt.figure(figsize=(12, 6))
    plt.plot(sarsa_rewards, label='SARSA Training Rewards', alpha=0.7)
    plt.plot(qlearning_rewards, label='Q-Learning Training Rewards', alpha=0.7)
    
    # 计算移动平均以获得更平滑的曲线
    sarsa_moving_avg = np.convolve(sarsa_rewards, np.ones(50)/50, mode='valid')
    qlearning_moving_avg = np.convolve(qlearning_rewards, np.ones(50)/50, mode='valid')
    plt.plot(sarsa_moving_avg, label='SARSA (Smoothed)', linewidth=2)
    plt.plot(qlearning_moving_avg, label='Q-Learning (Smoothed)', linewidth=2)
    
    plt.title('SARSA vs. Q-Learning Training Performance on CliffWalking-v0')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 可视化学习到的策略路径
    def plot_policy(q_table, title):
        env = gym.make("CliffWalking-v1")
        policy = np.zeros(env.observation_space.n, dtype=int)
        for state in q_table:
            policy[state] = np.argmax(q_table[state])
        
        # 0: up, 1: right, 2: down, 3: left
        actions_symbols = ['↑', '→', '↓', '←']
        policy_symbols = [actions_symbols[a] for a in policy]
        
        # Reshape to grid
        grid_policy = np.array(policy_symbols).reshape(4, 12)
        
        # Mark start, goal, and cliff
        grid_policy[3, 0] = 'S'
        grid_policy[3, 1:11] = 'C' # Cliff
        grid_policy[3, 11] = 'G'
        
        print(f"\n--- {title} Policy ---")
        print(grid_policy)

    plot_policy(sarsa_q_table, "SARSA")
    plot_policy(qlearning_q_table, "Q-Learning")
    # print(sarsa_q_table)
    # print(qlearning_q_table)






    