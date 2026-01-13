import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils
import os

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
    
    def predict(self, x):
        x = torch.tensor(x,dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        result =  self.fc2(x)
        action = result.argmax().item()
        return action
        
learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

save_dir = './outputs'
os.makedirs(save_dir, exist_ok=True)

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
#env.seed(0)
env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = Qnet(state_dim, hidden_dim, action_dim, device)
save_dir = './outputs'
os.makedirs(save_dir, exist_ok=True)
weights_path = os.path.join(save_dir, 'cartpole_dqn.pth')
agent.to(device)
agent.load_state_dict(torch.load(weights_path, map_location=device))
state, _ = env.reset()
done = False
while not done:
    action = agent.predict(state)
    print('action:', action)
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    env.render()

