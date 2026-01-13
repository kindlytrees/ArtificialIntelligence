import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils
import os


class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)
        self.device = device

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q

    def predict(self, x):
        x = torch.tensor(x,dtype=torch.float).to(self.device)
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        #print(f'V, A shape: {V.shape}, {A.shape}')
        Q = V + A - A.mean(0).view(-1)  # Q值由V值和A值计算得到
        action = Q.argmax().item()
        return action
                
hidden_dim = 128
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
agent = VAnet(state_dim, hidden_dim, action_dim, device)
save_dir = './outputs'
os.makedirs(save_dir, exist_ok=True)
weights_path = os.path.join(save_dir, 'cartpole_dueling_dqn.pth')
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

