import os
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
    def predict(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(state))
        probs =  F.softmax(self.fc2(x), dim=1)
        action = torch.argmax(probs, dim=-1)
        return action.numpy()[0]

save_dir = './outputs'
os.makedirs(save_dir, exist_ok=True)
weights_path = os.path.join(save_dir, 'agent.pth')

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
#env.seed(0)
env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = PolicyNet(state_dim, hidden_dim, action_dim, device)
agent.load_state_dict(torch.load(weights_path, map_location=device))
state, _ = env.reset()
done = False
while not done:
    action = agent.predict(state)
    print('action:', action)
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    env.render()