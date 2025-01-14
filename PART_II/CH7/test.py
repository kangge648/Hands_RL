import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from ReplayBuffer import Qnet
from ReplayBuffer import ReplayBuffer


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


hidden_dim = 128
state_dim = 128
action_dim = 4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    
q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
print(q_net.parameters())