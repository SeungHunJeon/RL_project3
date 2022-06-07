import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from ReplayMemory import *
import torch.nn.functional as F

class act_space():
    def __init__(self, high, low, shape):
        self.high = torch.tensor(high) * torch.ones(shape)
        self.low = torch.tensor(low) * torch.ones(shape)
        self.shape = shape
        self.loc_action = (self.high + self.low) / 2
        self.scale_action = (self.high - self.low) / 2

class PPO:
    def __init__(self,
                 actor,
                 critic,
                 critic_target,
                 gamma=0.99,
                 lam=0.95,
                 lr = 1e-5,
                 tau = 0.005,
                 alpha = 0.2,
                 clip = 0.2,
                 value_coeff = 0.5,
                 entropy_coeff = 0.01,
                 action_space = None
                 ):
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_space = action_space
        self.clip = clip
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01

        ## Network
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target

        ## Optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

        # criterion
        self.criterion = nn.MSELoss()

        # learning rate (cosine annealing)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.actor_optimizer, T_max=50, eta_min=0)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.critic_optimizer, T_max=50, eta_min=0)

    def observe(self, actor_obs):
        self.actor_obs = torch.from_numpy(actor_obs).float().to(self.device)
        self.actions, self.actions_log_prob = self.actor.sample(self.actor_obs)

        return self.actions.cpu().numpy()

    def update(self, memory, batch_size, training_num):

