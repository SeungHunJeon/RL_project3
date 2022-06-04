import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from ReplayMemory import *
from utils import *
import torch.nn.functional as F

class act_space():
    def __init__(self, high, low, shape):
        self.high = torch.FloatTensor(high) * torch.ones(shape, dtype=torch.float32)
        self.low = torch.FloatTensor(low) * torch.ones(shape, dtype=torch.float32)
        self.shape = shape
        self.loc_action = (self.high + self.low) / 2
        self.scale_action = (self.high - self.low) / 2

class SAC:
    def __init__(self,
                 actor,
                 critic,
                 critic_target,
                 gamma=0.99,
                 lam=0.95,
                 lr = 1e-4,
                 tau = 0.005,
                 alpha = 0.2,
                 alpha_update = False,
                 action_space = None
                 ):
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.alpha = alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha_update = alpha_update
        self.action_space = action_space

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

        # alpha update
        if (self.alpha_update):
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device))
            self.alpha_ = torch.tensor([self.target_entropy], requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.alpha_], lr=lr)

    def observe(self, actor_obs):
        self.actor_obs = torch.FloatTensor(actor_obs).to(self.device)
        self.actions, self.actions_log_prob = self.actor.sample(self.actor_obs)

        return self.actions.cpu().numpy()

    def update(self, memory, batch_size, training_num):
        transitions = memory.sample(batch_size=batch_size)
        mini_batch = memory.Transition(*zip(*transitions))
        state_batch = torch.FloatTensor(np.array(mini_batch.state)).to(self.device)
        action_batch = torch.FloatTensor(np.array(mini_batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(mini_batch.reward)).to(self.device).unsqueeze(-1)
        next_state_batch = torch.FloatTensor(np.array(mini_batch.next_state)).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target.architecture(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + self.gamma * (min_qf_next_target)

        # For critic
        qf1, qf2 = self.critic.architecture(state_batch, action_batch)
        qf1_loss = self.criterion(qf1, next_q_value)
        qf2_loss = self.criterion(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # For actor
        pi, log_pi = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic.architecture(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.zeros(1)

        if (self.alpha_update):
            alpha_loss = -(self.alpha_ * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.alpha_.detach()

        soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()



