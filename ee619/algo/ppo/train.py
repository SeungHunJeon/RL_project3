from ruamel.yaml import YAML, dump, RoundTripDumper
import os
import math
import time
from PPO import *
from ReplayMemory import *
from model import *
import torch.nn as nn
import numpy as np
import torch
import argparse
import wandb
from dm_control import suite
from dm_control.rl.control import Environment
import operator
from functools import reduce
from typing import Callable, Iterable, List, NamedTuple, Optional
from ee619.agent import flatten_and_concat

def product(iterable: Iterable[int]) -> int:
    return reduce(operator.mul, iterable, 1)

wandb.init(group="jsh",project="Project_3_SAC")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
domain = cfg['environment']['domain']
seed = cfg['environment']['seed']
task = cfg['environment']['task']

replay_size = cfg['replay']['replay_size']

batch_size = cfg['learning']['batch_size']
gradient_step = cfg['learning']['gradient_step']

Q_net_shape = cfg['architecture']['Q_net']
policy_net_shape = cfg['architecture']['policy_net']


#environment
env = suite.load(domain, task, task_kwargs={'random': seed})

observation_spec = env.observation_spec()
input_dim = sum(product(value.shape) for value in observation_spec.values())

action_spec = env.action_spec()
action_dim = product(action_spec.shape)
max_action = action_spec.maximum.copy()
min_action = action_spec.minimum.copy()
loc_action = (max_action + min_action) / 2
scale_action = (max_action - min_action) / 2

action_space = act_space(max_action, min_action, action_dim)

# Agent
actor_architecture = MLP(policy_net_shape, nn.LeakyReLU, input_dim, action_dim)
critic_architecture = MLP(Q_net_shape, nn.LeakyReLU, input_dim + action_dim, 1)
agent = PPO(actor=Actor(actor_architecture, MultiVariateGaussianDiagonalCovariance(action_dim, torch.tensor(scale_action)), device=device, action_space=action_space), critic=Critic(critic_architecture, device=device), critic_target=Critic(critic_architecture, device=device),action_space=action_space)

# memory
memory = ReplayMemory(capacity=replay_size)

updates = 0
total_steps = 0
training_num = 0

for update in range(10000):
    start = time.time()
    time_step = env.reset()
    reward_sum = 0
    data_log = {}

    while not time_step.last():
        with torch.no_grad():
            obs = flatten_and_concat(time_step.observation)
            action = agent.observe(actor_obs=obs)
            time_step = env.step(action)
            reward = time_step.reward
            next_obs = flatten_and_concat(time_step.observation)
            memory.push(obs, action, reward, next_obs)
            reward_sum = reward_sum + reward

    if len(memory) > batch_size:
        for i in range(gradient_step):
            critic_1_loss, critic_2_loss, policy_loss, alpha_loss = agent.update(memory=memory, batch_size=batch_size, training_num=training_num)
            training_num += 1

    end = time.time()
    # agent.actor.distribution.enforce_minimum_std((torch.ones(6)*0.2).to(device))

    ### For logging
    data_log['reward_sum'] = reward_sum
    data_log['critic_1_loss'] = critic_1_loss
    data_log['critic_2_loss'] = critic_2_loss
    data_log['policy_loss'] = policy_loss
    data_log['alpha_loss'] = alpha_loss
    wandb.log(data_log)

    wandb.watch(agent.actor.architecture, log='all', log_freq = 100)
    wandb.watch(agent.critic.architecture.modelA, log='all', log_freq = 100)
    wandb.watch(agent.critic.architecture.modelB, log='all', log_freq = 100)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("reward_sum: ", '{:0.10f}'.format(reward_sum)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('std: ')
    print(np.exp(agent.actor.distribution.std.cpu().detach().numpy()))
    print('----------------------------------------------------\n')


