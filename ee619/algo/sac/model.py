import torch.cuda
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, architecture, distribution, device='cpu', action_space=None):
        super(Actor, self).__init__()
        self.architecture = architecture
        self.distribution = distribution
        self.device = device
        self.architecture.to(device)
        self.distribution.to(device)
        self.action_space = action_space
        self.loc_action = self.action_space.loc_action.to(device)
        self.scale_action = self.action_space.scale_action.to(device)

    def sample(self, obs):
        return self.architecture.sample(obs)

    def evaluate(self, obs, actions):
        action_mean = self.architecture(obs)
        return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape

# class Actor(nn.Module):
#     def __init__(self, architecture, distribution, device='cpu', action_space=None):
#         super(Actor, self).__init__()
#         self.architecture = architecture
#         self.distribution = distribution
#         self.device = device
#         self.architecture.to(device)
#         self.distribution.to(device)
#         self.action_space = action_space
#         self.loc_action = self.action_space.loc_action.to(device)
#         self.scale_action = self.action_space.scale_action.to(device)
#
#     def sample(self, obs):
#         logits = self.architecture.architecture(obs)
#         actions, log_prob = self.distribution.sample(logits)
#         actions = self.loc_action + self.scale_action * actions
#         log_prob = log_prob.sum(-1, keepdim=True)
#         # return actions.cpu().detach(), log_prob.cpu().detach()
#         return actions, log_prob
#
#     def evaluate(self, obs, actions):
#         action_mean = self.architecture.architecture(obs)
#         return self.distribution.evaluate(obs, action_mean, actions)
#
#     def parameters(self):
#         return [*self.architecture.parameters(), *self.distribution.parameters()]
#
#     @property
#     def obs_shape(self):
#         return self.architecture.input_shape
#
#     @property
#     def action_shape(self):
#         return self.architecture.output_shape

class Critic(nn.Module):
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)
        self.device = device

    # def predict(self, obs):
    #     return self.architecture.architecture(obs).detach()
    #
    # def evaluate(self, obs):
    #     return self.architecture.architecture(obs)
    #
    # def parameters(self):
    #     return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

# class Q(nn.Module):
#     def __init__(self, architecture, device='cpu'):
#         super(Q, self).__init__()
#         self.architecture = architecture
#         self.architecture.to(device)
#         self.device = device
#
#     def predict(self, obs):
#         return self.architecture.architecture(obs).detach()
#
#     def evaluate(self, obs):
#         return self.architecture.architecture(obs)
#
#     def parameters(self):
#         return [*self.architecture.parameters()]
#
#     @property
#     def obs_shape(self):
#         return self.architecture.input_shape
#
#     @property
#     def action_shape(self):
#         return self.architecture.output_shape

class EnsembleMLP(nn.Module):
    def __init__(self, modelA, modelB):
        super(EnsembleMLP, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x1 = self.modelA.architecture(x)
        x2 = self.modelB.architecture(x)
        return x1, x2

class MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_size, output_size):
        super(MLP, self).__init__()

        self.activation_fn = activation_fn
        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]

        for i in range(len(shape)-1):
            modules.append(nn.Linear(shape[i], shape[i+1]))
            modules.append(self.activation_fn())

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)

        self.kaiming_init_(self.architecture)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def kaiming_init_(sequential_module):
        for idx, module in enumerate(mod for mod in sequential_module if isinstance(mod, nn.Linear)):
            torch.nn.init.kaiming_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)

class Gaussian_MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_size, output_size, action_space, device='cpu'):
        super(Gaussian_MLP, self).__init__()

        self.activation_fn = activation_fn

        self.linear1 = nn.Linear(input_size, shape)
        self.linear2 = nn.Linear(shape, shape)
        self.mean_linear = nn.Linear(shape, output_size)
        self.log_std_linear = nn.Linear(shape, output_size)
        self.action_scale = action_space.scale_action.to(device)
        self.action_bias = action_space.loc_action.to(device)

        self.apply(self.kaiming_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    @staticmethod
    def kaiming_init_(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

class MultiVariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultiVariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std)
        self.distribution = None

    def sample(self, logits):
        self.distribution = Normal(logits, self.std)
        samples = self.distribution.rsample()
        samples = torch.tanh(samples)
        log_prob = self.distribution.log_prob(samples)
        samples = samples.type(torch.float32)
        log_prob = log_prob.type(torch.float32)

        return samples, log_prob

    def evaluate(self, logits, outputs):
        distribution = Normal(logits, self.std)
        action_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return action_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std


