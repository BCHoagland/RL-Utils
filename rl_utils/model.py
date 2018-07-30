import torch
import torch.nn as nn
import numpy as np
from rl_utils.utils import *

DISCRETE = "Discrete"
CONTINUOUS = "Box"

MultiCategorical = torch.distributions.Categorical

sample_temp = MultiCategorical.sample
MultiCategorical.sample = lambda dist: sample_temp(dist).unsqueeze(-1)

log_prob_temp = MultiCategorical.log_prob
MultiCategorical.log_probs = lambda dist, a: log_prob_temp(dist, a.squeeze(-1)).unsqueeze(-1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(Policy, self).__init__()

        if action_space.__class__.__name__ == DISCRETE:
            n_acts = action_space.n
            if len(obs_shape) == 3:
                self.base = CNNDiscrete(obs_shape[0], n_acts)
            elif len(obs_shape) == 1:
                self.base = MLPDiscrete(obs_shape[0], n_acts)
        elif action_space.__class__.__name__ == CONTINUOUS:
            n_acts = action_space.shape[0]
            if len(obs_shape) == 3:
                self.base = CNNContinuous(obs_shape[0], n_acts)
            elif len(obs_shape) == 1:
                self.base = MLPContinuous(obs_shape[0], n_acts)
        else:
            raise NotImplementedError

    def forward(self, s):
        return self.base(s)

    def get_value(self, s):
        return self.base.get_value(s)

    def eval_a(self, s, a):
        return self.base.eval_a(s, a)

class MLPContinuous(nn.Module):
    def __init__(self, n_obs, n_acts):
        super(MLPContinuous, self).__init__()

        self.mean = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_acts)
        )

        self.log_std = nn.Parameter(torch.zeros(1, n_acts))

        self.critic = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        mean = self.mean(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        a = torch.normal(mean, std)
        log_p = normal_log_density(a, mean, log_std)
        v = self.critic(s)

        return a, log_p, v

    def get_value(self, s):
        return self.critic(s)

    def eval_a(self, s, a):
        mean = self.mean(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_p = normal_log_density(a, mean, log_std)
        v = self.critic(s)
        entropy = normal_entropy(std).mean()
        return log_p, v, entropy

class MLPDiscrete(nn.Module):
    def __init__(self, n_obs, n_acts):
        super(MLPDiscrete, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_acts)
        )

        self.critic = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        probs = self.actor(s)
        dist = MultiCategorical(logits=probs)

        a = dist.sample()
        log_p = dist.log_probs(a)
        v = self.critic(s)

        return a, log_p, v

    def get_value(self, s):
        return self.critic(s)

    def eval_a(self, s, a):
        probs = self.actor(s)
        dist = MultiCategorical(logits=probs)

        log_p = dist.log_probs(a)
        v = self.critic(s)
        entropy = dist.entropy().mean()

        return log_p, v, entropy

class CNNContinuous(nn.Module):
    def __init__(self, n_obs, n_acts):
        super(CNNContinuous, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(n_obs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(1568, 512),
            nn.ReLU()
        )

        self.mean = nn.Sequential(
            nn.Linear(512, n_acts)
        )

        self.log_std = nn.Parameter(torch.zeros(1, n_acts))

        self.critic = nn.Linear(512, 1)

    def forward(self, s):
        s = self.main(s)

        mean = self.mean(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        a = torch.normal(mean, std)
        log_p = normal_log_density(a, mean, log_std)
        v = self.critic(s)

        return a, log_p, v

    def get_value(self, s):
        s = self.main(s)
        return self.critic(s)

    def eval_a(self, s, a):
        s = self.main(s)

        mean = self.mean(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_p = normal_log_density(a, mean, log_std)
        v = self.critic(s)
        entropy = normal_entropy(std).mean()
        return log_p, v, entropy

class CNNDiscrete(nn.Module):
    def __init__(self, n_obs, n_acts):
        super(CNNDiscrete, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(n_obs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(1568, 512),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(512, n_acts)
        )

        self.critic = nn.Linear(512, 1)

    def forward(self, s):
        s = self.main(s)

        lgts = self.actor(s)
        dist = MultiCategorical(logits=lgts)

        a = dist.sample()
        log_p = dist.log_probs(a)
        v = self.critic(s)

        return a, log_p, v

    def get_value(self, s):
        s = self.main(s)
        return self.critic(s)

    def eval_a(self, s, a):
        s = self.main(s)

        lgts = self.actor(s)
        dist = MultiCategorical(logits=lgts)

        log_p = dist.log_probs(a)
        v = self.critic(s)
        entropy = dist.entropy().mean()

        return log_p, v, entropy
