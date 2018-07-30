import torch
import torch.nn as nn
import numpy as np

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * np.pi)
    return entropy.sum(1, keepdim=True)

def normal_log_density(x, mean, log_std):
    std = torch.exp(log_std)
    var = std.pow(2)
    log_density = - torch.pow(x - mean, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std

    return log_density.sum(1, keepdim=True)

class MLP(nn.Module):
    def __init__(self, n_obs, n_acts):
        super(MLP, self).__init__()

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

    def eval(self, s, a):
        mean = self.mean(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_p = normal_log_density(a, mean, log_std)
        v = self.critic(s)
        entropy = normal_entropy(std).mean()
        return log_p, v, entropy
