import numpy as np
import torch

#update the stack of input so all the old values get shifted up and the new input gets placed at the end
def update_stacked_s(stacked_s, obs, obs_shape):
    obs = torch.from_numpy(obs).float()
    dim_shape = obs_shape[0]
    stacked_s[:, :-dim_shape] = stacked_s[:, dim_shape:]
    stacked_s[:, -dim_shape:] = obs
    return stacked_s

#get the entropy of the normal distribution with the given standard deviation
def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * np.pi)
    return entropy.sum(1, keepdim=True)

#get the probability of choosing x from the normal distribution N(mean, std)
def normal_log_density(x, mean, log_std):
    std = torch.exp(log_std)
    var = std.pow(2)
    log_density = - torch.pow(x - mean, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std

    return log_density.sum(1, keepdim=True)
