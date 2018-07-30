import os
import gym
import gym.spaces
import gym_simple
from visdom import Visdom
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import multiprocessing

from rl_utils.model import *
from rl_utils.storage import RolloutStorage
from rl_utils.visualize import *

viz = Visdom()
xs, means, stds = [], [], []
xs, medians, first_quartiles, third_quartiles, mins, maxes = [], [], [], [], [], []

class Trainer(object):
    def __init__(self, env_name, args, iter_args, graph_info, filename_prefix, make_env):
        super(Trainer, self).__init__()

        self.gamma, self.tau, self.eps, self.num_mb, self.N, self.T, self.total_steps, self.epochs, self.lr, self.value_loss_coef, self.entropy_coef, self.max_grad_norm, self.clip = args

        self.iters = int(self.total_steps) // self.N // self.T
        self.log_iter, self.vis_iter, self.save_iter = iter_args

        self.graph_colors, self.win_name = graph_info

        self.filename = filename_prefix + "_params.pkl"

        self.make_env = make_env
        self.env_name = env_name

    def train(self):
        os.environ['OMP_NUM_THREADS'] = '1'

        envs = [self.make_env(self.env_name, 42, n) for n in range(self.N)]
        envs = SubprocVecEnv(envs)

        obs_shape = envs.observation_space.shape

        self.policy = MLP(obs_shape[0], envs.action_space.shape[0])
        rollouts = RolloutStorage()
        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)

        if os.path.isfile(self.filename):
            print("loading saved params")
            self.policy.load_state_dict(torch.load(self.filename))

        episode_rewards = torch.zeros([self.N, 1])
        final_rewards = torch.zeros([self.N, 1])

        s = envs.reset()

        for iter in range(self.iters):
            for step in range(self.T):
                with torch.no_grad():
                    a, log_p, v = self.policy(torch.FloatTensor(s))
                a_np = a.squeeze(1).cpu().numpy()

                s2, r, done, _ = envs.step(a_np)
                r = torch.from_numpy(r).view(-1, 1).float()
                episode_rewards += r

                mask = torch.FloatTensor([[0.0] if d else [1.0] for d in done])

                rollouts.add(s, log_p, v, a, r, mask)

                final_rewards *= mask
                final_rewards += (1 - mask) * episode_rewards
                episode_rewards *= mask

                s = s2

            with torch.no_grad():
                next_v = self.policy.get_value(torch.FloatTensor(s))
            rollouts.compute_adv_and_returns(next_v, self.gamma, self.tau, self.eps)

            for epoch in range(self.epochs):
                data = rollouts.get_mb(self.num_mb, self.N, self.T)

                for sample in data:
                    s_mb, log_p_old_mb, a_mb, returns_mb, adv_mb = sample

                    log_p_mb, v_mb, entropy = self.policy.eval(s_mb, a_mb)

                    ratio = torch.exp(log_p_mb - log_p_old_mb)
                    f1 = ratio * adv_mb
                    f2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_mb

                    policy_loss = -torch.min(f1, f2).mean()
                    value_loss = torch.pow(returns_mb - v_mb, 2).mean() * self.value_loss_coef
                    entropy_loss = (entropy * self.entropy_coef)
                    loss = policy_loss + value_loss - entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    optimizer.step()

            rollouts.reset()

            total_num_steps = (iter + 1) * self.N * self.T

            if iter % self.vis_iter == self.vis_iter - 1:
                xs.append(total_num_steps)

                graph_rewards = final_rewards.view(1, -1)
                mean_r = graph_rewards.mean().item()
                median_r = graph_rewards.median().item()
                min_r = torch.min(graph_rewards).item()
                max_r = torch.max(graph_rewards).item()
                std_r = graph_rewards.std().item()

                medians.append(median_r)
                first_quartiles.append(np.percentile(graph_rewards.numpy(), 25))
                third_quartiles.append(np.percentile(graph_rewards.numpy(), 75))
                mins.append(min_r)
                maxes.append(max_r)
                means.append(mean_r)
                stds.append(std_r)

                # update_viz_median(xs, medians, first_quartiles, third_quartiles, mins, maxes, self.graph_colors, self.env_name, self.win_name)
                # update_viz_mean(xs, means, stds, self.graph_colors[1:], self.env_name, self.win_name)
                update_viz_dots(xs, means, "Mean", self.env_name, self.win_name)

            if iter % self.log_iter == self.log_iter - 1:
                print("iter: %d, steps: %d -> mean: %.1f, median: %.1f / min: %.1f, max: %.1f / policy loss: %.3f, value loss: %.1f, entropy loss: %.3f" % (iter + 1, total_num_steps, mean_r, median_r, min_r, max_r, policy_loss, value_loss, entropy_loss))

            if iter % self.save_iter == self.save_iter - 1:
                torch.save(self.policy.state_dict(), self.filename)
                print("params saved")

        torch.save(self.policy.state_dict(), self.filename)
        print("params saved")
