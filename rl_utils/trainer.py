import os
from copy import deepcopy
from visdom import Visdom
import numpy as np

import gym, gym.spaces
import gym_service

import torch
import torch.nn as nn
import torch.optim as optim
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from rl_utils.model import Policy
from rl_utils.storage import RolloutStorage
from rl_utils.visualize import Visualizer
from rl_utils.utils import *

xs, means, stds = [], [], []
xs, medians, first_quartiles, third_quartiles, mins, maxes = [], [], [], [], [], []
losses = []

class Trainer(object):
    def __init__(self, env_name, args, iter_args, graph_info, filename_prefix, make_env):
        super(Trainer, self).__init__()

        self.gamma, self.tau, self.eps, self.num_mb, self.num_stack, self.N, self.T, self.total_steps, self.epochs, self.lr, self.value_loss_coef, self.entropy_coef, self.max_grad_norm, self.clip = args

        self.iters = int(self.total_steps) // self.N // self.T
        self.log_iter, self.vis_iter, self.save_iter = iter_args

        self.graph_colors, self.win_name = graph_info

        self.filename = filename_prefix + "_params.pkl"

        self.make_env = make_env
        self.env_name = env_name

        self.visualizer = Visualizer()

    def train(self):
        #my laptop only has 8 cores and I generally use 8 actors for stuff, so make sure that the multiprocessing module doesn't try to give each actor multiple threads and make them fight
        os.environ['OMP_NUM_THREADS'] = '1'

        #make the environments and set them to run in parallel
        #thank you OpenAI for doing the multiprocessing stuff for me
        envs = [self.make_env(self.env_name, 42, n) for n in range(self.N)]
        envs = SubprocVecEnv(envs)

        obs_shape = envs.observation_space.shape

        #create policy network and set it to training mode
        entry_obs_shape = (obs_shape[0] * self.num_stack, *obs_shape[1:])
        self.policy = Policy(entry_obs_shape, envs.action_space)
        self.policy.train()

        #create storage for past actions
        rollouts = RolloutStorage()

        #set optimizer for updating the weights of our network
        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)

        #load saved weights if you can
        if os.path.isfile(self.filename):
            print("loading saved params")
            self.policy.load_state_dict(torch.load(self.filename))

        #init some variables to track how much reward we're getting
        episode_rewards = torch.zeros([self.N, 1])
        final_rewards = torch.zeros([self.N, 1])

        #init the stack
        #with most things we won't stack inputs, but having a 'num_stack' works the same as not having a stack at all so we good
        stacked_s = torch.zeros(self.N, self.num_stack * obs_shape[0], *obs_shape[1:])
        s = envs.reset()
        stacked_s = update_stacked_s(stacked_s, s, obs_shape)

        #start the training
        for iter in range(self.iters):

            #go through some timesteps
            for step in range(self.T):

                #get the predicted action and how sure the network is of taking that action
                #get the predicted value of our current state too
                with torch.no_grad():
                    a, log_p, v = self.policy(stacked_s)

                #transform the action so it's only 1 dimension
                a_np = a.squeeze(1).cpu().numpy()

                #step through the environment and observe what happens
                s2, r, done, _ = envs.step(a_np)
                #reshape the rewards so they're all in separate rows
                #each actor has its own row
                r = torch.from_numpy(r).view(-1, 1).float()
                episode_rewards += r

                #set a mask for this state
                #we'll use this calculate returns and update the stack
                #if we're done, the mask is 0 -> this'll make returns stop cumulating at this point and it'll clear past actions from the stack so those past actions don't confuse the network
                #we should apply the mask to the stack after we've stored it (so we don't mess up the data we're currently using), so we don't do it just yet
                #I struggled with that last part for a bit, so imagine you're playing pong with frame stacking. Once the env resets, the last frames of the previous game don't affect you at all so they shouldnt be used to predict what comes next
                mask = torch.FloatTensor([[0.0] if d else [1.0] for d in done])

                #store the data from this state
                #since stacked_s is declared at a higher scope, chaning its value in the training loop will change all the stored stacked_s values unless you store a copy of it instead
                rollouts.add(deepcopy(stacked_s), log_p, v, a, r, mask)

                #clears the stack if the env is done
                #there's no point in resetting the stack if there's only 1 value in it. the value will get reset in a few lines anyway so why do unnecessary math
                if self.num_stack > 1:
                    stacked_s *= mask

                #keep track of those rewards
                final_rewards *= mask
                final_rewards += (1 - mask) * episode_rewards
                episode_rewards *= mask

                #update stacked_s
                s = s2
                stacked_s = update_stacked_s(stacked_s, s, obs_shape)

            #predict one more value so we can calculate returns and advantages
            with torch.no_grad():
                next_v = self.policy.get_value(stacked_s)
            rollouts.compute_adv_and_returns(next_v, self.gamma, self.tau, self.eps)

            #optimization epochs
            for epoch in range(self.epochs):

                #get the minibatches
                data = rollouts.get_mb(self.num_mb, self.N, self.T)

                #loop through the minibatches
                for sample in data:
                    s_mb, log_p_old_mb, a_mb, returns_mb, adv_mb = sample
                    log_p_mb, v_mb, entropy = self.policy.eval_a(s_mb, a_mb)

                    #calculate the surrogate function
                    #https://arxiv.org/pdf/1707.06347.pdf
                    ratio = torch.exp(log_p_mb - log_p_old_mb)
                    f1 = ratio * adv_mb
                    f2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_mb

                    #calculate the loss
                    #policy loss is based on the surrogate
                    policy_loss = -torch.min(f1, f2).mean()
                    #value loss is mean squared error of the returns and the predicted values
                    value_loss = torch.pow(returns_mb - v_mb, 2).mean() * self.value_loss_coef
                    #entropy loss isn't really loss -> it subtracts from the loss to promote exploration
                    entropy_loss = (entropy * self.entropy_coef)
                    loss = policy_loss + value_loss - entropy_loss

                    #backprop and update weights
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    optimizer.step()

            #clear storage
            rollouts.reset()

            #update plots
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

                losses.append(loss.item())

                self.visualizer.update_viz_median(xs, medians, first_quartiles, third_quartiles, mins, maxes, self.graph_colors, self.env_name, self.win_name)
                self.visualizer.update_viz_mean(xs, means, stds, self.graph_colors[1:], self.env_name, self.win_name)
                self.visualizer.update_viz_loss(xs, losses, self.graph_colors[2], self.env_name, self.win_name)

            #log the current data
            if iter % self.log_iter == self.log_iter - 1:
                print("iter: %d, steps: %d -> mean: %.1f, median: %.1f / min: %.1f, max: %.1f / policy loss: %.3f, value loss: %.1f, entropy loss: %.3f" % (iter + 1, total_num_steps, mean_r, median_r, min_r, max_r, policy_loss, value_loss, entropy_loss))

            #save current weights
            if iter % self.save_iter == self.save_iter - 1:
                torch.save(self.policy.state_dict(), self.filename)
                print("params saved")

        #save current weights when we're all done
        torch.save(self.policy.state_dict(), self.filename)
        print("params saved")
