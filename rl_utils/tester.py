import os
import gym
import gym.spaces
import gym_service

from rl_utils.model import *

def update_stacked_s(stacked_s, obs, obs_shape):
    obs = torch.from_numpy(obs).float()
    dim_shape = obs_shape[0]
    stacked_s[:, :-dim_shape] = stacked_s[:, dim_shape:]
    stacked_s[:, -dim_shape:] = obs
    return stacked_s

class Tester(object):
    def __init__(self, env_name, filename_prefix, num_stack):
        super(Tester, self).__init__()

        self.env = gym.make(env_name)
        self.filename = filename_prefix + "_params.pkl"
        self.num_stack = num_stack

    def test(self):
        policy = Policy(self.env.observation_space, self.env.action_space)
        policy.eval()
        
        if os.path.isfile(self.filename):
            policy.load_state_dict(torch.load(self.filename))

        obs_shape = self.env.observation_space.shape
        stacked_s = torch.zeros(1, self.num_stack * obs_shape[0], *obs_shape[1:])
        s = self.env.reset()
        update_stacked_s(stacked_s, s, obs_shape)

        while True:
            self.env.render()
            with torch.no_grad():
                a = policy(stacked_s)[0]
            a_np = a.squeeze(0).cpu().numpy()
            s, _, done, _ = self.env.step(a_np)
            if done:
                update_stacked_s(stacked_s, s, obs_shape)
                break
        self.env.close()
