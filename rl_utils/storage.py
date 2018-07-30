import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RolloutStorage(object):
    def __init__(self):
        super(RolloutStorage, self).__init__()

        self.init_stores()
        self.nnnn = 0

    def init_stores(self):
        self.s = []
        self.p = []
        self.v = []
        self.a = []
        self.returns = []
        self.r = []
        self.mask = []
        self.adv = []

    def add(self, s, p, v, a, r, mask):
        self.s.append(s)
        self.p.append(p)
        self.v.append(v)
        self.a.append(a)
        self.r.append(r)
        self.mask.append(mask)

    def reset(self):
        self.init_stores()

    def compute_adv_and_returns(self, next_v, gamma, tau, eps):
        T = len(self.s)
        delta = [0] * T

        next_delta = 0
        for t in reversed(range(T)):
            delta[t] = self.r[t] + (gamma * next_v * self.mask[t]) - self.v[t]
            self.adv.insert(0, delta[t] + (gamma * tau * next_delta * self.mask[t]))
            next_v = self.v[t]
            next_delta = delta[t]

        self.adv = torch.stack(self.adv)
        self.returns = self.adv + torch.stack(self.v)
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + eps)

    def get_mb(self, num_mb, N, T):
        s = torch.stack(self.s)
        p = torch.stack(self.p)
        a = torch.stack(self.a)
        returns = self.returns
        adv = self.adv

        M = (N * T) // num_mb
        sampler = BatchSampler(SubsetRandomSampler(range(N * T)), M, drop_last=False)

        for indices in sampler:
            s_mb = s.view(-1, *s.size()[2:])[indices]
            p_mb = p.view(-1, *p.size()[2:])[indices]
            a_mb = a.view(-1, *a.size()[2:])[indices]
            returns_mb = returns.view(-1, *returns.size()[2:])[indices]
            adv_mb = adv.view(-1, *adv.size()[2:])[indices]

            yield s_mb, p_mb, a_mb, returns_mb, adv_mb
