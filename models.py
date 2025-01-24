import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


USE_CUDA = torch.cuda.is_available()

    
class MLP(nn.Module):
    def __init__(self, stat_dim, act_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(stat_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, act_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

class NatureCNN(nn.Module):
    def __init__(self, frames=4, act_dim=3):
        super(NatureCNN, self).__init__()
        self.network = nn.Sequential(
#             Scale(1 / 255),
            nn.Conv2d(frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim)
        )

    def forward(self, x):
        # x = torch.Tensor(x).to(device)
        return self.network(x)


def f(input):
    sign = torch.sign(input)
    return sign * (torch.sqrt(torch.abs(input)))

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale
    
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, sig0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_sig = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_sig = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.bias_mu = None
        self.reset_parameters(sig0)
        self.dist = torch.distributions.Normal(0, 1)
        self.weight = None
        self.bias = None
        #self.sample()

    def reset_parameters(self,sig0):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_sig.data = self.weight_sig.data.zero_() + sig0 / self.weight_mu.shape[1]

        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_sig.data.zero_()
            self.bias_sig.data = self.bias_sig.data.zero_() + sig0 / self.weight_mu.shape[1]

    def sample(self):
        size_in = self.in_features
        size_out = self.out_features
        noise_in = f(self.dist.sample((1, size_in))).cuda()
        noise_out = f(self.dist.sample((1, size_out))).cuda()
        self.weight = self.weight_mu + self.weight_sig * torch.mm(noise_out.t(), noise_in)
        #self.weight=torch.autograd.Variable(self.weight)
        if self.bias_mu is not None:
            self.bias = (self.bias_mu + self.bias_sig * noise_out).squeeze()

    def forward(self, input):
        #self.sample()
        if self.bias_mu is not None:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight)

    def randomness(self):
        size_in = self.in_features
        size_out = self.out_features
        return torch.abs(self.bias_sig.data/self.bias_mu.data).numpy().sum()/size_out#+torch.abs(self.weight_sig.data/self.weight_mu.data).numpy().sum()/(size_in*size_out)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class NoisyNet(nn.Module):
    def __init__(self, env, frames=4):
        super(NoisyNet, self).__init__()
        self.network = nn.Sequential(
            Scale(1 / 255),
            nn.Conv2d(frames, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            NoisyLinear(3136, 512),
            nn.ReLU(),
            NoisyLinear(512, env.action_space.n)
        )

    def forward(self, x):
        # x = torch.Tensor(x).to(device)
        return self.network(x)
    
    def sample(self):
        self.network[8].sample()
        self.network[10].sample()

class NoisyNet_MLP(nn.Module):
    def __init__(self, env):
        super(NoisyNet_MLP, self).__init__()
        self.env = env
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            NoisyLinear(256, env.action_space.n)
        )

    def forward(self, x):
        # x = torch.Tensor(x).to(device)
        x = torch.reshape(x, (-1, np.array(self.env.observation_space.shape).prod()))
        return self.network(x)
    
    def sample(self):
        self.network[4].sample()