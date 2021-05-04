import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in first and second hidden layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc0 = nn.Linear(state_size, fc_units)
        self.bn0 = nn.BatchNorm1d(fc_units)
        self.fc1 = nn.Linear(fc_units, fc_units)
        # BN to keep the values flowing the network small, might also be good to do it for the critic
        self.bn1 = nn.BatchNorm1d(fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        #self.bn2 = nn.BatchNorm1d(action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc0.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.leaky_relu(self.fc0(state))
        # to keep the values flowing the network small and avoid dimishing them in one step via tanh
        x = self.bn0(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)  # same
        x = self.fc2(x)
        #x = self.bn2(x)
        return torch.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        #self.bn_fcs1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        #self.bn_fc2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)
