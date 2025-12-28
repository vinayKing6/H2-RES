import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    DDPG Actor Network.
    Approximates the deterministic policy function mu(s).
    Structure: Input(5) -> FC(256) -> ReLU -> FC(256) -> ReLU -> Output(3) -> Tanh
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # Tanh output to bound actions between [-1, 1]
        x = torch.tanh(self.l3(x))
        return x