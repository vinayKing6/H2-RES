import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    DDPG Critic Network.
    Approximates the Q-value function Q(s, a).
    Structure: Input(state + action) -> FC(256) -> ... -> Q-value
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Concatenate state and action in the first layer or after processing
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x