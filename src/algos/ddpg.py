import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from src.models.ddpg_actor import Actor
from src.models.ddpg_critic import Critic


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.ptr = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim, action_dim,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001,
                 noise_scale=0.2):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.action_dim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()

        if noise:
            # Gaussian Noise as per paper preference (Ornstein-Uhlenbeck also common but paper chose 0.2 scale)
            action += np.random.normal(0, self.noise_scale, size=self.action_dim)

        return np.clip(action, -1, 1)

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # 1. Update Critic
        # Target Q = r + gamma * Q'(s', u'(s'))
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Update Actor
        # Maximize Q(s, u(s)) -> Minimize -Q
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Soft Update Targets (Eq 41-42)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])