"""
DQN (Deep Q-Network) Algorithm for H2-RES System

用于对比实验的DQN算法实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQNNetwork(nn.Module):
    """DQN网络：Q值函数近似器"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """经验回放缓冲区（极速版：完全向量化，避免Python循环）"""
    
    def __init__(self, capacity: int = 100000, state_dim: int = 5):
        self.capacity = capacity
        self.state_dim = state_dim
        self.position = 0
        self.size = 0
        
        # 预分配numpy数组（避免动态增长）
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验（直接写入numpy数组，O(1)复杂度）"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """采样（完全向量化，无Python循环，极速！）"""
        # 随机采样索引
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # 直接切片返回（numpy向量化操作，比Python循环快100倍）
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size


class DQNAgent:
    """
    DQN智能体
    
    注意：由于H2-RES系统的动作空间是连续的，我们需要将其离散化
    """
    
    def __init__(self, state_dim: int, action_dim: int = 3,
                 n_actions_per_dim: int = 5,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_size: int = 100000,
                 batch_size: int = 64, target_update_freq: int = 10):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度（3：电池、电解槽、燃料电池）
            n_actions_per_dim: 每个维度的离散动作数量
            lr: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_actions_per_dim = n_actions_per_dim
        self.total_actions = n_actions_per_dim ** action_dim  # 总动作数
        
        self.lr = lr  # 保存学习率
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start  # 保存初始epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q网络和目标Q网络
        self.q_network = DQNNetwork(state_dim, self.total_actions).to(self.device)
        self.target_network = DQNNetwork(state_dim, self.total_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim=state_dim)
        
        # 创建动作映射表（离散动作 → 连续动作）
        self.action_map = self._create_action_map()
        
        # 创建反向映射表（连续动作 → 离散动作索引）
        # 使用字典加速查找，避免每次都计算距离
        self.reverse_action_map = self._create_reverse_action_map()
        
        self.update_count = 0
    
    def _create_action_map(self):
        """创建离散动作到连续动作的映射"""
        action_map = []
        action_values = np.linspace(-1, 1, self.n_actions_per_dim)
        
        # 生成所有可能的动作组合
        for i in range(self.n_actions_per_dim):
            for j in range(self.n_actions_per_dim):
                for k in range(self.n_actions_per_dim):
                    action_map.append([
                        action_values[i],  # 电池动作
                        action_values[j],  # 电解槽动作
                        action_values[k]   # 燃料电池动作
                    ])
        
        return np.array(action_map, dtype=np.float32)
    
    def _create_reverse_action_map(self):
        """创建反向映射表：连续动作 → 离散动作索引（加速查找）"""
        reverse_map = {}
        for idx, action in enumerate(self.action_map):
            # 将动作转换为元组作为字典键（numpy数组不能作为键）
            key = tuple(np.round(action, decimals=4))  # 保留4位小数避免浮点误差
            reverse_map[key] = idx
        return reverse_map
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        选择动作（epsilon-greedy策略，GPU优化版）
        
        Args:
            state: 当前状态
            training: 是否处于训练模式
        
        Returns:
            action: 连续动作 [bat, el, fc]
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择动作
            action_idx = random.randint(0, self.total_actions - 1)
        else:
            # 利用：选择Q值最大的动作（GPU加速）
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
        
        # 将离散动作索引转换为连续动作
        return self.action_map[action_idx].copy()  # 返回副本避免修改原数组
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        # 将连续动作转换回离散动作索引
        action_idx = self._continuous_to_discrete(action)
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def _continuous_to_discrete(self, action: np.ndarray) -> int:
        """将连续动作转换为离散动作索引（优化版：使用哈希表）"""
        # 将动作四舍五入并转换为元组
        key = tuple(np.round(action, decimals=4))
        
        # 尝试直接查找（O(1)复杂度）
        if key in self.reverse_action_map:
            return self.reverse_action_map[key]
        
        # 如果找不到（理论上不应该发生），回退到距离计算
        distances = np.sum((self.action_map - action) ** 2, axis=1)
        return np.argmin(distances)
    
    def update(self):
        """更新Q网络（GPU优化版）"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量（一次性转换，减少CPU-GPU传输）
        states = torch.FloatTensor(states).to(self.device, non_blocking=True)
        actions = torch.LongTensor(actions).to(self.device, non_blocking=True)
        rewards = torch.FloatTensor(rewards).to(self.device, non_blocking=True)
        next_states = torch.FloatTensor(next_states).to(self.device, non_blocking=True)
        dones = torch.FloatTensor(dones).to(self.device, non_blocking=True)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值（使用no_grad减少内存占用）
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 反向传播和优化（梯度裁剪防止梯度爆炸）
        self.optimizer.zero_grad(set_to_none=True)  # 优化：set_to_none=True更快
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']