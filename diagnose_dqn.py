"""
DQN训练诊断脚本
分析DQN训练效果和对比PSO/DDPG的差异
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from src.algos.dqn import DQNAgent
from src.envs.h2_res_env import H2RESEnv
import pandas as pd
import os

# 加载训练历史
rewards = np.load('results/dqn_training_rewards.npy')
h2_prod = np.load('results/dqn_training_h2prod.npy')

print("=" * 80)
print("DQN训练诊断报告")
print("=" * 80)

print(f"\n训练轮数: {len(rewards)}")
print(f"最终100轮平均奖励: {np.mean(rewards[-100:]):.2f}")
print(f"最终100轮平均制氢量: {np.mean(h2_prod[-100:]):.2f} kg/天")
print(f"最大奖励: {np.max(rewards):.2f}")
print(f"最大制氢量: {np.max(h2_prod):.2f} kg/天")

# 绘制训练曲线
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 奖励曲线
axes[0].plot(rewards, alpha=0.3, label='Raw')
window = 100
axes[0].plot(np.convolve(rewards, np.ones(window)/window, mode='valid'), 
             label=f'{window}-episode MA', linewidth=2)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].set_title('DQN Training Rewards')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 制氢量曲线
axes[1].plot(h2_prod, alpha=0.3, label='Raw')
axes[1].plot(np.convolve(h2_prod, np.ones(window)/window, mode='valid'), 
             label=f'{window}-episode MA', linewidth=2)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('H2 Production (kg/day)')
axes[1].set_title('DQN Training H2 Production')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/dqn_training_diagnosis.png', dpi=300)
print(f"\n训练曲线已保存到: results/dqn_training_diagnosis.png")

# 分析收敛情况
print("\n" + "=" * 80)
print("收敛分析")
print("=" * 80)

# 计算不同阶段的平均值
stages = [
    ("前1000轮", rewards[:1000], h2_prod[:1000]),
    ("中期(4000-5000)", rewards[4000:5000], h2_prod[4000:5000]),
    ("后期(9000-10000)", rewards[9000:], h2_prod[9000:])
]

for stage_name, stage_rewards, stage_h2 in stages:
    print(f"\n{stage_name}:")
    print(f"  平均奖励: {np.mean(stage_rewards):.2f}")
    print(f"  平均制氢量: {np.mean(stage_h2):.2f} kg/天")

# 检查是否收敛
final_100 = np.mean(h2_prod[-100:])
mid_100 = np.mean(h2_prod[4900:5000])
improvement = (final_100 - mid_100) / mid_100 * 100 if mid_100 > 0 else 0

print(f"\n中期到后期改进: {improvement:+.2f}%")

if abs(improvement) < 5:
    print("⚠️  警告：训练可能已经收敛或停滞")
elif improvement < -10:
    print("❌ 警告：性能下降，可能过拟合或训练不稳定")
else:
    print("✅ 训练仍在改进")

print("\n" + "=" * 80)
print("对比分析：为什么DQN表现差？")
print("=" * 80)

print("\n当前结果:")
print("  PSO:  795.55 kg (规则基线)")
print("  DDPG: 646.80 kg (连续动作)")
print("  DQN:  381.82 kg (离散动作)")

print("\n可能原因分析:")
print("1. 动作空间离散化损失")
print("   - DQN只有27个离散动作(3^3)")
print("   - DDPG有无限连续动作")
print("   - 离散化可能错过最优动作")

print("\n2. 训练环境差异")
print("   - 检查DQN是否使用了完整的H2RESEnv")
print("   - 检查奖励函数是否一致")

print("\n3. 探索-利用平衡")
print(f"   - 最终epsilon: 检查模型文件")

# 加载模型检查epsilon
try:
    checkpoint = torch.load('results/dqn_checkpoint.pth', map_location='cpu')
    if 'epsilon' in checkpoint:
        print(f"   - 实际epsilon: {checkpoint['epsilon']:.4f}")
        if checkpoint['epsilon'] > 0.1:
            print("   ⚠️  epsilon过高，探索过多")
except Exception as e:
    print(f"   - 无法加载模型: {e}")

print("\n4. 网络容量")
print("   - DQN网络: 256-256-27")
print("   - DDPG网络: 256-256-3 (Actor)")
print("   - 网络容量相似，不是主要问题")

print("\n" + "=" * 80)
print("建议")
print("=" * 80)
print("1. 增加动作空间精度: 3^3=27 → 5^3=125")
print("2. 延长训练时间: 10000 → 20000轮")
print("3. 调整奖励函数权重")
print("4. 使用优先经验回放(PER)")
print("=" * 80)