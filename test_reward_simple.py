"""
简化测试：验证新奖励函数
"""
import numpy as np
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.envs.h2_res_env import H2RESEnv

# 创建固定的测试场景：充足的可再生能源
def create_test_scenario():
    """创建24小时的测试数据，确保有充足的可再生能源"""
    hours = 24
    
    # 固定高风速（确保风电满发）
    wind_speed = np.ones(hours) * 15.0  # 15 m/s，风电满发
    
    # 白天有光伏，晚上没有
    irradiance = np.array([0, 0, 0, 0, 0, 0,  # 0-5点：夜间
                           200, 400, 600, 800, 900, 1000,  # 6-11点：上升
                           1000, 1000, 900, 800, 600, 400,  # 12-17点：下降
                           200, 0, 0, 0, 0, 0])  # 18-23点：夜间
    
    # 固定温度
    temperature = np.ones(hours) * 25.0
    
    # 固定负荷（2000kW）
    load = np.ones(hours) * 2000.0
    
    return pd.DataFrame({
        'wind_speed': wind_speed,
        'irradiance': irradiance,
        'temperature': temperature,
        'load': load
    })

df_data = create_test_scenario()
env = H2RESEnv(df_data, df_data)

print("=" * 80)
print("  新奖励函数测试")
print("=" * 80)

# 测试1：连续运行电解槽
print("\n测试1：电解槽连续运行3小时（中午时段，光伏充足）")
print("-" * 80)
env.reset(start_step=10, init_soc=0.5, init_soch=0.5)  # 从10点开始（光伏900W/m2）

print(f"初始状态: 风电={env.current_p_wt:.0f}kW, 光伏={env.current_p_pv:.0f}kW, "
      f"负荷={env.current_load:.0f}kW, 剩余={(env.current_p_wt + env.current_p_pv - env.current_load):.0f}kW")

total_reward = 0
for i in range(3):
    # 请求电解槽高功率运行
    action = np.array([0.0, 0.9, 0.0])  # 电池0, 电解槽0.9→2700kW, 燃料电池0
    obs, reward, done, info = env.step(action)
    
    total_reward += reward
    
    print(f"步骤{i+1}: 电解槽={info['real_p_el']:.0f}kW, 制氢={info['h2_prod']:.2f}kg, "
          f"奖励={reward:.2f}, 连续小时={env.el_continuous_hours}")

print(f"总奖励: {total_reward:.2f}")

# 测试2：频繁启停
print("\n" + "=" * 80)
print("测试2：电解槽频繁启停（开-关-开）")
print("-" * 80)
env.reset(start_step=10, init_soc=0.5, init_soch=0.5)

total_reward = 0
for i in range(3):
    if i % 2 == 0:
        action = np.array([0.0, 0.9, 0.0])  # 开启
    else:
        action = np.array([0.0, -1.0, 0.0])  # 关闭
    
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    status = "开启" if info['real_p_el'] >= 300 else "关闭"
    print(f"步骤{i+1}: 电解槽={info['real_p_el']:.0f}kW ({status}), "
          f"奖励={reward:.2f}, 连续小时={env.el_continuous_hours}")

print(f"总奖励: {total_reward:.2f}")

# 测试3：制氢 vs 充电
print("\n" + "=" * 80)
print("测试3：制氢 vs 充电的奖励对比")
print("-" * 80)

# 只制氢
env.reset(start_step=10, init_soc=0.5, init_soch=0.5)
action = np.array([0.0, 0.9, 0.0])
obs, reward_h2, done, info = env.step(action)
print(f"只制氢: 电解槽={info['real_p_el']:.0f}kW, 奖励={reward_h2:.2f}")

# 只充电
env.reset(start_step=10, init_soc=0.5, init_soch=0.5)
action = np.array([0.9, -1.0, 0.0])
obs, reward_bat, done, info = env.step(action)
print(f"只充电: 电池={info['real_p_bat']:.0f}kW, 奖励={reward_bat:.2f}")

# 两者都做
env.reset(start_step=10, init_soc=0.5, init_soch=0.5)
action = np.array([0.4, 0.5, 0.0])
obs, reward_both, done, info = env.step(action)
print(f"两者都做: 电池={info['real_p_bat']:.0f}kW, 电解槽={info['real_p_el']:.0f}kW, 奖励={reward_both:.2f}")

print("\n" + "=" * 80)
print("  测试完成")
print("=" * 80)