"""
测试新奖励函数：验证电解槽连续运行奖励和制氢激励
"""
import numpy as np
import pandas as pd
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.envs.h2_res_env import H2RESEnv

# 生成简单的合成数据用于测试
def generate_test_data(hours=8760):
    time = np.arange(hours)
    
    # 风速
    wind_speed = 6.0 + 2.0 * np.sin(2 * np.pi * time / 8760) + np.random.weibull(2.0, hours) * 2.0
    wind_speed = np.clip(wind_speed, 0, 25)
    
    # 光照
    day_progress = (time % 24) / 24.0
    seasonal_solar = 0.8 + 0.2 * np.cos(2 * np.pi * time / 8760)
    solar_profile = np.maximum(0, -np.cos(2 * np.pi * day_progress))
    irradiance = 1000 * solar_profile * seasonal_solar * np.random.uniform(0.8, 1.0, hours)
    
    # 温度
    temp = 15 + 10 * np.sin(2 * np.pi * time / 8760 - np.pi / 2) + 5 * np.cos(2 * np.pi * day_progress)
    
    # 负荷
    base_load = 2000
    daily_pattern = 1000 * (np.exp(-((time % 24 - 9) ** 2) / 10) + np.exp(-((time % 24 - 19) ** 2) / 10))
    load = base_load + daily_pattern + np.random.normal(0, 100, hours)
    
    return pd.DataFrame({
        'wind_speed': wind_speed,
        'irradiance': irradiance,
        'temperature': temp,
        'load': load
    })

# 加载数据
df_data = generate_test_data()
env = H2RESEnv(df_data, df_data)

print("=" * 80)
print("  测试新奖励函数：电解槽连续运行 vs 频繁启停")
print("=" * 80)

# 场景1：电解槽连续运行5小时
print("\n场景1：电解槽连续运行5小时")
print("-" * 80)
env.reset(start_step=1000, init_soc=0.5, init_soch=0.5)

total_reward_continuous = 0
total_h2_continuous = 0

for i in range(5):
    # 持续请求电解槽运行（高功率）
    action = np.array([0.0, 0.8, 0.0])  # 电池0, 电解槽高, 燃料电池0
    obs, reward, done, info = env.step(action)
    
    total_reward_continuous += reward
    total_h2_continuous += info['h2_prod']
    
    print(f"  步骤{i+1}: 电解槽={info['real_p_el']:.0f}kW, "
          f"制氢={info['h2_prod']:.2f}kg, 奖励={reward:.2f}, "
          f"连续小时数={env.el_continuous_hours}")

print(f"\n总奖励: {total_reward_continuous:.2f}")
print(f"总制氢: {total_h2_continuous:.2f} kg")

# 场景2：电解槽频繁启停（每小时开关一次）
print("\n" + "=" * 80)
print("场景2：电解槽频繁启停（5次启停）")
print("-" * 80)
env.reset(start_step=1000, init_soc=0.5, init_soch=0.5)

total_reward_frequent = 0
total_h2_frequent = 0

for i in range(5):
    if i % 2 == 0:
        # 开启电解槽
        action = np.array([0.0, 0.8, 0.0])
    else:
        # 关闭电解槽
        action = np.array([0.0, -1.0, 0.0])  # 请求0功率
    
    obs, reward, done, info = env.step(action)
    
    total_reward_frequent += reward
    total_h2_frequent += info['h2_prod']
    
    status = "开启" if info['real_p_el'] >= 300 else "关闭"
    print(f"  步骤{i+1}: 电解槽={info['real_p_el']:.0f}kW ({status}), "
          f"制氢={info['h2_prod']:.2f}kg, 奖励={reward:.2f}, "
          f"连续小时数={env.el_continuous_hours}")

print(f"\n总奖励: {total_reward_frequent:.2f}")
print(f"总制氢: {total_h2_frequent:.2f} kg")

# 对比分析
print("\n" + "=" * 80)
print("  对比分析")
print("=" * 80)
print(f"连续运行总奖励: {total_reward_continuous:.2f}")
print(f"频繁启停总奖励: {total_reward_frequent:.2f}")
print(f"奖励差异: {total_reward_continuous - total_reward_frequent:.2f}")
print(f"\n连续运行制氢: {total_h2_continuous:.2f} kg")
print(f"频繁启停制氢: {total_h2_frequent:.2f} kg")
print(f"制氢差异: {total_h2_continuous - total_h2_frequent:.2f} kg")

if total_reward_continuous > total_reward_frequent:
    print("\n✓ 成功：连续运行获得更高奖励，Agent会学习避免频繁启停")
else:
    print("\n✗ 失败：频繁启停获得更高奖励，需要调整奖励函数")

# 场景3：测试制氢奖励 vs 电池充电奖励
print("\n" + "=" * 80)
print("场景3：制氢 vs 电池充电的奖励对比")
print("=" * 80)

# 只制氢
env.reset(start_step=1000, init_soc=0.5, init_soch=0.5)
action = np.array([0.0, 0.8, 0.0])  # 只电解槽
obs, reward_h2, done, info = env.step(action)
print(f"只制氢: 电解槽={info['real_p_el']:.0f}kW, 奖励={reward_h2:.2f}")

# 只充电池
env.reset(start_step=1000, init_soc=0.5, init_soch=0.5)
action = np.array([0.8, -1.0, 0.0])  # 只电池
obs, reward_bat, done, info = env.step(action)
print(f"只充电池: 电池={info['real_p_bat']:.0f}kW, 奖励={reward_bat:.2f}")

# 两者都做
env.reset(start_step=1000, init_soc=0.5, init_soch=0.5)
action = np.array([0.4, 0.4, 0.0])  # 电池+电解槽
obs, reward_both, done, info = env.step(action)
print(f"两者都做: 电池={info['real_p_bat']:.0f}kW, 电解槽={info['real_p_el']:.0f}kW, 奖励={reward_both:.2f}")

print("\n" + "=" * 80)
print("  测试完成")
print("=" * 80)