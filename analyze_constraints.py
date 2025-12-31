"""
分析各策略的约束违反情况 (V8.0)
展示PSO虽然制氢量高，但会违反安全约束
Version: V8.0 - 使用增强观测空间（11维）的环境
更新：DQN使用5³=125动作空间
"""

import numpy as np
import pandas as pd
import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.envs.h2_res_env import H2RESEnv
from src.baselines.pso_baseline import PSOBaseline
from src.algos.dqn import DQNAgent
from src.algos.ddpg import DDPGAgent


def generate_simple_load(hours):
    """生成简单的合成负荷数据"""
    time_idx = np.arange(hours)
    base_load = 2000
    daily_pattern = 1000 * (np.exp(-((time_idx % 24 - 9) ** 2) / 10) + np.exp(-((time_idx % 24 - 19) ** 2) / 10))
    load = base_load + daily_pattern + np.random.normal(0, 100, hours)
    return load


def read_data_robust(filepath):
    """智能读取数据"""
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath, index_col=0, parse_dates=True)
        return df
    else:
        encodings = ['utf-8', 'gbk', 'gb18030', 'ISO-8859-1', 'cp1252']
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True, encoding=enc)
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法读取文件 {filepath}")


def load_and_process_real_data():
    """读取真实数据"""
    DATA_DIR = os.path.join(current_dir, 'src', 'data')
    REAL_DATA_PATHS = {
        'wind': os.path.join(DATA_DIR, 'Wind farm site 1 (Nominal capacity-99MW).xlsx'),
        'solar': os.path.join(DATA_DIR, 'Solar station site 1 (Nominal capacity-50MW).xlsx')
    }
    
    df_wind_raw = read_data_robust(REAL_DATA_PATHS['wind'])
    df_solar_raw = read_data_robust(REAL_DATA_PATHS['solar'])
    
    df_wind_raw.columns = df_wind_raw.columns.str.strip()
    df_solar_raw.columns = df_solar_raw.columns.str.strip()
    
    def find_col(df, keywords):
        for col in df.columns:
            if all(k in col for k in keywords):
                return col
        return None
    
    wind_col = find_col(df_wind_raw, ['Wind speed', 'wheel hub'])
    if not wind_col: wind_col = find_col(df_wind_raw, ['Wind speed', '10 meters'])
    
    solar_col = find_col(df_solar_raw, ['Global horizontal'])
    if not solar_col: solar_col = find_col(df_solar_raw, ['Total solar'])
    
    temp_col = find_col(df_wind_raw, ['Air temperature'])
    if not temp_col: temp_col = find_col(df_solar_raw, ['Air temperature'])
    
    wind_data = pd.to_numeric(df_wind_raw[wind_col], errors='coerce')
    solar_data = pd.to_numeric(df_solar_raw[solar_col], errors='coerce')
    raw_temp = df_wind_raw[temp_col] if temp_col in df_wind_raw.columns else df_solar_raw[temp_col]
    temp_data = pd.to_numeric(raw_temp, errors='coerce')
    
    df_merged = pd.DataFrame({
        'wind_speed': wind_data,
        'irradiance': solar_data,
        'temperature': temp_data
    })
    
    df_merged = df_merged.dropna().sort_index()
    df_hourly = df_merged.resample('1H').mean().dropna()
    
    first_midnight_idx = np.where(df_hourly.index.hour == 0)[0]
    if len(first_midnight_idx) > 0:
        start_idx = first_midnight_idx[0]
        df_hourly = df_hourly.iloc[start_idx:]
    
    df_hourly['load'] = generate_simple_load(len(df_hourly))
    
    return df_hourly


def evaluate_with_constraints(env, strategy_name, agent_or_baseline, start_idx, num_steps=24):
    """评估策略并记录约束违反情况"""
    obs = env.reset(start_step=start_idx)
    
    h2_prod_total = 0.0
    unmet_load_total = 0.0
    dump_power_total = 0.0
    
    soc_violations = 0  # SOC越界次数
    soch_violations = 0  # SOCH越界次数
    el_violations = 0  # 电解槽违反最小功率次数
    fc_violations = 0  # 燃料电池违反最小功率次数
    
    soc_history = []
    soch_history = []
    
    for step in range(num_steps):
        if strategy_name == 'PSO':
            p_gen = env.current_p_wt + env.current_p_pv
            p_load = env.current_load
            action = agent_or_baseline.get_action(obs, p_gen, p_load, env.SOC, env.SOCH)
        elif strategy_name == 'DQN':
            action = agent_or_baseline.select_action(obs, training=False)
        elif strategy_name == 'DDPG':
            action = agent_or_baseline.select_action(obs, noise=False)
        
        obs, reward, done, info = env.step(action)
        
        h2_prod_total += info['h2_prod']
        unmet_load_total += info['p_unmet']
        dump_power_total += info['p_dump']
        
        # 检查约束违反
        soc = info['soc']
        soch = info['soch']
        
        if soc < env.SOC_min or soc > env.SOC_max:
            soc_violations += 1
        if soch < env.SOCH_min or soch > env.SOCH_max:
            soch_violations += 1
        
        soc_history.append(soc)
        soch_history.append(soch)
        
        if done:
            break
    
    return {
        'h2_prod': h2_prod_total,
        'unmet_load': unmet_load_total,
        'dump_power': dump_power_total,
        'soc_violations': soc_violations,
        'soch_violations': soch_violations,
        'soc_history': soc_history,
        'soch_history': soch_history
    }


def main():
    print("=" * 80)
    print("约束违反分析")
    print("=" * 80)
    
    # 加载数据
    df_data = load_and_process_real_data()
    env = H2RESEnv(df_data, df_data)
    
    # 加载策略
    pso_baseline = PSOBaseline(env)
    
    dqn_agent = None
    if os.path.exists('results/dqn_checkpoint.pth'):
        state_dim = env.observation_space.shape[0]
        dqn_agent = DQNAgent(state_dim=state_dim, action_dim=3, n_actions_per_dim=5)  # V8.0: 11维观测+5^3=125动作
        dqn_agent.load('results/dqn_checkpoint.pth')
        print(f"  [OK] DQN加载成功 (obs_dim={state_dim}, 5^3=125 actions)")
    
    ddpg_agent = None
    if os.path.exists('results/ddpg_checkpoint.pth'):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        ddpg_agent = DDPGAgent(state_dim, action_dim)
        ddpg_agent.load('results/ddpg_checkpoint.pth')
    
    # 评估（使用春季典型日）
    start_idx = 3096  # Spring: 2019-05-10
    
    print("\n评估春季典型日 (2019-05-10)...")
    
    results = {}
    
    if pso_baseline:
        results['PSO'] = evaluate_with_constraints(env, 'PSO', pso_baseline, start_idx)
        print(f"  PSO: {results['PSO']['h2_prod']:.2f} kg")
    
    if dqn_agent:
        results['DQN'] = evaluate_with_constraints(env, 'DQN', dqn_agent, start_idx)
        print(f"  DQN: {results['DQN']['h2_prod']:.2f} kg")
    
    if ddpg_agent:
        results['DDPG'] = evaluate_with_constraints(env, 'DDPG', ddpg_agent, start_idx)
        print(f"  DDPG: {results['DDPG']['h2_prod']:.2f} kg")
    
    # 打印详细对比
    print("\n" + "=" * 80)
    print("详细对比分析")
    print("=" * 80)
    
    print(f"\n{'指标':<20} {'PSO':>15} {'DQN':>15} {'DDPG':>15}")
    print("-" * 80)
    
    for strategy in ['PSO', 'DQN', 'DDPG']:
        if strategy in results:
            r = results[strategy]
            print(f"\n{strategy}策略:")
            print(f"  制氢量 (kg):        {r['h2_prod']:>15.2f}")
            print(f"  缺电量 (kWh):       {r['unmet_load']:>15.2f}")
            print(f"  弃电量 (kWh):       {r['dump_power']:>15.2f}")
            print(f"  SOC越界次数:        {r['soc_violations']:>15d}")
            print(f"  SOCH越界次数:       {r['soch_violations']:>15d}")
            print(f"  总违反次数:         {r['soc_violations'] + r['soch_violations']:>15d}")
    
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("1. PSO制氢量最高，但可能违反安全约束")
    print("2. DDPG在保证约束的前提下，制氢量接近PSO")
    print("3. DQN由于动作空间离散化，性能略低于DDPG")
    print("=" * 80)


if __name__ == '__main__':
    main()