"""
Strategy Comparison Visualization (Similar to Paper Figure 16)

对比不同策略（PSO、DQN、DDPG）在四个季节典型日的净氢气产量
Version: V12.0 - 支持V12环境（Agent-Driven Allocation）
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import torch
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 支持多版本环境
try:
    from src.envs.h2_res_env_v12 import H2RESEnv as H2RESEnvV12
    USE_V12 = True
except ImportError:
    USE_V12 = False

try:
    from src.envs.h2_res_env import H2RESEnv
except ImportError:
    H2RESEnv = None

from src.baselines.pso_baseline import PSOBaseline
from src.algos.dqn import DQNAgent
from src.algos.ddpg import DDPGAgent


def generate_simple_load(hours):
    """生成简单的合成负荷数据"""
    time = np.arange(hours)
    base_load = 2000  # kW
    daily_pattern = 1000 * (np.exp(-((time % 24 - 9) ** 2) / 10) + np.exp(-((time % 24 - 19) ** 2) / 10))
    load = base_load + daily_pattern + np.random.normal(0, 100, hours)
    return load


def read_data_robust(filepath):
    """智能读取数据：自动区分 Excel (.xlsx) 和 CSV (.csv)"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")
    
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        print(f"检测到 Excel 文件，正在读取: {os.path.basename(filepath)} ...")
        try:
            df = pd.read_excel(filepath, index_col=0, parse_dates=True)
            print(f"[OK] 成功读取 Excel 文件")
            return df
        except ImportError:
            raise ImportError("读取 .xlsx 文件需要 'openpyxl' 库。请运行: pip install openpyxl")
    else:
        encodings = ['utf-8', 'gbk', 'gb18030', 'ISO-8859-1', 'cp1252']
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True, encoding=enc)
                print(f"[OK] 成功使用编码 '{enc}' 读取文件")
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法读取 CSV 文件 {filepath}")


def load_and_process_real_data():
    """读取真实数据，提取气象参数，重采样为每小时"""
    DATA_DIR = os.path.join(current_dir, 'src', 'data')
    REAL_DATA_PATHS = {
        'wind': os.path.join(DATA_DIR, 'Wind farm site 1 (Nominal capacity-99MW).xlsx'),
        'solar': os.path.join(DATA_DIR, 'Solar station site 1 (Nominal capacity-50MW).xlsx')
    }
    
    print(f"正在加载真实数据 (from {DATA_DIR})...")
    
    # 读取数据
    df_wind_raw = read_data_robust(REAL_DATA_PATHS['wind'])
    df_solar_raw = read_data_robust(REAL_DATA_PATHS['solar'])
    
    # 清理列名
    df_wind_raw.columns = df_wind_raw.columns.str.strip()
    df_solar_raw.columns = df_solar_raw.columns.str.strip()
    
    # 智能查找列名
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
    
    if not (wind_col and solar_col and temp_col):
        raise ValueError(f"无法找到所需的列。请检查 Excel 列名。")
    
    print(f"映射列名:\n 风速 -> {wind_col}\n 光照 -> {solar_col}\n 温度 -> {temp_col}")
    
    # 提取数据并对齐
    wind_data = pd.to_numeric(df_wind_raw[wind_col], errors='coerce')
    solar_data = pd.to_numeric(df_solar_raw[solar_col], errors='coerce')
    raw_temp = df_wind_raw[temp_col] if temp_col in df_wind_raw.columns else df_solar_raw[temp_col]
    temp_data = pd.to_numeric(raw_temp, errors='coerce')
    
    df_merged = pd.DataFrame({
        'wind_speed': wind_data,
        'irradiance': solar_data,
        'temperature': temp_data
    })
    
    # 移除空值并排序
    df_merged = df_merged.dropna().sort_index()
    
    # 重采样为 1 小时均值
    print("正在重采样为每小时均值...")
    df_hourly = df_merged.resample('1H').mean().dropna()
    
    # 确保数据从 00:00 开始
    first_midnight_idx = np.where(df_hourly.index.hour == 0)[0]
    if len(first_midnight_idx) > 0:
        start_idx = first_midnight_idx[0]
        print(f"正在截取数据，确保从 {df_hourly.index[start_idx]} (00:00) 开始...")
        df_hourly = df_hourly.iloc[start_idx:]
    
    # 生成负荷
    print("生成匹配长度的简单合成负荷数据...")
    df_hourly['load'] = generate_simple_load(len(df_hourly))
    
    print(f"成功加载并处理真实数据! 最终数据点数: {len(df_hourly)} 小时")
    return df_hourly


def find_seasonal_high_re_days(df_weather):
    """
    找到四个季节的风光大发日
    
    Returns:
        季节典型日的起始索引列表 [(season_name, start_idx, date), ...]
    """
    # 计算每天的总可再生能源发电量
    n_days = len(df_weather) // 24
    daily_re = []
    
    for day in range(n_days):
        start_idx = day * 24
        end_idx = start_idx + 24
        
        day_data = df_weather.iloc[start_idx:end_idx]
        
        # 计算风电和光伏
        wind_power = day_data['wind_speed'].apply(lambda v: 
            4000 * ((v - 3) / (12 - 3)) ** 3 if 3 <= v < 12 
            else 4000 if v >= 12 else 0
        ).sum()
        
        solar_power = day_data['irradiance'].apply(lambda G:
            3000 * (G / 1000) if G > 0 else 0
        ).sum()
        
        total_re = wind_power + solar_power
        date = day_data.index[0].date()
        month = date.month
        
        daily_re.append({
            'day': day,
            'start_idx': start_idx,
            'date': date,
            'month': month,
            'total_re': total_re
        })
    
    daily_re_df = pd.DataFrame(daily_re)
    
    # 定义季节
    seasons = {
        'Spring': [3, 4, 5],    # 春季：3-5月
        'Summer': [6, 7, 8],    # 夏季：6-8月
        'Autumn': [9, 10, 11],  # 秋季：9-11月
        'Winter': [12, 1, 2]    # 冬季：12-2月
    }
    
    seasonal_days = []
    
    for season_name, months in seasons.items():
        # 筛选该季节的数据
        season_data = daily_re_df[daily_re_df['month'].isin(months)]
        
        if len(season_data) > 0:
            # 选择该季节中可再生能源发电量最高的一天
            best_day = season_data.loc[season_data['total_re'].idxmax()]
            seasonal_days.append((
                season_name,
                int(best_day['start_idx']),
                best_day['date'],
                best_day['total_re']
            ))
    
    return seasonal_days


def evaluate_strategy(env, strategy_name, agent_or_baseline, start_idx, num_steps=24):
    """
    评估单个策略在指定时间段的性能
    
    Args:
        env: 环境
        strategy_name: 策略名称 ('PSO', 'DQN', 'DDPG')
        agent_or_baseline: 智能体或基线策略
        start_idx: 起始时间步
        num_steps: 评估步数
    
    Returns:
        h2_prod_cumulative: 累积氢气产量 (kg)
    """
    obs = env.reset(start_step=start_idx)
    
    h2_prod_cumulative = [0.0]  # 累积氢气产量
    
    for step in range(num_steps):
        if strategy_name == 'PSO':
            # PSO基线策略
            p_gen = env.current_p_wt + env.current_p_pv
            p_load = env.current_load
            action = agent_or_baseline.get_action(obs, p_gen, p_load, env.SOC, env.SOCH)
        elif strategy_name == 'DQN':
            # DQN策略
            action = agent_or_baseline.select_action(obs, training=False)
        elif strategy_name == 'DDPG':
            # DDPG策略
            action = agent_or_baseline.select_action(obs, noise=False)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        obs, reward, done, info = env.step(action)
        
        # 累积氢气产量
        h2_prod_cumulative.append(h2_prod_cumulative[-1] + info['h2_prod'])
        
        if done:
            break
    
    return np.array(h2_prod_cumulative)


def plot_strategy_comparison(seasonal_results, save_path='results/strategy_comparison.png'):
    """
    绘制策略对比图（类似论文图16）
    
    Args:
        seasonal_results: 季节结果字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = {
        'PSO': '#2E7D32',      # 深绿色
        'DQN': '#1976D2',      # 深蓝色
        'DDPG': '#D32F2F'      # 深红色
    }
    
    alphas = {
        'PSO': 0.6,
        'DQN': 0.7,
        'DDPG': 0.8
    }
    
    for idx, (season_name, results) in enumerate(seasonal_results.items()):
        ax = axes[idx]
        
        time_hours = np.arange(len(results['PSO']))
        
        # 绘制堆叠面积图
        ax.fill_between(time_hours, 0, results['PSO'], 
                        color=colors['PSO'], alpha=alphas['PSO'], 
                        label='PSO', linewidth=2)
        ax.fill_between(time_hours, 0, results['DQN'], 
                        color=colors['DQN'], alpha=alphas['DQN'], 
                        label='DQN', linewidth=2)
        ax.fill_between(time_hours, 0, results['DDPG'], 
                        color=colors['DDPG'], alpha=alphas['DDPG'], 
                        label='DDPG', linewidth=2)
        
        # 设置标题和标签
        ax.set_title(f'({chr(97+idx)}) {season_name} Scenario\n'
                    f'Variation in net hydrogen production', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Time / h', fontsize=12)
        ax.set_ylabel('Net hydrogen production / kg', fontsize=12)
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置图例
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        # 设置坐标轴范围
        ax.set_xlim(0, 24)
        ax.set_ylim(0, max(results['DDPG'].max(), results['DQN'].max(), results['PSO'].max()) * 1.1)
        
        # 添加最终产量标注
        final_pso = results['PSO'][-1]
        final_dqn = results['DQN'][-1]
        final_ddpg = results['DDPG'][-1]
        
        ax.text(0.98, 0.02, 
                f'Final Production:\n'
                f'PSO: {final_pso:.1f} kg\n'
                f'DQN: {final_dqn:.1f} kg\n'
                f'DDPG: {final_ddpg:.1f} kg',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Strategy comparison plot saved to {save_path}")
    plt.close()


def main(version='v12'):
    """主函数：运行策略对比实验
    
    Args:
        version: 'v12' for V12 environment, 'standard' for standard environment
    """
    
    print("=" * 80)
    print(f"Strategy Comparison Experiment ({version.upper()})")
    print("=" * 80)
    
    # 选择环境
    if version == 'v12':
        if not USE_V12:
            print("[ERROR] V12环境未找到，请确保 src/envs/h2_res_env_v12.py 存在")
            return
        EnvClass = H2RESEnvV12
        model_suffix = '_v12'
    else:
        if H2RESEnv is None:
            print("[ERROR] 标准环境未找到")
            return
        EnvClass = H2RESEnv
        model_suffix = ''
    
    # 1. 加载数据
    print("\n[1/6] Loading data...")
    df_data = load_and_process_real_data()
    env = EnvClass(df_data, df_data)
    
    # 2. 找到四个季节的风光大发日
    print("\n[2/6] Finding seasonal high-RE days...")
    seasonal_days = find_seasonal_high_re_days(df_data)
    
    print("\nSelected seasonal days:")
    for season_name, start_idx, date, total_re in seasonal_days:
        print(f"  {season_name}: {date} (Start index: {start_idx}, Total RE: {total_re:.0f} kWh)")
    
    # 3. 加载/创建策略
    print("\n[3/6] Loading strategies...")
    
    # PSO基线
    pso_baseline = PSOBaseline(env)
    print("  [OK] PSO baseline created")
    
    # DQN
    dqn_agent = None
    state_dim = env.observation_space.shape[0]
    
    # 根据版本选择DQN模型路径
    if version == 'v12':
        dqn_paths = [
            'results/dqn_v12.pth',
            f'results/dqn{model_suffix}.pth'
        ]
        train_script = 'python train_dqn_v12.py'
    else:
        dqn_paths = [
            'results/dqn_checkpoint.pth',
            'results/dqn_fast_checkpoint.pth'
        ]
        train_script = 'python train_dqn_fast.py'
    
    # 尝试加载DQN模型
    for path in dqn_paths:
        if os.path.exists(path):
            try:
                dqn_agent = DQNAgent(state_dim=state_dim, action_dim=3, n_actions_per_dim=5)
                dqn_agent.load(path)
                print(f"  [OK] DQN agent loaded from {path} (obs_dim={state_dim}, 5^3=125 actions)")
                break
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"  [SKIP] DQN checkpoint {path} incompatible (dimension mismatch)")
                    continue
                else:
                    raise
    
    if dqn_agent is None:
        print(f"  [SKIP] DQN checkpoint not found. Please train DQN first using: {train_script}")
    
    # DDPG
    ddpg_agent = None
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 根据版本选择模型路径
    if version == 'v12':
        ddpg_paths = [
            f'results/ddpg{model_suffix}.pth',
            'results/ddpg_v12.pth'
        ]
        train_script = 'python train_v12.py'
    else:
        ddpg_paths = [
            'results/ddpg_improved.pth',
            'results/ddpg_checkpoint.pth',
            'results/ddpg_cheakpoint.pth'
        ]
        train_script = 'python train_improved.py'
    
    # 尝试加载DDPG模型
    for path in ddpg_paths:
        if os.path.exists(path):
            ddpg_agent = DDPGAgent(state_dim, action_dim, device="cpu")
            ddpg_agent.load(path)
            print(f"  [OK] DDPG agent loaded from {path} (obs_dim={state_dim})")
            break
    
    if ddpg_agent is None:
        print(f"  [SKIP] DDPG checkpoint not found. Please train DDPG first using: {train_script}")
    
    # 4. 评估所有策略
    print("\n[4/6] Evaluating strategies on seasonal days...")
    seasonal_results = {}
    
    for season_name, start_idx, date, total_re in seasonal_days:
        print(f"\n  Evaluating {season_name} ({date})...")
        
        results = {}
        
        # PSO
        results['PSO'] = evaluate_strategy(env, 'PSO', pso_baseline, start_idx)
        print(f"    PSO: {results['PSO'][-1]:.2f} kg")
        
        # DQN
        if dqn_agent is not None:
            results['DQN'] = evaluate_strategy(env, 'DQN', dqn_agent, start_idx)
            print(f"    DQN: {results['DQN'][-1]:.2f} kg")
        else:
            results['DQN'] = np.zeros_like(results['PSO'])
            print(f"    DQN: N/A (not trained)")
        
        # DDPG
        if ddpg_agent is not None:
            results['DDPG'] = evaluate_strategy(env, 'DDPG', ddpg_agent, start_idx)
            print(f"    DDPG: {results['DDPG'][-1]:.2f} kg")
        else:
            results['DDPG'] = np.zeros_like(results['PSO'])
            print(f"    DDPG: N/A (not trained)")
        
        seasonal_results[season_name] = results
    
    # 5. 绘制对比图
    print("\n[5/6] Plotting comparison figure...")
    os.makedirs('results', exist_ok=True)
    save_path = f'results/strategy_comparison{model_suffix}.png'
    plot_strategy_comparison(seasonal_results, save_path)
    
    # 6. 保存数值结果
    print("\n[6/6] Saving numerical results...")
    results_path = f'results/strategy_comparison_results{model_suffix}.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("Strategy Comparison Results\n")
        f.write("=" * 80 + "\n\n")
        
        for season_name, results in seasonal_results.items():
            f.write(f"{season_name} Scenario:\n")
            f.write(f"  PSO:  {results['PSO'][-1]:.2f} kg\n")
            f.write(f"  DQN:  {results['DQN'][-1]:.2f} kg\n")
            f.write(f"  DDPG: {results['DDPG'][-1]:.2f} kg\n")
            
            if results['DDPG'][-1] > 0:
                improvement_vs_pso = (results['DDPG'][-1] - results['PSO'][-1]) / results['PSO'][-1] * 100
                improvement_vs_dqn = (results['DDPG'][-1] - results['DQN'][-1]) / results['DQN'][-1] * 100 if results['DQN'][-1] > 0 else 0
                f.write(f"  DDPG vs PSO: {improvement_vs_pso:+.2f}%\n")
                f.write(f"  DDPG vs DQN: {improvement_vs_dqn:+.2f}%\n")
            f.write("\n")
    
    print("\n" + "=" * 80)
    print(f"Strategy comparison completed! ({version.upper()})")
    print("Results saved to:")
    print(f"  - {save_path}")
    print(f"  - {results_path}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='策略对比实验')
    parser.add_argument('--version', type=str, default='v12',
                       choices=['v12', 'standard'],
                       help='环境版本: v12 (V12环境) 或 standard (标准环境)')
    args = parser.parse_args()
    
    main(version=args.version)