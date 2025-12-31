"""
DQN Training Script for H2-RES System
Version: V8.0 - 增强观测空间（11维：历史+时间特征）+ 5³=125动作空间
环境改进：按比例分配 + 物理约束检查 + 历史特征 + 时间编码
"""

import sys
import numpy as np
import pandas as pd
import torch
from src.envs.h2_res_env import H2RESEnv
from src.algos.dqn import DQNAgent
import os
import time  # 引入时间模块用于统计耗时

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def generate_simple_load(hours):
    """生成简单的合成负荷数据"""
    time_idx = np.arange(hours)
    base_load = 2000  # kW
    daily_pattern = 1000 * (np.exp(-((time_idx % 24 - 9) ** 2) / 10) + np.exp(-((time_idx % 24 - 19) ** 2) / 10))
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


def train_dqn(num_episodes=20000, save_freq=2000):
    """训练DQN模型（V6.2优化版：5³=125动作空间 + 20000轮训练）"""
    
    # 强制使用GPU
    print("\n" + "=" * 30)
    print("      设备检测 (GPU/CPU)")
    print("=" * 30)
    if not torch.cuda.is_available():
        print("[WARNING] 未检测到GPU，将使用CPU训练（速度会很慢）。")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"[OK] PyTorch 版本: {torch.__version__}")
        print(f"[OK] 强制使用GPU训练")
        print(f"[OK] 显卡名称: {torch.cuda.get_device_name(0)}")
    print("=" * 30 + "\n")
    
    # 加载数据
    print("Loading data...")
    df_data = load_and_process_real_data()
    
    # 创建环境
    env = H2RESEnv(df_data, df_data)
    
    # 创建DQN智能体（V6.2优化版：5³=125动作空间）
    state_dim = env.observation_space.shape[0]
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=3,
        n_actions_per_dim=5,  # 125个动作（5³），提升精度
        lr=8e-4,  # 略微降低学习率，提升稳定性
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,  # 保持更多探索（0.01->0.05）
        epsilon_decay=0.9995,  # 更慢的衰减（0.995->0.9995）
        buffer_size=150000,  # 增加缓冲区（100000->150000）
        batch_size=256,
        target_update_freq=10
    )
    
    # 确保agent使用正确设备
    agent.q_network = agent.q_network.to(device)
    agent.target_network = agent.target_network.to(device)
    agent.device = device
    
    # ================= 关键参数设置 =================
    MAX_STEPS = 24  # [关键修复] 每个episode强制只跑24小时
    # ===============================================

    print("\n" + "=" * 60)
    print(f"  开始训练 DQN V8.0 模型 - 共 {num_episodes} 轮")
    print("=" * 60)
    print(f"环境版本: V8.0 (增强观测空间)")
    print(f"观测空间: {state_dim}维 (包含历史+时间特征)")
    print(f"Episode长度限制: {MAX_STEPS} 小时")
    print(f"数据长度: {len(df_data)} 小时")
    print(f"离散动作空间: {agent.total_actions} 个动作 (5^3=125)")
    print(f"Batch Size: {agent.batch_size}")
    print(f"Buffer Size: {agent.replay_buffer.capacity}")
    print(f"学习率: {agent.lr}")
    print(f"Epsilon衰减: {agent.epsilon_decay}")
    print(f"设备: {device}")
    print("=" * 60 + "\n")
    
    # 计算最大可用天数
    max_days = (len(df_data) - MAX_STEPS) // 24
    if max_days < 1:
        print("[ERROR] 数据长度不足 24 小时，无法训练。")
        return
    
    # 训练循环
    rewards_history = []
    h2_prod_history = []
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    start_time = time.time() # 记录开始时间
    
    for episode in range(num_episodes):
        # 随机采样起始点（滑动窗口）
        start_step = np.random.randint(0, len(df_data) - MAX_STEPS)
        
        # 随机初始状态，避开边界区域
        init_soc = np.random.uniform(0.4, 0.85) if episode > 0 else 0.5
        init_soch = np.random.uniform(0.35, 0.8) if episode > 0 else 0.5
        
        # 重置环境
        state = env.reset(start_step=start_step, init_soc=init_soc, init_soch=init_soch)
        
        episode_reward = 0
        episode_h2_prod = 0
        episode_loss = 0
        steps = 0
        
        done = False
        update_counter = 0
        
        # 优化：批量收集经验，减少Python循环开销
        while not done:
            # ================= [核心修复] =================
            # 强制截断，防止跑完整个数据集
            if steps >= MAX_STEPS:
                break
            # ============================================

            # 选择动作（GPU加速）
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 记录
            episode_reward += reward
            episode_h2_prod += info['h2_prod']
            steps += 1
            state = next_state
            
            # 优化：每4步更新一次（125动作需要更频繁更新）
            update_counter += 1
            if update_counter % 4 == 0 and len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.update()
                if loss > 0:
                    episode_loss += loss
        
        rewards_history.append(episode_reward)
        h2_prod_history.append(episode_h2_prod)
        
        # 改进的进度打印 - 每10轮打印一次
        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(rewards_history[-10:])
            avg_h2_10 = np.mean(h2_prod_history[-10:])
            avg_loss = episode_loss / steps if steps > 0 else 0
            
            # 估算剩余时间
            elapsed = time.time() - start_time
            speed = (episode + 1) / elapsed
            remaining_sec = (num_episodes - (episode + 1)) / speed
            
            print(f"Episode {episode+1:5d}/{num_episodes} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg(10): {avg_reward_10:8.2f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Speed: {speed:.1f} iter/s | "
                  f"ETA: {remaining_sec/60:.1f} min")
        
        # 每100轮打印详细统计
        if (episode + 1) % 100 == 0:
            avg_reward_100 = np.mean(rewards_history[-100:])
            avg_h2_100 = np.mean(h2_prod_history[-100:])
            print("-" * 60)
            print(f"[统计] 最近100轮平均: Reward={avg_reward_100:.2f}, H2={avg_h2_100:.2f} kg")
            print("-" * 60)
        
        # 保存模型
        if (episode + 1) % save_freq == 0:
            agent.save(f'results/dqn_checkpoint_ep{episode+1}.pth')
            print(f"Model saved at episode {episode+1}")
    
    # 保存最终模型
    print("\n" + "=" * 60)
    print("  训练完成！正在保存模型...")
    print("=" * 60)
    
    agent.save('results/dqn_checkpoint.pth')
    np.save('results/dqn_training_rewards.npy', rewards_history)
    np.save('results/dqn_training_h2prod.npy', h2_prod_history)
    
    # 打印最终统计
    final_avg_reward = np.mean(rewards_history[-100:])
    final_avg_h2 = np.mean(h2_prod_history[-100:])
    print(f"\n最终性能 (最近100轮平均):")
    print(f"  平均奖励: {final_avg_reward:.2f}")
    print(f"  平均制氢量: {final_avg_h2:.2f} kg/天")
    print(f"  最终Epsilon: {agent.epsilon:.3f}")
    print(f"\n模型已保存到: results/dqn_checkpoint.pth")
    print("=" * 60 + "\n")
    
    return agent


if __name__ == '__main__':
    train_dqn(num_episodes=20000, save_freq=2000)