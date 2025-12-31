"""
DDPG Training Script for H2-RES System (V12: Gemini's Approach)
Version: V12 - Agent-Driven Allocation (完整实施Gemini建议)
核心改进：
1. ✅ 移除"负荷优先"硬约束
2. ✅ Agent自由分配功率（负荷、制氢、储能）
3. ✅ 环境只负责总功率约束（按比例缩放）
4. ✅ 负荷通过惩罚引导（软约束）
5. ✅ 让Agent学习最优平衡策略
"""

import sys
import os
import pandas as pd
import numpy as np
import torch

# --- 核心修复：将当前脚本所在的目录加入 Python 搜索路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# -----------------------------------------------------

# 导入V12环境
from src.envs.h2_res_env_v12 import H2RESEnv
from src.algos.ddpg import DDPGAgent, ReplayBuffer

# --- Hyperparameters (V12优化版) ---
MAX_EPISODES = 23000  # 10000轮训练
MAX_STEPS = 24  # 24 hours per episode
BATCH_SIZE = 256  # 大批次，减少梯度方差
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
REPLAY_SIZE = 1000

# --- 真实数据配置 ---
USE_REAL_DATA = True
DATA_DIR = os.path.join(current_dir, 'src', 'data')
REAL_DATA_PATHS = {
    'wind': os.path.join(DATA_DIR, 'Wind farm site 1 (Nominal capacity-99MW).xlsx'),
    'solar': os.path.join(DATA_DIR, 'Solar station site 1 (Nominal capacity-50MW).xlsx')
}


# ✨ V12改进：固定起始点调度器（保留V11的优点）
class FixedStartScheduler:
    """
    固定起始点调度器：使用4个固定的季节起始点
    
    核心改进：
    - 前期（0-5000轮）：只使用春季起始点（单一场景，快速学习）
    - 中期（5000-8000轮）：使用春夏两个起始点（增加多样性）
    - 后期（8000-10000轮）：使用全部4个起始点（完整评估）
    """
    def __init__(self):
        # 固定的季节起始点（春夏秋冬）
        self.season_starts = [0, 2190, 4380, 6570]
        
    def get_start_point(self, episode):
        """获取当前episode的起始点"""
        if episode < 5000:
            # 前期：只使用春季（单一场景）
            return self.season_starts[0]
        elif episode < 8000:
            # 中期：使用春夏（两个场景）
            idx = episode % 2
            return self.season_starts[idx]
        else:
            # 后期：使用全部4个季节
            idx = episode % 4
            return self.season_starts[idx]


class NoiseScheduler:
    """
    噪声调度器：从高噪声（探索）逐渐降低到低噪声（利用）
    """
    def __init__(self, initial_noise=0.2, final_noise=0.02, decay_episodes=8000):
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.decay_episodes = decay_episodes
    
    def get_noise(self, episode):
        """获取当前episode的噪声水平"""
        if episode >= self.decay_episodes:
            return self.final_noise
        # 线性衰减
        progress = episode / self.decay_episodes
        return self.initial_noise - (self.initial_noise - self.final_noise) * progress


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
        print(f"[ERROR] 错误：找不到文件")
        print(f"   期望路径: {filepath}")
        print(f"   请确认 'src/data' 文件夹下是否存在该文件。")
        raise FileNotFoundError(f"找不到文件: {filepath}")

    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        print(f"检测到 Excel 文件，正在读取: {os.path.basename(filepath)} ...")
        try:
            df = pd.read_excel(filepath, index_col=0, parse_dates=True)
            print(f"[OK] 成功读取 Excel 文件")
            return df
        except ImportError:
            raise ImportError("读取 .xlsx 文件需要 'openpyxl' 库。请运行: pip install openpyxl")
        except Exception as e:
            print(f"读取 Excel 失败: {e}")
            raise e
    else:
        encodings = ['utf-8', 'gbk', 'gb18030', 'ISO-8859-1', 'cp1252']
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True, encoding=enc)
                print(f"[OK] 成功使用编码 '{enc}' 读取文件")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"读取 {filepath} 时发生非编码错误: {e}")
                raise e
        raise ValueError(f"无法使用以下编码读取 CSV 文件 {filepath}: {encodings}")


def load_and_process_real_data():
    """读取真实数据，提取气象参数，重采样为每小时，并确保从 00:00 开始"""
    print(f"正在加载真实数据 (from {DATA_DIR})...")

    try:
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

        if not (wind_col and solar_col and temp_col):
            raise ValueError(f"无法找到所需的列。请检查 Excel 列名。")

        print(f"映射列名:\n 风速 -> {wind_col}\n 光照 -> {solar_col}\n 温度 -> {temp_col}")

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

        print("正在重采样为每小时均值...")
        df_hourly = df_merged.resample('1H').mean().dropna()

        print("\n" + "-" * 30)
        print("[INFO] 数据质量检查报告")
        daily_profile = df_hourly.groupby(df_hourly.index.hour)['irradiance'].mean()
        peak_hour = daily_profile.idxmax()
        peak_val = daily_profile.max()
        print(f"   光照日均峰值时间: {peak_hour}:00")
        print(f"   光照日均峰值强度: {peak_val:.2f} W/m2")

        non_zero_solar = df_hourly[df_hourly['irradiance'] > 10]
        if len(non_zero_solar) == 0:
            print("[WARNING] 警告：所有光照数据均为 0！请检查源文件数据列是否正确。")
        else:
            print(f"   有效光照小时数 (G>10): {len(non_zero_solar)} 小时")
        print("-" * 30 + "\n")

        first_midnight_idx = np.where(df_hourly.index.hour == 0)[0]
        if len(first_midnight_idx) > 0:
            start_idx = first_midnight_idx[0]
            print(f"正在截取数据，确保从 {df_hourly.index[start_idx]} (00:00) 开始...")
            df_hourly = df_hourly.iloc[start_idx:]
        else:
            print("[WARNING] 警告：数据中未找到 00:00 时刻。")

        print("生成匹配长度的简单合成负荷数据...")
        df_hourly['load'] = generate_simple_load(len(df_hourly))

        print(f"成功加载并处理真实数据! 最终数据点数: {len(df_hourly)} 小时")
        return df_hourly

    except Exception as e:
        print(f"[WARNING] 加载真实数据失败: {e}")
        import traceback
        traceback.print_exc()
        raise e


def evaluate_policy(env, agent, eval_episodes=4):
    """
    使用确定性策略（无噪声）评估Agent性能
    """
    # 固定的典型日起始点（春夏秋冬）
    eval_starts = [0, 2190, 4380, 6570]
    
    total_reward = 0
    total_h2 = 0
    total_unmet = 0
    
    for start_step in eval_starts[:eval_episodes]:
        state = env.reset(start_step=start_step, init_soc=0.5, init_soch=0.5)
        episode_reward = 0
        episode_h2 = 0
        episode_unmet = 0
        
        for step in range(MAX_STEPS):
            # 使用确定性策略（noise=False）
            action = agent.select_action(state, noise=False)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_h2 += info.get('h2_prod', 0)
            episode_unmet += info.get('p_unmet', 0)
            state = next_state
            
            if done:
                break
        
        total_reward += episode_reward
        total_h2 += episode_h2
        total_unmet += episode_unmet
    
    avg_reward = total_reward / eval_episodes
    avg_h2 = total_h2 / eval_episodes
    avg_unmet = total_unmet / eval_episodes
    
    return avg_reward, avg_h2, avg_unmet


def train():
    # --- GPU 强制检测 ---
    print("\n" + "=" * 30)
    print("      GPU 强制检测")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("[ERROR] 未检测到GPU！")
        print("[ERROR] 本训练脚本要求使用GPU加速。")
        print("[ERROR] 请确保：")
        print("  1. 已安装支持CUDA的PyTorch版本")
        print("  2. 系统有可用的NVIDIA GPU")
        print("  3. 已安装正确版本的CUDA驱动")
        print("\n安装GPU版PyTorch命令：")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("=" * 30 + "\n")
        raise RuntimeError("GPU不可用，训练终止。")
    
    device = torch.device("cuda")
    print(f"[OK] PyTorch 版本: {torch.__version__}")
    print(f"[OK] 强制使用GPU训练")
    print(f"[OK] 显卡名称: {torch.cuda.get_device_name(0)}")
    print(f"[OK] 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 30 + "\n")

    # 1. 加载数据
    if USE_REAL_DATA:
        if not os.path.exists(DATA_DIR):
            print(f"[WARNING] 警告: 找不到 'src/data' 文件夹: {DATA_DIR}")
            return
        df_data = load_and_process_real_data()
    else:
        print("未启用真实数据模式，生成合成数据...")
        synth_hours = 8760
        df_data = pd.DataFrame({
            'wind_speed': np.random.weibull(2.0, synth_hours) * 6.0,
            'irradiance': np.maximum(0, np.sin(np.linspace(0, 365 * 2 * np.pi, synth_hours))) * 1000,
            'temperature': 20 + 5 * np.sin(np.linspace(0, 365 * 2 * np.pi, synth_hours)),
            'load': generate_simple_load(synth_hours)
        })

    # 2. 初始化环境（V12）
    env = H2RESEnv(df_data, df_data)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化Agent
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        tau=TAU,
        noise_scale=0.2,
        device="cuda"
    )
    buffer = ReplayBuffer(REPLAY_SIZE)
    
    # ✨ V12改进：创建固定起始点调度器和噪声调度器
    start_scheduler = FixedStartScheduler()
    noise_scheduler = NoiseScheduler(initial_noise=0.2, final_noise=0.02, decay_episodes=8000)

    # 3. 训练循环
    if not os.path.exists('results'):
        os.makedirs('results')

    rewards_history = []
    h2_prod_history = []
    unmet_history = []
    eval_rewards_history = []
    eval_h2_history = []
    eval_unmet_history = []
    noise_history = []
    
    print("\n" + "=" * 60)
    print(f"  开始训练 DDPG V12 模型 - 共 {MAX_EPISODES} 轮")
    print("=" * 60)
    print(f"环境版本: V12 (Gemini's Approach - Agent-Driven Allocation)")
    print(f"核心改进:")
    print(f"  1. [OK] 移除'负荷优先'硬约束")
    print(f"  2. [OK] Agent自由分配功率（负荷、制氢、储能）")
    print(f"  3. [OK] 环境只负责总功率约束（按比例缩放）")
    print(f"  4. [OK] 负荷通过惩罚引导（软约束）")
    print(f"  5. [OK] 让Agent学习最优平衡策略")
    print(f"观测空间: {state_dim}维")
    print(f"数据长度: {len(df_data)} 小时")
    print(f"每轮步数: {MAX_STEPS} 小时")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"TAU: {TAU}")
    print(f"噪声衰减: 0.2→0.02 (8000轮)")
    print(f"设备: cuda (强制GPU训练)")
    print("=" * 60 + "\n")

    for episode in range(MAX_EPISODES):
        # ✨ V12改进：使用固定起始点（降低随机性）
        start_step = start_scheduler.get_start_point(episode)
        
        # ✨ V12改进：动态调整噪声
        current_noise = noise_scheduler.get_noise(episode)
        agent.noise_scale = current_noise
        noise_history.append(current_noise)
        
        # 固定初始状态（进一步降低随机性）
        init_soc = 0.5
        init_soch = 0.5
        
        state = env.reset(start_step=start_step, init_soc=init_soc, init_soch=init_soch)

        episode_reward = 0
        episode_h2_prod = 0
        episode_unmet = 0
        episode_critic_loss = 0
        episode_actor_loss = 0
        steps = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state, noise=True)
            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            
            # 只有当buffer中有足够样本时才进行学习
            if len(buffer) >= BATCH_SIZE:
                losses = agent.update(buffer, BATCH_SIZE)
                if losses is not None:
                    critic_loss, actor_loss = losses
                    episode_critic_loss += critic_loss
                    episode_actor_loss += actor_loss
            
            state = next_state
            episode_reward += reward
            episode_h2_prod += info.get('h2_prod', 0)
            episode_unmet += info.get('p_unmet', 0)
            steps += 1
            if done: break

        rewards_history.append(episode_reward)
        h2_prod_history.append(episode_h2_prod)
        unmet_history.append(episode_unmet)

        # 改进的进度打印
        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(rewards_history[-10:])
            avg_h2_10 = np.mean(h2_prod_history[-10:])
            avg_unmet_10 = np.mean(unmet_history[-10:])
            avg_critic_loss = episode_critic_loss / steps if steps > 0 else 0
            avg_actor_loss = episode_actor_loss / steps if steps > 0 else 0
            print(f"Episode {episode+1:5d}/{MAX_EPISODES} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg(10): {avg_reward_10:8.2f} | "
                  f"H2: {episode_h2_prod:6.2f} kg | "
                  f"Unmet: {episode_unmet:6.2f} kW | "
                  f"Noise: {current_noise:.3f} | "
                  f"C_Loss: {avg_critic_loss:.4f} | "
                  f"A_Loss: {avg_actor_loss:.4f}")
        
        # ✨ 每100轮进行确定性评估
        if (episode + 1) % 100 == 0:
            eval_reward, eval_h2, eval_unmet = evaluate_policy(env, agent, eval_episodes=4)
            eval_rewards_history.append(eval_reward)
            eval_h2_history.append(eval_h2)
            eval_unmet_history.append(eval_unmet)
            
            avg_reward_100 = np.mean(rewards_history[-100:])
            avg_h2_100 = np.mean(h2_prod_history[-100:])
            avg_unmet_100 = np.mean(unmet_history[-100:])
            print("-" * 60)
            print(f"[统计] 最近100轮训练平均: Reward={avg_reward_100:.2f}, H2={avg_h2_100:.2f} kg, Unmet={avg_unmet_100:.2f} kW")
            print(f"[评估] 确定性策略性能: Reward={eval_reward:.2f}, H2={eval_h2:.2f} kg, Unmet={eval_unmet:.2f} kW")
            print("-" * 60)
        
        # 每1000轮保存检查点
        if (episode + 1) % 1000 == 0:
            checkpoint_path = f'results/ddpg_v12_ep{episode+1}.pth'
            agent.save(checkpoint_path)
            print(f"[保存] 检查点已保存: {checkpoint_path}")

    # 4. Save
    print("\n" + "=" * 60)
    print("  训练完成！正在保存模型...")
    print("=" * 60)
    
    agent.save('results/ddpg_v12.pth')
    np.save('results/training_rewards_v12.npy', rewards_history)
    np.save('results/training_h2prod_v12.npy', h2_prod_history)
    np.save('results/training_unmet_v12.npy', unmet_history)
    np.save('results/eval_rewards_v12.npy', eval_rewards_history)
    np.save('results/eval_h2_v12.npy', eval_h2_history)
    np.save('results/eval_unmet_v12.npy', eval_unmet_history)
    np.save('results/noise_history_v12.npy', noise_history)
    
    # 打印最终统计
    final_avg_reward = np.mean(rewards_history[-100:])
    final_avg_h2 = np.mean(h2_prod_history[-100:])
    final_avg_unmet = np.mean(unmet_history[-100:])
    final_eval_reward = eval_rewards_history[-1] if eval_rewards_history else 0
    final_eval_h2 = eval_h2_history[-1] if eval_h2_history else 0
    final_eval_unmet = eval_unmet_history[-1] if eval_unmet_history else 0
    
    print(f"\n最终性能 (最近100轮训练平均):")
    print(f"  训练平均奖励: {final_avg_reward:.2f}")
    print(f"  训练平均制氢量: {final_avg_h2:.2f} kg/天")
    print(f"  训练平均缺电: {final_avg_unmet:.2f} kW")
    print(f"\n最终性能 (确定性评估):")
    print(f"  评估平均奖励: {final_eval_reward:.2f}")
    print(f"  评估平均制氢量: {final_eval_h2:.2f} kg/天")
    print(f"  评估平均缺电: {final_eval_unmet:.2f} kW")
    print(f"\n模型已保存到: results/ddpg_v12.pth")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train()