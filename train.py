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

from src.envs.h2_res_env import H2RESEnv
from src.algos.ddpg import DDPGAgent, ReplayBuffer

# --- Hyperparameters ---
MAX_EPISODES = 10000 # Paper says 20000. Set to 200 for testing.
MAX_STEPS = 24  # 24 hours per episode
BATCH_SIZE = 16  # 降低batch size，确保能够及时开始学习（24步/episode，16<24）
GAMMA = 0.99
TAU = 0.001
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
REPLAY_SIZE = 100000
NOISE = 0.2

# --- 真实数据配置 ---
USE_REAL_DATA = True
DATA_DIR = os.path.join(current_dir, 'src', 'data')
REAL_DATA_PATHS = {
    'wind': os.path.join(DATA_DIR, 'Wind farm site 1 (Nominal capacity-99MW).xlsx'),
    'solar': os.path.join(DATA_DIR, 'Solar station site 1 (Nominal capacity-50MW).xlsx')
}


def generate_simple_load(hours):
    """
    生成简单的合成负荷数据
    """
    time = np.arange(hours)
    # 典型的双峰负荷曲线 (Base + Morning Peak + Evening Peak)
    base_load = 2000  # kW
    daily_pattern = 1000 * (np.exp(-((time % 24 - 9) ** 2) / 10) + np.exp(-((time % 24 - 19) ** 2) / 10))
    # 添加一些随机波动
    load = base_load + daily_pattern + np.random.normal(0, 100, hours)
    return load


def read_data_robust(filepath):
    """
    智能读取数据：自动区分 Excel (.xlsx) 和 CSV (.csv)
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] 错误：找不到文件")
        print(f"   期望路径: {filepath}")
        print(f"   请确认 'src/data' 文件夹下是否存在该文件。")
        raise FileNotFoundError(f"找不到文件: {filepath}")

    # 1. 如果是 Excel 文件
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        print(f"检测到 Excel 文件，正在读取: {os.path.basename(filepath)} ...")
        try:
            # 需要安装 openpyxl: pip install openpyxl
            df = pd.read_excel(filepath, index_col=0, parse_dates=True)
            print(f"[OK] 成功读取 Excel 文件")
            return df
        except ImportError:
            raise ImportError("读取 .xlsx 文件需要 'openpyxl' 库。请运行: pip install openpyxl")
        except Exception as e:
            print(f"读取 Excel 失败: {e}")
            raise e

    # 2. 如果是 CSV 文件 (尝试多种编码)
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
    """
    读取真实数据，提取气象参数，重采样为每小时，并确保从 00:00 开始
    """
    print(f"正在加载真实数据 (from {DATA_DIR})...")

    try:
        # 1. 读取数据
        df_wind_raw = read_data_robust(REAL_DATA_PATHS['wind'])
        df_solar_raw = read_data_robust(REAL_DATA_PATHS['solar'])

        # 2. 清理列名
        df_wind_raw.columns = df_wind_raw.columns.str.strip()
        df_solar_raw.columns = df_solar_raw.columns.str.strip()

        # 3. 智能查找列名
        def find_col(df, keywords):
            for col in df.columns:
                if all(k in col for k in keywords):
                    return col
            return None

        # 查找列名
        wind_col = find_col(df_wind_raw, ['Wind speed', 'wheel hub'])
        if not wind_col: wind_col = find_col(df_wind_raw, ['Wind speed', '10 meters'])

        solar_col = find_col(df_solar_raw, ['Global horizontal'])
        if not solar_col: solar_col = find_col(df_solar_raw, ['Total solar'])

        temp_col = find_col(df_wind_raw, ['Air temperature'])
        if not temp_col: temp_col = find_col(df_solar_raw, ['Air temperature'])

        if not (wind_col and solar_col and temp_col):
            raise ValueError(f"无法找到所需的列。请检查 Excel 列名。")

        print(f"映射列名:\n 风速 -> {wind_col}\n 光照 -> {solar_col}\n 温度 -> {temp_col}")

        # 4. 提取数据并对齐
        # ⚠️ 关键修改：强制转换为 numeric，处理可能存在的非数值字符
        wind_data = pd.to_numeric(df_wind_raw[wind_col], errors='coerce')
        solar_data = pd.to_numeric(df_solar_raw[solar_col], errors='coerce')
        # 温度优先取风场的，若无则取光伏场的
        raw_temp = df_wind_raw[temp_col] if temp_col in df_wind_raw.columns else df_solar_raw[temp_col]
        temp_data = pd.to_numeric(raw_temp, errors='coerce')

        df_merged = pd.DataFrame({
            'wind_speed': wind_data,
            'irradiance': solar_data,
            'temperature': temp_data
        })

        # 移除空值并排序
        df_merged = df_merged.dropna().sort_index()

        # 5. 重采样为 1 小时均值
        print("正在重采样为每小时均值...")
        df_hourly = df_merged.resample('1H').mean().dropna()

        # --- 数据体检模块 (新增) ---
        print("\n" + "-" * 30)
        print("[INFO] 数据质量检查报告")
        # 检查光照峰值时间
        daily_profile = df_hourly.groupby(df_hourly.index.hour)['irradiance'].mean()
        peak_hour = daily_profile.idxmax()
        peak_val = daily_profile.max()
        print(f"   光照日均峰值时间: {peak_hour}:00")
        print(f"   光照日均峰值强度: {peak_val:.2f} W/m2")

        # 检查是否全为0
        non_zero_solar = df_hourly[df_hourly['irradiance'] > 10]
        if len(non_zero_solar) == 0:
            print("[WARNING] 警告：所有光照数据均为 0！请检查源文件数据列是否正确。")
        else:
            print(f"   有效光照小时数 (G>10): {len(non_zero_solar)} 小时")
            print("   部分光照数据预览 (12:00时刻):")
            noon_data = df_hourly[df_hourly.index.hour == 12]['irradiance'].head(3)
            print(noon_data)
        print("-" * 30 + "\n")
        # ------------------------

        # 6. 确保数据从 00:00 开始
        first_midnight_idx = np.where(df_hourly.index.hour == 0)[0]
        if len(first_midnight_idx) > 0:
            start_idx = first_midnight_idx[0]
            print(f"正在截取数据，确保从 {df_hourly.index[start_idx]} (00:00) 开始...")
            df_hourly = df_hourly.iloc[start_idx:]
        else:
            print("[WARNING] 警告：数据中未找到 00:00 时刻。")

        # 7. 生成负荷
        print("生成匹配长度的简单合成负荷数据...")
        df_hourly['load'] = generate_simple_load(len(df_hourly))

        print(f"成功加载并处理真实数据! 最终数据点数: {len(df_hourly)} 小时")
        return df_hourly

    except Exception as e:
        print(f"[WARNING] 加载真实数据失败: {e}")
        import traceback
        traceback.print_exc()
        raise e


def train():
    # --- GPU 检测 ---
    print("\n" + "=" * 30)
    print("      设备检测 (GPU/CPU)")
    print("=" * 30)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"当前训练设备: {device}")
    if device.type == 'cuda':
        print(f"显卡名称: {torch.cuda.get_device_name(0)}")
    print("=" * 30 + "\n")

    # 1. 加载数据
    if USE_REAL_DATA:
        if not os.path.exists(DATA_DIR):
            print(f"[WARNING] 警告: 找不到 'src/data' 文件夹: {DATA_DIR}")
            return
        df_data = load_and_process_real_data()
    else:
        # Fallback
        print("未启用真实数据模式，生成合成数据...")
        synth_hours = 8760
        df_data = pd.DataFrame({
            'wind_speed': np.random.weibull(2.0, synth_hours) * 6.0,
            'irradiance': np.maximum(0, np.sin(np.linspace(0, 365 * 2 * np.pi, synth_hours))) * 1000,
            'temperature': 20 + 5 * np.sin(np.linspace(0, 365 * 2 * np.pi, synth_hours)),
            'load': generate_simple_load(synth_hours)
        })

    # 2. 初始化环境
    env = H2RESEnv(df_data, df_data)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim, LR_ACTOR, LR_CRITIC, GAMMA, TAU, NOISE)
    buffer = ReplayBuffer(REPLAY_SIZE)

    # 3. 训练循环
    if not os.path.exists('results'):
        os.makedirs('results')

    rewards_history = []
    print(f"Starting training for {MAX_EPISODES} episodes...")

    max_days = (len(df_data) - MAX_STEPS) // 24
    if max_days < 1:
        print("[ERROR] 数据长度不足 24 小时，无法训练。")
        return

    for episode in range(MAX_EPISODES):
        # 修复: 使用滑动窗口随机采样，支持任意时刻开始
        start_step = np.random.randint(0, len(df_data) - MAX_STEPS)
        
        # 修复: 随机初始状态，增加探索多样性
        init_soc = np.random.uniform(0.3, 0.95) if episode > 0 else 0.5
        init_soch = np.random.uniform(0.2, 0.9) if episode > 0 else 0.5
        
        # 使用新的reset接口
        state = env.reset(start_step=start_step, init_soc=init_soc, init_soch=init_soch)

        episode_reward = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state, noise=True)
            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            
            # 只有当buffer中有足够样本时才进行学习
            if len(buffer) >= BATCH_SIZE:
                agent.update(buffer, BATCH_SIZE)
            
            state = next_state
            episode_reward += reward
            if done: break

        rewards_history.append(episode_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}")

    # 4. Save
    agent.save('results/ddpg_checkpoint.pth')
    np.save('results/training_rewards.npy', rewards_history)
    print("Training Complete. Model saved.")


if __name__ == "__main__":
    train()