import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# --- 路径设置 ---
# 确保能找到 src 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.envs.h2_res_env import H2RESEnv
from src.algos.ddpg import DDPGAgent

# --- 配置 ---
USE_REAL_DATA = True
# 数据文件夹路径
DATA_DIR = os.path.join(current_dir, 'src', 'data')
REAL_DATA_PATHS = {
    'wind': os.path.join(DATA_DIR, 'Wind farm site 1 (Nominal capacity-99MW).xlsx'),
    'solar': os.path.join(DATA_DIR, 'Solar station site 1 (Nominal capacity-50MW).xlsx')
}
N_CLUSTERS = 4  # 聚类出典型的 4 种天气


# --- 辅助函数 ---
def calculate_wind_power_normalized(v_array):
    """计算归一化风机出力 (0-1)，考虑切入切出风速"""
    v_cut_in = 3.0
    v_rated = 12.0
    v_cut_out = 25.0
    p = np.zeros_like(v_array)

    # 爬坡区
    mask_ramp = (v_array >= v_cut_in) & (v_array < v_rated)
    p[mask_ramp] = ((v_array[mask_ramp] - v_cut_in) / (v_rated - v_cut_in)) ** 3

    # 额定区
    mask_rated = (v_array >= v_rated) & (v_array <= v_cut_out)
    p[mask_rated] = 1.0

    # 切出区自动为 0
    return p


def generate_simple_load(hours):
    """生成简单的双峰负荷曲线"""
    time = np.arange(hours)
    base_load = 2000
    daily_pattern = 1000 * (np.exp(-((time % 24 - 9) ** 2) / 10) + np.exp(-((time % 24 - 19) ** 2) / 10))
    load = base_load + daily_pattern + np.random.normal(0, 100, hours)
    return load


def read_data_robust(filepath):
    """读取数据，支持 xlsx 和 csv"""
    if not os.path.exists(filepath):
        print(f"[ERROR] 找不到文件 {filepath}")
        raise FileNotFoundError(f"找不到文件")

    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        try:
            return pd.read_excel(filepath, index_col=0, parse_dates=True)
        except ImportError:
            raise ImportError("需要安装 openpyxl: pip install openpyxl")
    else:
        for enc in ['utf-8', 'gbk', 'ISO-8859-1']:
            try:
                return pd.read_csv(filepath, index_col=0, parse_dates=True, encoding=enc)
            except:
                continue
        raise ValueError(f"无法读取文件 {filepath}")


def load_real_data_for_eval():
    """加载真实数据并预处理"""
    print("正在加载全量数据用于聚类分析...")
    df_wind = read_data_robust(REAL_DATA_PATHS['wind'])
    df_solar = read_data_robust(REAL_DATA_PATHS['solar'])

    # 清理列名
    df_wind.columns = df_wind.columns.str.strip()
    df_solar.columns = df_solar.columns.str.strip()

    # 智能查找列名
    def find_col(df, keywords):
        for col in df.columns:
            if all(k in col for k in keywords): return col
        return None

    wind_col = find_col(df_wind, ['Wind speed', 'wheel hub']) or find_col(df_wind, ['Wind speed', '10 meters'])
    solar_col = find_col(df_solar, ['Global horizontal']) or find_col(df_solar, ['Total solar'])
    temp_col = find_col(df_wind, ['Air temperature']) or find_col(df_solar, ['Air temperature'])

    # 合并
    df_merged = pd.DataFrame({
        'wind_speed': pd.to_numeric(df_wind[wind_col], errors='coerce'),
        'irradiance': pd.to_numeric(df_solar[solar_col], errors='coerce'),
        'temperature': pd.to_numeric(df_wind[temp_col] if temp_col in df_wind else df_solar[temp_col], errors='coerce')
    }).dropna().sort_index()

    # 重采样
    df_hourly = df_merged.resample('1H').mean().dropna()
    df_hourly['load'] = generate_simple_load(len(df_hourly))

    return df_hourly


def find_representative_days(df, n_clusters=4):
    """基于风光出力特征聚类，寻找典型日"""
    print(f"正在进行 K-Means 聚类 (k={n_clusters})...")
    n_days = len(df) // 24
    df_cut = df.iloc[:n_days * 24]

    # 提取特征
    raw_wind_speed = df_cut['wind_speed'].values
    wind_power_norm = calculate_wind_power_normalized(raw_wind_speed)
    wind_features = wind_power_norm.reshape(n_days, 24)

    raw_irradiance = df_cut['irradiance'].values
    solar_norm_val = raw_irradiance / (raw_irradiance.max() + 1e-5)
    solar_features = solar_norm_val.reshape(n_days, 24)

    # 组合特征
    features = np.hstack([wind_features, solar_features])

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features)

    # 找最近的样本
    centers = kmeans.cluster_centers_
    distances = cdist(centers, features, metric='euclidean')
    rep_day_indices = np.argmin(distances, axis=1)

    results = []
    for cluster_id, day_idx in enumerate(rep_day_indices):
        start_idx = day_idx * 24
        date = df_cut.index[start_idx].date()

        # 描述生成
        avg_wind_p_cap = wind_features[day_idx].mean()
        avg_solar_p_cap = solar_features[day_idx].mean()

        desc = []
        if avg_wind_p_cap > 0.4:
            desc.append("High Wind")
        elif avg_wind_p_cap > 0.15:
            desc.append("Med Wind")
        else:
            desc.append("Low Wind")

        if avg_solar_p_cap > 0.25:
            desc.append("High Solar")
        elif avg_solar_p_cap > 0.1:
            desc.append("Med Solar")
        else:
            desc.append("Low Solar")

        results.append({
            'cluster_id': cluster_id, 'start_idx': start_idx, 'date': date, 'desc': ", ".join(desc)
        })
        print(f"   类别 {cluster_id}: {desc} | 日期: {date}")

    return results


def plot_training_history():
    """绘制训练收敛曲线"""
    log_path = 'results/training_rewards.npy'
    if not os.path.exists(log_path): return
    rewards = np.load(log_path)

    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
    else:
        moving_avg = rewards

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, color='gray', label='Raw Reward')
    plt.plot(np.arange(len(moving_avg)) + (len(rewards) - len(moving_avg)), moving_avg, color='red',
             label='Moving Average')
    plt.title("DDPG Training Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/training_convergence.png')
    plt.close()


def evaluate_and_plot():
    # 0. 画训练曲线
    plot_training_history()

    # 1. 准备数据
    if USE_REAL_DATA and os.path.exists(REAL_DATA_PATHS['wind']):
        df_data = load_real_data_for_eval()
    else:
        print("未找到真实数据。")
        return

    # 2. 聚类寻找典型日
    typical_days = find_representative_days(df_data, n_clusters=N_CLUSTERS)

    # 3. 初始化模型
    # 临时创建环境以获取维度
    temp_env = H2RESEnv(df_data.iloc[:48], df_data.iloc[:48])
    agent = DDPGAgent(temp_env.observation_space.shape[0], temp_env.action_space.shape[0])

    if os.path.exists('results/ddpg_checkpoint.pth'):
        agent.load('results/ddpg_checkpoint.pth')
        print("[OK] 已加载训练好的模型权重。")
    else:
        print("[WARNING] 未找到 checkpoint，将使用随机策略！")

    # 4. 循环评估每一个典型日
    plt.rcParams['figure.figsize'] = (14, 12)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    for i, day_info in enumerate(typical_days):
        start_idx = day_info['start_idx']
        if start_idx + 48 > len(df_data): continue

        df_eval = df_data.iloc[start_idx: start_idx + 48].copy()
        env = H2RESEnv(df_eval, df_eval)

        state = env.reset()
        done = False

        history = {'P_WT': [], 'P_PV': [], 'P_Bat': [], 'P_EL': [], 'P_FC': [], 'SOC': [], 'SOCH': [], 'Load': [],
                   'Time': []}
        eval_times = df_eval.index
        step_count = 0

        while not done and step_count < 48:
            action = agent.select_action(state, noise=False)
            next_state, reward, done, info = env.step(action)

            # --- [CRITICAL FIX] 使用 info 中的真实物理量绘图 ---
            # 这样画出来的图，就是环境强制约束后的结果，绝对符合物理定律
            history['P_WT'].append(env.current_p_wt)
            history['P_PV'].append(env.current_p_pv)

            # 使用 info 中的真实值 (Real Values)，而不是 action
            history['P_Bat'].append(info['real_p_bat'])
            history['P_EL'].append(info['real_p_el'])
            history['P_FC'].append(info['real_p_fc'])

            history['SOC'].append(env.SOC)
            history['SOCH'].append(env.SOCH)
            history['Load'].append(env.current_load)
            history['Time'].append(eval_times[step_count])

            state = next_state
            step_count += 1

        # --- 绘图 ---
        fig, axs = plt.subplots(3, 1, sharex=True)
        time_indices = np.arange(len(history['Time']))
        time_labels = [t.strftime("%H:%M") if idx % 4 == 0 else "" for idx, t in enumerate(history['Time'])]

        fig.suptitle(f"Typical Day {i + 1}: {day_info['desc']} ({day_info['date']})", fontsize=16)

        # 1. 源荷
        axs[0].set_title("Renewable Energy & Load")
        axs[0].plot(time_indices, history['P_WT'], label='Wind', color='skyblue', linewidth=2)
        axs[0].plot(time_indices, history['P_PV'], label='PV', color='orange', linewidth=2)
        axs[0].plot(time_indices, history['Load'], label='Load', color='black', linestyle='--', linewidth=2)
        axs[0].set_ylabel("Power (kW)")
        axs[0].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)

        # 2. 氢能与电池 (真实物理响应)
        axs[1].set_title("H2 & Battery Response (Physically Constrained)")
        axs[1].plot(time_indices, history['P_EL'], label='Electrolyzer (Real)', color='green')
        axs[1].plot(time_indices, history['P_FC'], label='Fuel Cell (Real)', color='red')
        axs[1].plot(time_indices, history['P_Bat'], label='Battery (Real)', color='blue', linestyle='-.')
        axs[1].set_ylabel("Power (kW)")
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)

        # 3. 储能状态
        axs[2].set_title("Storage State (SOC)")
        axs[2].plot(time_indices, history['SOC'], label='Battery SOC', color='blue', linewidth=2)
        axs[2].plot(time_indices, history['SOCH'], label='H2 Tank Level', color='purple', linewidth=2)
        axs[2].set_ylabel("SOC (0-1)")
        axs[2].set_ylim(-0.1, 1.1)
        axs[2].legend(loc='upper right')
        axs[2].grid(True, alpha=0.3)
        axs[2].set_xticks(time_indices)
        axs[2].set_xticklabels(time_labels, rotation=45)

        plt.tight_layout()
        save_name = f'results/typical_day_{i + 1}_{day_info["date"]}.png'
        plt.savefig(save_name)
        plt.close()
        print(f"[OK] 图表已保存: {save_name}")


if __name__ == "__main__":
    evaluate_and_plot()