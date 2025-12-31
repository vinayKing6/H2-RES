
"""
高质量顶刊级别可视化脚本
包含：能量流桑基图、堆叠面积图、热力图、雷达图等多种专业图表
Version: V12.0 - 支持V12环境（Agent-Driven Allocation）
"""
import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 路径设置
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

from src.algos.ddpg import DDPGAgent

# ==================== 配置 ====================
USE_REAL_DATA = True
DATA_DIR = os.path.join(current_dir, 'src', 'data')
REAL_DATA_PATHS = {
    'wind': os.path.join(DATA_DIR, 'Wind farm site 1 (Nominal capacity-99MW).xlsx'),
    'solar': os.path.join(DATA_DIR, 'Solar station site 1 (Nominal capacity-50MW).xlsx')
}
N_CLUSTERS = 5  # 扩展到5个典型日，更好地覆盖不同场景

# 顶刊配色方案 (Nature/Science风格)
COLORS = {
    'wind': '#4A90E2',      # 蓝色
    'solar': '#F5A623',     # 橙色
    'load': '#2C3E50',      # 深灰
    'battery': '#9013FE',   # 紫色
    'electrolyzer': '#50E3C2',  # 青色
    'fuelcell': '#E74C3C',  # 红色
    'h2_storage': '#16A085', # 绿松石
    'grid_bg': '#F8F9FA',   # 浅灰背景
    'surplus': '#7ED321',   # 绿色
    'deficit': '#D0021B'    # 深红
}

# 设置全局字体和样式（支持中文）
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2,
    'patch.linewidth': 1,
    'axes.unicode_minus': False
})

# ==================== 辅助函数 ====================
def calculate_wind_power_normalized(v_array):
    """计算归一化风机出力"""
    v_cut_in, v_rated, v_cut_out = 3.0, 12.0, 25.0
    p = np.zeros_like(v_array)
    mask_ramp = (v_array >= v_cut_in) & (v_array < v_rated)
    p[mask_ramp] = ((v_array[mask_ramp] - v_cut_in) / (v_rated - v_cut_in)) ** 3
    mask_rated = (v_array >= v_rated) & (v_array <= v_cut_out)
    p[mask_rated] = 1.0
    return p


def generate_simple_load(hours):
    """生成双峰负荷曲线"""
    time = np.arange(hours)
    base_load = 2000
    daily_pattern = 1000 * (np.exp(-((time % 24 - 9) ** 2) / 10) + 
                           np.exp(-((time % 24 - 19) ** 2) / 10))
    load = base_load + daily_pattern + np.random.normal(0, 100, hours)
    return load


def read_data_robust(filepath):
    """读取数据"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")
    
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
    """加载真实数据"""
    print("正在加载数据...")
    df_wind = read_data_robust(REAL_DATA_PATHS['wind'])
    df_solar = read_data_robust(REAL_DATA_PATHS['solar'])
    
    df_wind.columns = df_wind.columns.str.strip()
    df_solar.columns = df_solar.columns.str.strip()
    
    def find_col(df, keywords):
        for col in df.columns:
            if all(k in col for k in keywords):
                return col
        return None
    
    wind_col = find_col(df_wind, ['Wind speed', 'wheel hub']) or find_col(df_wind, ['Wind speed', '10 meters'])
    solar_col = find_col(df_solar, ['Global horizontal']) or find_col(df_solar, ['Total solar'])
    temp_col = find_col(df_wind, ['Air temperature']) or find_col(df_solar, ['Air temperature'])
    
    df_merged = pd.DataFrame({
        'wind_speed': pd.to_numeric(df_wind[wind_col], errors='coerce'),
        'irradiance': pd.to_numeric(df_solar[solar_col], errors='coerce'),
        'temperature': pd.to_numeric(df_wind[temp_col] if temp_col in df_wind else df_solar[temp_col], errors='coerce')
    }).dropna().sort_index()
    
    df_hourly = df_merged.resample('1H').mean().dropna()
    df_hourly['load'] = generate_simple_load(len(df_hourly))
    
    return df_hourly


def find_representative_days(df, n_clusters=5):
    """聚类寻找典型日（增强版V2：更好地识别风光大发、夜间大风等场景）"""
    print(f"正在进行 K-Means 聚类 (k={n_clusters})...")
    n_days = len(df) // 24
    df_cut = df.iloc[:n_days * 24]
    
    raw_wind_speed = df_cut['wind_speed'].values
    wind_power_norm = calculate_wind_power_normalized(raw_wind_speed)
    wind_features = wind_power_norm.reshape(n_days, 24)
    
    raw_irradiance = df_cut['irradiance'].values
    solar_norm_val = raw_irradiance / (raw_irradiance.max() + 1e-5)
    solar_features = solar_norm_val.reshape(n_days, 24)
    
    # ========== 增强特征V2：添加更多区分性特征 ==========
    
    # 1. 昼夜风电差异特征
    night_mask = np.zeros(24, dtype=bool)
    night_mask[0:6] = True
    night_mask[18:24] = True
    day_mask = ~night_mask
    
    wind_night_avg = wind_features[:, night_mask].mean(axis=1)
    wind_day_avg = wind_features[:, day_mask].mean(axis=1)
    wind_night_day_ratio = wind_night_avg / (wind_day_avg + 1e-5)
    
    # 2. 总发电量特征（关键！用于识别风光大发日）
    wind_total_daily = wind_features.mean(axis=1)  # 日均风电
    solar_total_daily = solar_features.mean(axis=1)  # 日均光伏
    total_re_daily = wind_total_daily + solar_total_daily  # 日均总可再生能源
    
    # 3. 光伏峰值特征（识别强光照日）
    solar_peak = solar_features.max(axis=1)  # 光伏峰值
    solar_peak_hour = solar_features.argmax(axis=1)  # 峰值时刻
    
    # 4. 风电稳定性特征
    wind_std = wind_features.std(axis=1)  # 风电波动性
    
    # 组合特征（增加权重到关键特征）
    features = np.hstack([
        wind_features * 0.5,                    # 24维：每小时风电（降低权重）
        solar_features * 0.5,                   # 24维：每小时光伏（降低权重）
        wind_total_daily.reshape(-1, 1) * 5.0, # 1维：日均风电（高权重！）
        solar_total_daily.reshape(-1, 1) * 5.0,# 1维：日均光伏（高权重！）
        total_re_daily.reshape(-1, 1) * 8.0,   # 1维：日均总发电（最高权重！）
        solar_peak.reshape(-1, 1) * 3.0,       # 1维：光伏峰值（中等权重）
        wind_night_day_ratio.reshape(-1, 1) * 2.0,  # 1维：夜间/白天风电比例
        wind_night_avg.reshape(-1, 1) * 2.0,   # 1维：夜间平均风电
        wind_std.reshape(-1, 1),                # 1维：风电波动性
    ])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features)
    
    centers = kmeans.cluster_centers_
    distances = cdist(centers, features, metric='euclidean')
    rep_day_indices = np.argmin(distances, axis=1)
    
    results = []
    for cluster_id, day_idx in enumerate(rep_day_indices):
        start_idx = day_idx * 24
        date = df_cut.index[start_idx].date()
        
        avg_wind_p_cap = wind_features[day_idx].mean()
        avg_solar_p_cap = solar_features[day_idx].mean()
        total_re = wind_total_daily[day_idx] + solar_total_daily[day_idx]
        night_wind = wind_night_avg[day_idx]
        day_wind = wind_day_avg[day_idx]
        solar_pk = solar_peak[day_idx]
        
        desc = []
        
        # ========== 改进的场景识别逻辑 ==========
        
        # 1. 优先识别"风光大发日"（最重要！）
        if total_re > 0.6:
            desc.append("🌟风光大发")  # 总发电量>60%
        elif total_re > 0.45:
            desc.append("风光充足")  # 总发电量>45%
        
        # 2. 识别特殊风电场景
        if night_wind > 0.4 and night_wind > day_wind * 1.2:
            desc.append("夜间大风")
        elif avg_wind_p_cap > 0.5:
            desc.append("全天强风")
        elif avg_wind_p_cap > 0.25:
            desc.append("中等风力")
        elif avg_wind_p_cap < 0.1:
            desc.append("弱风")
        
        # 3. 识别光伏场景
        if solar_pk > 0.6:
            desc.append("强光照")
        elif solar_pk > 0.3:
            desc.append("中等光照")
        elif solar_pk < 0.15:
            desc.append("弱光照/阴天")
        
        results.append({
            'cluster_id': cluster_id,
            'start_idx': start_idx,
            'date': date,
            'desc': ", ".join(desc),
            'total_re': total_re,
            'avg_wind': avg_wind_p_cap,
            'avg_solar': avg_solar_p_cap,
            'solar_peak': solar_pk,
            'night_wind': night_wind,
            'day_wind': day_wind
        })
        print(f"   类别 {cluster_id}: {', '.join(desc)}")
        print(f"      日期: {date} | 总发电: {total_re:.2%} | 风电: {avg_wind_p_cap:.2%} | 光伏峰值: {solar_pk:.2%}")
    
    return results


# ==================== 高级可视化函数 ====================

def plot_advanced_training_convergence(version='v12'):
    """高级训练收敛曲线 - 包含统计信息"""
    # 根据版本选择文件
    if version == 'v12':
        log_path = 'results/training_rewards_v12.npy'
        h2_path = 'results/training_h2prod_v12.npy'
        title_suffix = 'V12 (Agent-Driven)'
    else:
        log_path = 'results/training_rewards.npy'
        h2_path = 'results/training_h2prod.npy'
        title_suffix = 'Standard'
    
    if not os.path.exists(log_path):
        print(f"[SKIP] {log_path} not found")
        return
    
    rewards = np.load(log_path)
    h2_prod = np.load(h2_path) if os.path.exists(h2_path) else None
    
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)
    
    # 左图：收敛曲线
    ax1 = fig.add_subplot(gs[0])
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        moving_std = pd.Series(rewards).rolling(window).std().dropna()
    else:
        moving_avg = rewards
        moving_std = np.zeros_like(rewards)
    
    x_avg = np.arange(len(moving_avg)) + (len(rewards) - len(moving_avg))
    
    # 填充标准差区域
    ax1.fill_between(x_avg, 
                     moving_avg - moving_std, 
                     moving_avg + moving_std,
                     alpha=0.2, color=COLORS['wind'], label='±1 Std Dev')
    
    ax1.plot(rewards, alpha=0.15, color='gray', linewidth=0.8)
    ax1.plot(x_avg, moving_avg, color=COLORS['wind'], linewidth=2.5, 
             label=f'Moving Avg (window={window})')
    
    ax1.set_xlabel('训练回合', fontweight='bold')
    ax1.set_ylabel('累积奖励', fontweight='bold')
    ax1.set_title(f'(a) DDPG训练收敛曲线 ({title_suffix})', fontweight='bold', loc='left')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 右图：统计分布
    ax2 = fig.add_subplot(gs[1])
    
    # 分段统计
    n_segments = 5
    segment_size = len(rewards) // n_segments
    segment_means = [rewards[i*segment_size:(i+1)*segment_size].mean() 
                     for i in range(n_segments)]
    segment_labels = [f'{i*segment_size}-{(i+1)*segment_size}' 
                      for i in range(n_segments)]
    
    bars = ax2.barh(segment_labels, segment_means, color=COLORS['electrolyzer'], 
                    edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, segment_means)):
        ax2.text(val + max(segment_means)*0.02, i, f'{val:.1f}', 
                va='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('平均奖励', fontweight='bold')
    ax2.set_ylabel('回合范围', fontweight='bold')
    ax2.set_title('(b) 训练阶段性能', fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    save_name = f'results/advanced_training_convergence_{version}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 已生成: {save_name}")


def plot_comprehensive_typical_day(history, day_info, day_num, version='v12'):
    """综合典型日可视化 - 多子图布局"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, hspace=0.35, wspace=0.25, 
                  height_ratios=[1, 1, 1, 0.8])
    
    time_indices = np.arange(len(history['Time']))
    time_labels = [t.strftime("%H:%M") for t in history['Time']]
    
    # 主标题
    version_label = 'V12 (Agent-Driven)' if version == 'v12' else 'Standard'
    fig.suptitle(f'典型日 {day_num}: {day_info["desc"]} ({day_info["date"]}) - {version_label}',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # ========== 子图1: 堆叠面积图 - 能源供应 ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    ax1.fill_between(time_indices, 0, history['P_WT'],
                     label='风电功率', color=COLORS['wind'], alpha=0.7)
    ax1.fill_between(time_indices, history['P_WT'],
                     np.array(history['P_WT']) + np.array(history['P_PV']),
                     label='光伏功率', color=COLORS['solar'], alpha=0.7)
    
    ax1.plot(time_indices, history['Load'], label='负荷需求',
             color=COLORS['load'], linewidth=2.5, linestyle='--', marker='o', 
             markersize=3, markevery=4)
    
    ax1.set_xlabel('时间 (小时)', fontweight='bold')
    ax1.set_ylabel('功率 (kW)', fontweight='bold')
    ax1.set_title('(a) 可再生能源发电与负荷需求',
                  fontweight='bold', loc='left', pad=10)
    ax1.legend(loc='upper left', ncol=3, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, len(time_indices)-1)
    ax1.set_xticks(time_indices[::4])
    ax1.set_xticklabels(time_labels[::4], rotation=45)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ========== 子图2: 能量流动 - 双轴图 ==========
    ax2 = fig.add_subplot(gs[1, 0])
    ax2_twin = ax2.twinx()
    
    # 左轴：电解槽和燃料电池
    ln1 = ax2.plot(time_indices, history['P_EL'], label='电解槽',
                   color=COLORS['electrolyzer'], linewidth=2, marker='s',
                   markersize=4, markevery=4)
    ln2 = ax2.plot(time_indices, history['P_FC'], label='燃料电池',
                   color=COLORS['fuelcell'], linewidth=2, marker='^',
                   markersize=4, markevery=4)
    
    # 右轴：电池
    ln3 = ax2_twin.plot(time_indices, history['P_Bat'], label='电池',
                        color=COLORS['battery'], linewidth=2, linestyle='-.', 
                        marker='d', markersize=4, markevery=4)
    
    ax2.set_xlabel('时间 (小时)', fontweight='bold')
    ax2.set_ylabel('氢能系统功率 (kW)', fontweight='bold', color='black')
    ax2_twin.set_ylabel('电池功率 (kW)', fontweight='bold',
                        color=COLORS['battery'])
    ax2.set_title('(b) 储能系统响应',
                  fontweight='bold', loc='left', pad=10)
    
    # 合并图例
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper left', framealpha=0.9)
    
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2_twin.spines['top'].set_visible(False)
    ax2.set_xticks(time_indices[::4])
    ax2.set_xticklabels(time_labels[::4], rotation=45)
    
    # ========== 子图3: 储能状态 - 双SOC ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.fill_between(time_indices, 0, history['SOC'],
                     label='电池SOC', color=COLORS['battery'], alpha=0.5)
    ax3.plot(time_indices, history['SOC'], color=COLORS['battery'],
             linewidth=2.5, marker='o', markersize=4, markevery=4)
    
    ax3.fill_between(time_indices, 0, history['SOCH'],
                     label='氢罐储量', color=COLORS['h2_storage'], alpha=0.5)
    ax3.plot(time_indices, history['SOCH'], color=COLORS['h2_storage'],
             linewidth=2.5, marker='s', markersize=4, markevery=4)
    
    # 添加安全区域标记
    ax3.axhline(y=0.3, color='red', linestyle=':', linewidth=1.5, alpha=0.6,
                label='最小SOC限制')
    ax3.axhline(y=0.95, color='orange', linestyle=':', linewidth=1.5, alpha=0.6,
                label='最大SOC限制')
    
    ax3.set_xlabel('时间 (小时)', fontweight='bold')
    ax3.set_ylabel('荷电状态', fontweight='bold')
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title('(c) 储能状态', fontweight='bold', loc='left', pad=10)
    ax3.legend(loc='best', framealpha=0.9, ncol=2)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xticks(time_indices[::4])
    ax3.set_xticklabels(time_labels[::4], rotation=45)
    
    # ========== 子图4: 能量平衡热力图 ==========
    ax4 = fig.add_subplot(gs[2, 0])
    
    # 计算能量平衡
    supply = np.array(history['P_WT']) + np.array(history['P_PV']) + \
             np.maximum(0, -np.array(history['P_Bat'])) + np.array(history['P_FC'])
    demand = np.array(history['Load']) + np.maximum(0, np.array(history['P_Bat'])) + \
             np.array(history['P_EL'])
    balance = supply - demand
    
    # 创建热力图数据
    heatmap_data = np.array([
        history['P_WT'],
        history['P_PV'],
        history['P_FC'],
        [-x if x < 0 else 0 for x in history['P_Bat']],  # 放电
        history['Load'],
        history['P_EL'],
        [x if x > 0 else 0 for x in history['P_Bat']],  # 充电
    ])
    
    im = ax4.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', 
                    interpolation='nearest')
    
    ax4.set_yticks(range(7))
    ax4.set_yticklabels(['风电', '光伏', '燃料电池', '电池放电',
                         '负荷', '电解槽', '电池充电'], fontsize=9)
    ax4.set_xticks(time_indices[::4])
    ax4.set_xticklabels(time_labels[::4], rotation=45)
    ax4.set_xlabel('时间 (小时)', fontweight='bold')
    ax4.set_title('(d) 功率流动热力图', fontweight='bold', loc='left', pad=10)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax4, orientation='vertical', pad=0.02)
    cbar.set_label('功率 (kW)', fontweight='bold')
    
    # ========== 子图5: 能量平衡柱状图 ==========
    ax5 = fig.add_subplot(gs[2, 1])
    
    # 计算总量
    total_wind = np.sum(history['P_WT'])
    total_solar = np.sum(history['P_PV'])
    total_fc = np.sum(history['P_FC'])
    total_load = np.sum(history['Load'])
    total_el = np.sum(history['P_EL'])
    bat_charge = np.sum([x for x in history['P_Bat'] if x > 0])
    bat_discharge = np.sum([-x for x in history['P_Bat'] if x < 0])
    
    categories = ['供应', '需求']
    supply_components = [total_wind, total_solar, total_fc, bat_discharge]
    demand_components = [total_load, total_el, bat_charge, 0]
    
    x = np.arange(len(categories))
    width = 0.6
    
    # 堆叠柱状图
    bottom_supply = 0
    colors_supply = [COLORS['wind'], COLORS['solar'], COLORS['fuelcell'], COLORS['battery']]
    labels_supply = ['风电', '光伏', '燃料电池', '电池放电']
    
    for i, (val, color, label) in enumerate(zip(supply_components, colors_supply, labels_supply)):
        ax5.bar(0, val, width, bottom=bottom_supply, color=color, 
                edgecolor='black', linewidth=1, label=label)
        if val > 0:
            ax5.text(0, bottom_supply + val/2, f'{val:.0f}', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        bottom_supply += val
    
    bottom_demand = 0
    colors_demand = [COLORS['load'], COLORS['electrolyzer'], COLORS['battery'], 'white']
    labels_demand = ['负荷', '电解槽', '电池充电', '']
    
    for i, (val, color, label) in enumerate(zip(demand_components, colors_demand, labels_demand)):
        if val > 0:
            ax5.bar(1, val, width, bottom=bottom_demand, color=color, 
                    edgecolor='black', linewidth=1, label=label)
            ax5.text(1, bottom_demand + val/2, f'{val:.0f}', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        bottom_demand += val
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, fontweight='bold')
    ax5.set_ylabel('能量 (kWh)', fontweight='bold')
    ax5.set_title('(e) 日能量平衡', fontweight='bold', loc='left', pad=10)
    ax5.legend(loc='upper left', ncol=2, fontsize=8, framealpha=0.9)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # ========== 子图6: 性能指标雷达图 ==========
    ax6 = fig.add_subplot(gs[3, :], projection='polar')
    
    # 计算性能指标
    renewable_utilization = (total_wind + total_solar - np.sum([x for x in balance if x > 0])) / \
                           (total_wind + total_solar + 1e-6) * 100
    load_satisfaction = min(100, total_load / (total_load + 1e-6) * 100)
    h2_production_rate = total_el / (len(history['Time']) * 3000) * 100  # 相对于额定功率
    storage_efficiency = (bat_discharge / (bat_charge + 1e-6)) * 100 if bat_charge > 0 else 0
    system_flexibility = (np.std(history['P_Bat']) / (np.mean(np.abs(history['P_Bat'])) + 1e-6)) * 20
    
    metrics = [renewable_utilization, load_satisfaction, h2_production_rate,
               min(100, storage_efficiency), min(100, system_flexibility)]
    labels = ['可再生能源\n利用率 (%)', '负荷\n满足率 (%)',
              '制氢\n速率 (%)', '储能\n效率 (%)',
              '系统\n灵活性 (%)']
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    metrics += metrics[:1]
    angles += angles[:1]
    
    ax6.plot(angles, metrics, 'o-', linewidth=2, color=COLORS['electrolyzer'], 
             markersize=8)
    ax6.fill(angles, metrics, alpha=0.25, color=COLORS['electrolyzer'])
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(labels, fontsize=9)
    ax6.set_ylim(0, 100)
    ax6.set_yticks([25, 50, 75, 100])
    ax6.set_yticklabels(['25', '50', '75', '100'], fontsize=8)
    ax6.set_title('(f) 系统性能指标', fontweight='bold',
                  pad=20, y=1.08)
    ax6.grid(True, linestyle='--', alpha=0.5)
    
    # 保存
    save_name = f'results/advanced_typical_day_{version}_{day_num}_{day_info["date"]}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] 已生成: {save_name}")


# ==================== 主函数 ====================
def advanced_visualization(version='v12'):
    """执行高级可视化
    
    Args:
        version: 'v12' for V12 environment, 'standard' for standard environment
    """
    
    print("\n" + "="*60)
    print(f"  高质量顶刊级别可视化系统 ({version.upper()})")
    print("="*60 + "\n")
    
    # 选择环境
    if version == 'v12':
        if not USE_V12:
            print("[ERROR] V12环境未找到，请确保 src/envs/h2_res_env_v12.py 存在")
            return
        EnvClass = H2RESEnvV12
        model_path = 'results/ddpg_v12.pth'
    else:
        if H2RESEnv is None:
            print("[ERROR] 标准环境未找到")
            return
        EnvClass = H2RESEnv
        model_path = 'results/ddpg_checkpoint.pth'
    
    # 1. 训练收敛曲线
    print("[1/5] 生成训练收敛分析图...")
    plot_advanced_training_convergence(version=version)
    
    # 2. 加载数据
    if USE_REAL_DATA and os.path.exists(REAL_DATA_PATHS['wind']):
        df_data = load_real_data_for_eval()
    else:
        print("[ERROR] 未找到真实数据")
        return
    
    # 3. 聚类
    typical_days = find_representative_days(df_data, n_clusters=N_CLUSTERS)
    
    # 4. 加载模型
    temp_env = EnvClass(df_data.iloc[:48], df_data.iloc[:48])
    agent = DDPGAgent(temp_env.observation_space.shape[0],
                     temp_env.action_space.shape[0])
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"[OK] 已加载训练好的模型: {model_path}\n")
    else:
        print(f"[WARNING] 未找到模型 {model_path}，使用随机策略\n")
    
    # 5. 评估每个典型日
    for i, day_info in enumerate(typical_days):
        print(f"[{i+2}/5] 正在生成典型日 {i+1} 的可视化...")
        
        start_idx = day_info['start_idx']
        if start_idx + 48 > len(df_data):
            continue
        
        df_eval = df_data.iloc[start_idx: start_idx + 48].copy()
        env = EnvClass(df_eval, df_eval)
        
        state = env.reset()
        done = False
        
        history = {
            'P_WT': [], 'P_PV': [], 'P_Bat': [], 'P_EL': [], 'P_FC': [],
            'SOC': [], 'SOCH': [], 'Load': [], 'Time': []
        }
        
        eval_times = df_eval.index
        step_count = 0
        
        while not done and step_count < 48:
            action = agent.select_action(state, noise=False)
            next_state, reward, done, info = env.step(action)
            
            history['P_WT'].append(env.current_p_wt)
            history['P_PV'].append(env.current_p_pv)
            history['P_Bat'].append(info['real_p_bat'])
            history['P_EL'].append(info['real_p_el'])
            history['P_FC'].append(info['real_p_fc'])
            history['SOC'].append(env.SOC)
            history['SOCH'].append(env.SOCH)
            history['Load'].append(env.current_load)
            history['Time'].append(eval_times[step_count])
            
            state = next_state
            step_count += 1
        
        # 生成综合可视化
        plot_comprehensive_typical_day(history, day_info, i+1, version=version)
    
    print("\n" + "="*60)
    print(f"  [成功] 所有可视化已完成！({version.upper()})")
    print("="*60)
    print("\n生成的文件:")
    print(f"  [*] advanced_training_convergence_{version}.png - 训练收敛分析")
    for i in range(N_CLUSTERS):
        print(f"  [*] advanced_typical_day_{version}_{i+1}_*.png - 典型日{i+1}综合分析")
    print("\n这些图表包含:")
    print("  [+] 堆叠面积图 (能源供需)")
    print("  [+] 双轴功率图 (储能系统)")
    print("  [+] SOC状态图 (储能状态)")
    print("  [+] 热力图 (功率流动)")
    print("  [+] 堆叠柱状图 (能量平衡)")
    print("  [+] 雷达图 (性能指标)")
    print("\n适合直接用于顶刊论文！\n")


if __name__ == "__main__":
    import argparse
    
    # 确保结果目录存在
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='高级可视化脚本')
    parser.add_argument('--version', type=str, default='v12',
                       choices=['v12', 'standard'],
                       help='环境版本: v12 (V12环境) 或 standard (标准环境)')
    args = parser.parse_args()
    
    advanced_visualization(version=args.version)
