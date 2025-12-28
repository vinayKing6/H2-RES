"""
训练模型评估脚本
测试训练后模型的各项性能指标和物理合理性
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from src.envs.h2_res_env import H2RESEnv
from src.algos.ddpg import DDPGAgent
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class ModelEvaluator:
    """训练模型评估器"""
    
    def __init__(self, df_data, model_path='results/ddpg_checkpoint.pth'):
        """
        初始化评估器
        
        Args:
            df_data: 包含气象和负荷数据的DataFrame
            model_path: 训练好的模型路径
        """
        self.env = H2RESEnv(df_data, df_data)
        self.model_path = model_path
        self.df_data = df_data
        
        # 加载模型
        self.agent = DDPGAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0]
        )
        
        try:
            checkpoint = torch.load(model_path)
            self.agent.actor.load_state_dict(checkpoint['actor'])
            self.agent.critic.load_state_dict(checkpoint['critic'])
            print(f"[OK] 成功加载模型: {model_path}")
        except Exception as e:
            print(f"[ERROR] 加载模型失败: {e}")
            raise
        
        # 评估指标
        self.metrics = {
            'episodes': [],
            'total_rewards': [],
            'avg_rewards': [],
            'h2_production': [],
            'electrolyzer_startups': [],
            'electrolyzer_shutdowns': [],
            'electrolyzer_runtime_hours': [],
            'avg_electrolyzer_power': [],
            'battery_cycles': [],
            'avg_battery_power': [],
            'load_satisfaction_rate': [],
            'renewable_utilization_rate': [],
            'dump_rate': [],
            'soc_violations': [],
            'soch_violations': [],
        }
    
    def run_episode(self, episode_num=0, verbose=False):
        """
        运行一个完整的episode并收集数据
        
        Args:
            episode_num: episode编号
            verbose: 是否打印详细信息
            
        Returns:
            episode_data: 包含该episode所有数据的字典
        """
        state = self.env.reset()
        done = False
        
        # Episode数据
        episode_data = {
            'rewards': [],
            'electrolyzer_power': [],
            'battery_power': [],
            'fuel_cell_power': [],
            'h2_produced': [],
            'soc': [],
            'soch': [],
            'load': [],
            'renewable': [],
            'load_met': [],
            'dump': [],
        }
        
        step = 0
        while not done:
            # 获取动作
            action = self.agent.select_action(state, noise=False)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 记录数据
            episode_data['rewards'].append(reward)
            episode_data['electrolyzer_power'].append(info.get('p_el', 0))
            episode_data['battery_power'].append(info.get('p_bat', 0))
            episode_data['fuel_cell_power'].append(info.get('p_fc', 0))
            episode_data['h2_produced'].append(info.get('h2_produced', 0))
            episode_data['soc'].append(info.get('SOC', 0))
            episode_data['soch'].append(info.get('SOCH', 0))
            episode_data['load'].append(info.get('p_load', 0))
            episode_data['renewable'].append(info.get('p_wind', 0) + info.get('p_solar', 0))
            episode_data['load_met'].append(info.get('load_met', 0))
            episode_data['dump'].append(info.get('p_dump', 0))
            
            state = next_state
            step += 1
        
        if verbose:
            print(f"\nEpisode {episode_num} 完成，共 {step} 步")
        
        return episode_data
    
    def analyze_episode(self, episode_data):
        """
        分析单个episode的数据
        
        Args:
            episode_data: episode数据字典
            
        Returns:
            metrics: 该episode的评估指标
        """
        metrics = {}
        
        # 转换为numpy数组
        el_power = np.array(episode_data['electrolyzer_power'])
        bat_power = np.array(episode_data['battery_power'])
        h2_produced = np.array(episode_data['h2_produced'])
        soc = np.array(episode_data['soc'])
        soch = np.array(episode_data['soch'])
        load = np.array(episode_data['load'])
        renewable = np.array(episode_data['renewable'])
        load_met = np.array(episode_data['load_met'])
        dump = np.array(episode_data['dump'])
        rewards = np.array(episode_data['rewards'])
        
        # 1. 奖励指标
        metrics['total_reward'] = np.sum(rewards)
        metrics['avg_reward'] = np.mean(rewards)
        metrics['min_reward'] = np.min(rewards)
        metrics['max_reward'] = np.max(rewards)
        
        # 2. 制氢指标
        metrics['total_h2_production'] = np.sum(h2_produced)  # kg
        metrics['avg_electrolyzer_power'] = np.mean(el_power[el_power > 0]) if np.any(el_power > 0) else 0
        
        # 电解槽启停次数
        el_on = (el_power >= 300).astype(int)  # 电解槽最小功率300kW
        el_diff = np.diff(el_on)
        metrics['electrolyzer_startups'] = np.sum(el_diff == 1)
        metrics['electrolyzer_shutdowns'] = np.sum(el_diff == -1)
        metrics['electrolyzer_runtime_hours'] = np.sum(el_on)
        
        # 3. 电池指标
        bat_charge = bat_power > 0
        bat_discharge = bat_power < 0
        metrics['battery_charge_hours'] = np.sum(bat_charge)
        metrics['battery_discharge_hours'] = np.sum(bat_discharge)
        metrics['avg_battery_power'] = np.mean(np.abs(bat_power[bat_power != 0])) if np.any(bat_power != 0) else 0
        
        # 电池循环次数（简化计算：充放电切换次数/2）
        bat_state = np.zeros_like(bat_power)
        bat_state[bat_charge] = 1
        bat_state[bat_discharge] = -1
        bat_switches = np.sum(np.abs(np.diff(bat_state)) > 0)
        metrics['battery_cycles'] = bat_switches / 2
        
        # 4. 负荷满足率
        metrics['load_satisfaction_rate'] = np.mean(load_met) * 100  # %
        
        # 5. 可再生能源利用率
        total_renewable = np.sum(renewable)
        total_dump = np.sum(dump)
        metrics['renewable_utilization_rate'] = (1 - total_dump / total_renewable) * 100 if total_renewable > 0 else 0
        metrics['dump_rate'] = (total_dump / total_renewable) * 100 if total_renewable > 0 else 0
        
        # 6. SOC/SOCH违规
        metrics['soc_violations'] = np.sum((soc < 0.3) | (soc > 0.95))
        metrics['soch_violations'] = np.sum((soch < 0.2) | (soch > 0.9))
        metrics['avg_soc'] = np.mean(soc)
        metrics['avg_soch'] = np.mean(soch)
        
        return metrics
    
    def evaluate(self, num_episodes=10, verbose=True):
        """
        评估模型性能
        
        Args:
            num_episodes: 评估的episode数量
            verbose: 是否打印详细信息
        """
        print(f"\n{'='*80}")
        print(f"  开始评估训练模型（{num_episodes} episodes）")
        print(f"{'='*80}\n")
        
        all_episode_data = []
        
        for ep in range(num_episodes):
            episode_data = self.run_episode(episode_num=ep+1, verbose=False)
            metrics = self.analyze_episode(episode_data)
            
            # 保存数据
            all_episode_data.append(episode_data)
            
            # 记录指标
            self.metrics['episodes'].append(ep + 1)
            self.metrics['total_rewards'].append(metrics['total_reward'])
            self.metrics['avg_rewards'].append(metrics['avg_reward'])
            self.metrics['h2_production'].append(metrics['total_h2_production'])
            self.metrics['electrolyzer_startups'].append(metrics['electrolyzer_startups'])
            self.metrics['electrolyzer_shutdowns'].append(metrics['electrolyzer_shutdowns'])
            self.metrics['electrolyzer_runtime_hours'].append(metrics['electrolyzer_runtime_hours'])
            self.metrics['avg_electrolyzer_power'].append(metrics['avg_electrolyzer_power'])
            self.metrics['battery_cycles'].append(metrics['battery_cycles'])
            self.metrics['avg_battery_power'].append(metrics['avg_battery_power'])
            self.metrics['load_satisfaction_rate'].append(metrics['load_satisfaction_rate'])
            self.metrics['renewable_utilization_rate'].append(metrics['renewable_utilization_rate'])
            self.metrics['dump_rate'].append(metrics['dump_rate'])
            self.metrics['soc_violations'].append(metrics['soc_violations'])
            self.metrics['soch_violations'].append(metrics['soch_violations'])
            
            if verbose:
                print(f"Episode {ep+1}/{num_episodes}: "
                      f"奖励={metrics['total_reward']:.1f}, "
                      f"制氢={metrics['total_h2_production']:.1f}kg, "
                      f"启停={metrics['electrolyzer_startups']}次")
        
        # 打印统计结果
        self.print_summary()
        
        # 生成评估报告
        self.generate_report()
        
        return all_episode_data
    
    def print_summary(self):
        """打印评估摘要"""
        print(f"\n{'='*80}")
        print(f"  评估结果摘要")
        print(f"{'='*80}\n")
        
        # 1. 奖励指标
        print("【1. 奖励指标】")
        print(f"  平均总奖励: {np.mean(self.metrics['total_rewards']):.2f} ± {np.std(self.metrics['total_rewards']):.2f}")
        print(f"  平均步奖励: {np.mean(self.metrics['avg_rewards']):.2f} ± {np.std(self.metrics['avg_rewards']):.2f}")
        print(f"  奖励范围: [{np.min(self.metrics['total_rewards']):.2f}, {np.max(self.metrics['total_rewards']):.2f}]")
        
        # 2. 制氢性能
        print("\n【2. 制氢性能】")
        avg_h2 = np.mean(self.metrics['h2_production'])
        print(f"  平均制氢量: {avg_h2:.2f} ± {np.std(self.metrics['h2_production']):.2f} kg/天")
        print(f"  制氢量范围: [{np.min(self.metrics['h2_production']):.2f}, {np.max(self.metrics['h2_production']):.2f}] kg/天")
        print(f"  平均电解槽功率: {np.mean(self.metrics['avg_electrolyzer_power']):.2f} kW")
        print(f"  平均运行时长: {np.mean(self.metrics['electrolyzer_runtime_hours']):.1f} 小时/天")
        
        # 评估制氢量是否合理（假设目标>50kg/天）
        if avg_h2 < 30:
            print(f"  [WARNING] 警告: 制氢量过低（<30kg/天），需要调整奖励函数")
        elif avg_h2 < 50:
            print(f"  [NOTICE] 注意: 制氢量偏低（<50kg/天），可以进一步优化")
        else:
            print(f"  [OK] 制氢量正常（≥50kg/天）")
        
        # 3. 电解槽启停
        print("\n【3. 电解槽启停分析】")
        avg_startups = np.mean(self.metrics['electrolyzer_startups'])
        avg_shutdowns = np.mean(self.metrics['electrolyzer_shutdowns'])
        print(f"  平均启动次数: {avg_startups:.2f} 次/天")
        print(f"  平均关闭次数: {avg_shutdowns:.2f} 次/天")
        
        # 评估启停频率（理想情况：≤2次/天）
        if avg_startups > 5:
            print(f"  [WARNING] 警告: 启停过于频繁（>5次/天），需要增大启停惩罚")
        elif avg_startups > 3:
            print(f"  [NOTICE] 注意: 启停较频繁（>3次/天），建议优化")
        else:
            print(f"  [OK] 启停频率正常（≤3次/天）")
        
        # 4. 电池性能
        print("\n【4. 电池性能】")
        print(f"  平均循环次数: {np.mean(self.metrics['battery_cycles']):.2f} 次/天")
        print(f"  平均充电时长: {np.mean([m for m in self.metrics['battery_cycles']]):.1f} 小时/天")
        print(f"  平均电池功率: {np.mean(self.metrics['avg_battery_power']):.2f} kW")
        
        # 5. 负荷满足
        print("\n【5. 负荷满足率】")
        avg_load_sat = np.mean(self.metrics['load_satisfaction_rate'])
        print(f"  平均负荷满足率: {avg_load_sat:.2f}%")
        
        if avg_load_sat < 95:
            print(f"  [WARNING] 警告: 负荷满足率过低（<95%），系统可靠性不足")
        elif avg_load_sat < 98:
            print(f"  [NOTICE] 注意: 负荷满足率偏低（<98%），建议优化")
        else:
            print(f"  [OK] 负荷满足率正常（≥98%）")
        
        # 6. 可再生能源利用
        print("\n【6. 可再生能源利用】")
        avg_util = np.mean(self.metrics['renewable_utilization_rate'])
        avg_dump = np.mean(self.metrics['dump_rate'])
        print(f"  平均利用率: {avg_util:.2f}%")
        print(f"  平均弃电率: {avg_dump:.2f}%")
        
        if avg_dump > 20:
            print(f"  [WARNING] 警告: 弃电率过高（>20%），需要增大弃电惩罚")
        elif avg_dump > 10:
            print(f"  [NOTICE] 注意: 弃电率偏高（>10%），建议优化")
        else:
            print(f"  [OK] 弃电率正常（≤10%）")
        
        # 7. 物理约束违规
        print("\n【7. 物理约束检查】")
        total_soc_violations = np.sum(self.metrics['soc_violations'])
        total_soch_violations = np.sum(self.metrics['soch_violations'])
        print(f"  SOC违规次数: {total_soc_violations}")
        print(f"  SOCH违规次数: {total_soch_violations}")
        
        if total_soc_violations > 0 or total_soch_violations > 0:
            print(f"  [WARNING] 警告: 存在物理约束违规，检查环境代码")
        else:
            print(f"  [OK] 无物理约束违规")
        
        print(f"\n{'='*80}\n")
    
    def generate_report(self, save_path='results/evaluation_report.png'):
        """生成可视化评估报告"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('训练模型评估报告', fontsize=16, fontweight='bold')
        
        # 1. 奖励分布
        ax = axes[0, 0]
        ax.hist(self.metrics['total_rewards'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(self.metrics['total_rewards']), color='red', linestyle='--', 
                   label=f'均值={np.mean(self.metrics["total_rewards"]):.1f}')
        ax.set_xlabel('总奖励')
        ax.set_ylabel('频数')
        ax.set_title('(a) 奖励分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 制氢量分布
        ax = axes[0, 1]
        ax.hist(self.metrics['h2_production'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(self.metrics['h2_production']), color='red', linestyle='--',
                   label=f'均值={np.mean(self.metrics["h2_production"]):.1f}kg')
        ax.set_xlabel('制氢量 (kg/天)')
        ax.set_ylabel('频数')
        ax.set_title('(b) 制氢量分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 电解槽启停次数
        ax = axes[0, 2]
        x = np.arange(len(self.metrics['episodes']))
        ax.bar(x, self.metrics['electrolyzer_startups'], alpha=0.7, label='启动')
        ax.bar(x, self.metrics['electrolyzer_shutdowns'], alpha=0.7, label='关闭')
        ax.axhline(3, color='red', linestyle='--', label='目标≤3次')
        ax.set_xlabel('Episode')
        ax.set_ylabel('次数')
        ax.set_title('(c) 电解槽启停次数')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 电解槽运行时长
        ax = axes[1, 0]
        ax.plot(self.metrics['episodes'], self.metrics['electrolyzer_runtime_hours'], 
                marker='o', markersize=4, linewidth=1.5)
        ax.axhline(np.mean(self.metrics['electrolyzer_runtime_hours']), 
                   color='red', linestyle='--', 
                   label=f'均值={np.mean(self.metrics["electrolyzer_runtime_hours"]):.1f}h')
        ax.set_xlabel('Episode')
        ax.set_ylabel('运行时长 (小时)')
        ax.set_title('(d) 电解槽运行时长')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 负荷满足率
        ax = axes[1, 1]
        ax.plot(self.metrics['episodes'], self.metrics['load_satisfaction_rate'],
                marker='o', markersize=4, linewidth=1.5, color='purple')
        ax.axhline(98, color='red', linestyle='--', label='目标≥98%')
        ax.set_xlabel('Episode')
        ax.set_ylabel('负荷满足率 (%)')
        ax.set_title('(e) 负荷满足率')
        ax.set_ylim([90, 101])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 弃电率
        ax = axes[1, 2]
        ax.plot(self.metrics['episodes'], self.metrics['dump_rate'],
                marker='o', markersize=4, linewidth=1.5, color='orange')
        ax.axhline(10, color='red', linestyle='--', label='目标≤10%')
        ax.set_xlabel('Episode')
        ax.set_ylabel('弃电率 (%)')
        ax.set_title('(f) 弃电率')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. 电池循环次数
        ax = axes[2, 0]
        ax.hist(self.metrics['battery_cycles'], bins=20, alpha=0.7, color='cyan', edgecolor='black')
        ax.axvline(np.mean(self.metrics['battery_cycles']), color='red', linestyle='--',
                   label=f'均值={np.mean(self.metrics["battery_cycles"]):.1f}')
        ax.set_xlabel('循环次数 (次/天)')
        ax.set_ylabel('频数')
        ax.set_title('(g) 电池循环次数分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 8. 关键指标雷达图
        ax = axes[2, 1]
        categories = ['制氢\n(归一化)', '负荷\n满足率', '可再生\n利用率', '启停\n频率', '奖励\n(归一化)']
        
        # 归一化指标到0-1
        values = [
            np.mean(self.metrics['h2_production']) / 100,  # 假设100kg为满分
            np.mean(self.metrics['load_satisfaction_rate']) / 100,
            np.mean(self.metrics['renewable_utilization_rate']) / 100,
            1 - min(np.mean(self.metrics['electrolyzer_startups']) / 10, 1),  # 启停越少越好
            (np.mean(self.metrics['total_rewards']) + 5000) / 10000,  # 假设奖励范围[-5000, 5000]
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax = plt.subplot(3, 3, 8, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('(h) 综合性能雷达图')
        ax.grid(True)
        
        # 9. 性能评分
        ax = axes[2, 2]
        ax.axis('off')
        
        # 计算综合评分
        score_h2 = min(np.mean(self.metrics['h2_production']) / 50 * 100, 100)
        score_startup = max(100 - np.mean(self.metrics['electrolyzer_startups']) * 20, 0)
        score_load = np.mean(self.metrics['load_satisfaction_rate'])
        score_util = np.mean(self.metrics['renewable_utilization_rate'])
        total_score = (score_h2 * 0.3 + score_startup * 0.2 + score_load * 0.3 + score_util * 0.2)
        
        # 显示评分
        ax.text(0.5, 0.9, '(i) 综合评分', ha='center', va='top', fontsize=12, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.5, 0.7, f'{total_score:.1f} / 100', ha='center', va='center', fontsize=36,
                fontweight='bold', color='blue', transform=ax.transAxes)
        
        # 评级
        if total_score >= 90:
            grade = 'A (优秀)'
            color = 'green'
        elif total_score >= 80:
            grade = 'B (良好)'
            color = 'blue'
        elif total_score >= 70:
            grade = 'C (合格)'
            color = 'orange'
        else:
            grade = 'D (需改进)'
            color = 'red'
        
        ax.text(0.5, 0.5, grade, ha='center', va='center', fontsize=16,
                fontweight='bold', color=color, transform=ax.transAxes)
        
        # 分项得分
        ax.text(0.1, 0.3, f'制氢性能: {score_h2:.1f}', ha='left', va='center', fontsize=10,
                transform=ax.transAxes)
        ax.text(0.1, 0.2, f'启停控制: {score_startup:.1f}', ha='left', va='center', fontsize=10,
                transform=ax.transAxes)
        ax.text(0.1, 0.1, f'负荷满足: {score_load:.1f}', ha='left', va='center', fontsize=10,
                transform=ax.transAxes)
        ax.text(0.1, 0.0, f'能源利用: {score_util:.1f}', ha='left', va='center', fontsize=10,
                transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[OK] 评估报告已保存: {save_path}")
        plt.close()


def load_data_from_train():
    """
    从train.py复制的数据加载函数
    """
    DATA_DIR = os.path.join(current_dir, 'src', 'data')
    REAL_DATA_PATHS = {
        'wind': os.path.join(DATA_DIR, 'Wind farm site 1 (Nominal capacity-99MW).xlsx'),
        'solar': os.path.join(DATA_DIR, 'Solar station site 1 (Nominal capacity-50MW).xlsx')
    }
    
    def generate_simple_load(hours):
        """生成简单的合成负荷数据"""
        time = np.arange(hours)
        base_load = 2000  # kW
        daily_pattern = 1000 * (np.exp(-((time % 24 - 9) ** 2) / 10) + np.exp(-((time % 24 - 19) ** 2) / 10))
        load = base_load + daily_pattern + np.random.normal(0, 100, hours)
        return load
    
    def read_data_robust(filepath):
        """智能读取数据"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到文件: {filepath}")
        
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            print(f"读取 Excel 文件: {os.path.basename(filepath)} ...")
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
            raise ValueError(f"无法读取 CSV 文件: {filepath}")
    
    print("正在加载数据...")
    
    # 读取数据
    df_wind_raw = read_data_robust(REAL_DATA_PATHS['wind'])
    df_solar_raw = read_data_robust(REAL_DATA_PATHS['solar'])
    
    # 清理列名
    df_wind_raw.columns = df_wind_raw.columns.str.strip()
    df_solar_raw.columns = df_solar_raw.columns.str.strip()
    
    # 查找列名
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
        raise ValueError("无法找到所需的列")
    
    # 提取数据
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
    
    # 重采样为1小时
    df_hourly = df_merged.resample('1H').mean().dropna()
    
    # 确保从00:00开始
    first_midnight_idx = np.where(df_hourly.index.hour == 0)[0]
    if len(first_midnight_idx) > 0:
        start_idx = first_midnight_idx[0]
        df_hourly = df_hourly.iloc[start_idx:]
    
    # 生成负荷
    df_hourly['load'] = generate_simple_load(len(df_hourly))
    
    print(f"[OK] 数据加载完成: {len(df_hourly)} 小时")
    return df_hourly


def main():
    """主函数"""
    print("\n" + "="*80)
    print("  训练模型评估工具")
    print("="*80 + "\n")
    
    # 加载数据
    try:
        df_data = load_data_from_train()
    except Exception as e:
        print(f"[ERROR] 数据加载失败: {e}")
        return
    
    # 创建评估器
    try:
        evaluator = ModelEvaluator(df_data, model_path='results/ddpg_checkpoint.pth')
    except Exception as e:
        print(f"[ERROR] 评估器初始化失败: {e}")
        return
    
    # 运行评估（10个episodes）
    evaluator.evaluate(num_episodes=10, verbose=True)
    
    print("\n评估完成！")
    print("- 查看详细报告: results/evaluation_report.png")
    print("- 根据评估结果调整奖励函数参数")


if __name__ == '__main__':
    main()