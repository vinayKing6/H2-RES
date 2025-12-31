"""
PSO (Particle Swarm Optimization) Baseline for H2-RES System

基于粒子群优化的能量管理基线策略
参考文献中的PSO方法，用于对比实验
"""

import numpy as np
from typing import Tuple, List


class PSOBaseline:
    """
    PSO基线策略：基于规则的能量管理
    
    策略逻辑（参考文献）：
    1. Pnet > 0（有剩余功率）：
       - 优先制氢（电解槽）
       - 剩余功率给电池充电
    2. Pnet ≤ 0（功率不足）：
       - 优先使用燃料电池
       - 不足部分由电池补充
    """
    
    def __init__(self, env):
        self.env = env
        
        # 系统参数
        self.P_EL_rated = env.P_EL_rated
        self.P_FC_rated = env.P_FC_rated
        self.E_bat_rated = env.E_bat_rated
        
        # V12环境兼容：最小功率约束（如果环境没有，使用默认值0）
        self.P_el_min = getattr(env, 'P_el_min', 0.0)
        self.P_fc_min = getattr(env, 'P_fc_min', 0.0)
        
        self.SOC_min = env.SOC_min
        self.SOC_max = env.SOC_max
        self.SOCH_min = env.SOCH_min
        self.SOCH_max = env.SOCH_max
        
    def get_action(self, obs: np.ndarray, p_gen: float, p_load: float, 
                   soc: float, soch: float) -> np.ndarray:
        """
        根据当前状态计算PSO策略的动作
        
        Args:
            obs: 观测值 [p_wt_norm, p_pv_norm, load_norm, soc, soch]
            p_gen: 总发电功率 (kW)
            p_load: 负荷功率 (kW)
            soc: 当前电池SOC
            soch: 当前氢气储罐SOCH
        
        Returns:
            action: [bat_action, el_action, fc_action] (归一化到[-1, 1])
        """
        # 计算净功率
        p_net = p_gen - p_load
        
        # 初始化动作
        p_bat_target = 0.0  # 电池功率（正=充电，负=放电）
        p_el_target = 0.0   # 电解槽功率
        p_fc_target = 0.0   # 燃料电池功率
        
        # ========== 场景1：Pnet > 0（有剩余功率）==========
        if p_net > 0:
            # 燃料电池关闭
            p_fc_target = 0.0
            
            # 优先制氢（如果储罐未满）
            if soch < self.SOCH_max:
                if p_net >= self.P_el_min:
                    # 剩余功率足够启动电解槽
                    p_el_target = min(p_net, self.P_EL_rated)
                    p_surplus_after_el = p_net - p_el_target
                    
                    # 剩余功率给电池充电（如果电池未满）
                    if soc < self.SOC_max and p_surplus_after_el > 0:
                        p_bat_target = min(p_surplus_after_el, self.E_bat_rated)
                else:
                    # 剩余功率不足启动电解槽，全部给电池
                    if soc < self.SOC_max:
                        p_bat_target = min(p_net, self.E_bat_rated)
            else:
                # 储罐已满，全部给电池
                if soc < self.SOC_max:
                    p_bat_target = min(p_net, self.E_bat_rated)
        
        # ========== 场景2：Pnet ≤ 0（功率不足）==========
        else:
            # 电解槽关闭
            p_el_target = 0.0
            
            p_deficit = abs(p_net)  # 缺电量
            
            # 如果有氢气，优先使用燃料电池
            if soch > self.SOCH_min:
                if p_deficit >= self.P_fc_min:
                    # 缺电量足够启动燃料电池
                    p_fc_target = min(p_deficit, self.P_FC_rated)
                    p_deficit_after_fc = p_deficit - p_fc_target
                    
                    # 不足部分由电池补充
                    if p_deficit_after_fc > 0 and soc > self.SOC_min:
                        p_bat_target = -min(p_deficit_after_fc, self.E_bat_rated)
                else:
                    # 缺电量不足启动燃料电池，仅用电池
                    if soc > self.SOC_min:
                        p_bat_target = -min(p_deficit, self.E_bat_rated)
            else:
                # 无氢气，仅用电池
                if soc > self.SOC_min:
                    p_bat_target = -min(p_deficit, self.E_bat_rated)
        
        # ========== 转换为归一化动作 ==========
        # bat_action: [-1, 1] → [-E_bat_rated, E_bat_rated]
        bat_action = p_bat_target / self.E_bat_rated
        bat_action = np.clip(bat_action, -1.0, 1.0)
        
        # el_action: [-1, 1] → [0, P_EL_rated]
        el_action = (p_el_target / self.P_EL_rated) * 2.0 - 1.0
        el_action = np.clip(el_action, -1.0, 1.0)
        
        # fc_action: [-1, 1] → [0, P_FC_rated]
        fc_action = (p_fc_target / self.P_FC_rated) * 2.0 - 1.0
        fc_action = np.clip(fc_action, -1.0, 1.0)
        
        return np.array([bat_action, el_action, fc_action], dtype=np.float32)
    
    def evaluate(self, start_step: int = 0, num_steps: int = 24) -> dict:
        """
        评估PSO策略在指定时间段的性能
        
        Args:
            start_step: 起始时间步
            num_steps: 评估步数（默认24小时）
        
        Returns:
            results: 评估结果字典
        """
        obs = self.env.reset(start_step=start_step)
        
        total_reward = 0.0
        total_h2_prod = 0.0
        total_unmet = 0.0
        
        h2_prod_history = []
        soc_history = [self.env.SOC]
        soch_history = [self.env.SOCH]
        
        for step in range(num_steps):
            # 获取当前状态
            p_gen = self.env.current_p_wt + self.env.current_p_pv
            p_load = self.env.current_load
            soc = self.env.SOC
            soch = self.env.SOCH
            
            # 计算PSO动作
            action = self.get_action(obs, p_gen, p_load, soc, soch)
            
            # 执行动作
            obs, reward, done, info = self.env.step(action)
            
            # 记录结果
            total_reward += reward
            total_h2_prod += info['h2_prod']
            total_unmet += info['p_unmet']
            
            h2_prod_history.append(info['h2_prod'])
            soc_history.append(info['soc'])
            soch_history.append(info['soch'])
            
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'total_h2_prod': total_h2_prod,
            'total_unmet': total_unmet,
            'avg_reward': total_reward / num_steps,
            'avg_h2_prod': total_h2_prod / num_steps,
            'avg_unmet': total_unmet / num_steps,
            'h2_prod_history': h2_prod_history,
            'soc_history': soc_history,
            'soch_history': soch_history
        }


def test_pso_baseline():
    """测试PSO基线策略"""
    import sys
    sys.path.append('.')
    
    from src.envs.h2_res_env import H2RESEnv
    from src.data.nasa_loader import load_nasa_data
    
    # 加载数据
    df_weather, df_load = load_nasa_data()
    
    # 创建环境
    env = H2RESEnv(df_weather, df_load)
    
    # 创建PSO基线
    pso = PSOBaseline(env)
    
    # 评估（从第0步开始，评估24小时）
    results = pso.evaluate(start_step=0, num_steps=24)
    
    print("=" * 60)
    print("PSO Baseline Evaluation Results")
    print("=" * 60)
    print(f"Total Reward: {results['total_reward']:.2f}")
    print(f"Total H2 Production: {results['total_h2_prod']:.2f} kg")
    print(f"Total Unmet Load: {results['total_unmet']:.2f} kW")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Average H2 Production: {results['avg_h2_prod']:.4f} kg/h")
    print(f"Average Unmet Load: {results['avg_unmet']:.2f} kW")
    print("=" * 60)


if __name__ == '__main__':
    test_pso_baseline()