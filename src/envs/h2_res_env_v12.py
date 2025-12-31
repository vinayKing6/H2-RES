
import gym
import numpy as np
from gym import spaces
from collections import deque


class H2RESEnv(gym.Env):
    """
    H2-RES System Environment (V12: Gemini's Approach - Agent-Driven Allocation)

    Version History:
    - V1-V10: Various attempts with different reward functions and allocation logic
    - V11: Complete redesign for training stability (Failed: 剩余功率计算错误)
    - V11.1: Fixed surplus power calculation (Failed: 制氢量极低，0.45 kg/天)
    - V12: Gemini's approach - Agent decides allocation, environment only enforces total power constraint
    
    V12 Core Philosophy (Gemini's Recommendation):
    ✅ Agent freely requests power allocation
    ✅ Environment only enforces total power constraint (proportional scaling)
    ✅ Load is soft constraint (penalty-based), not hard constraint
    ✅ Agent learns to balance: hydrogen production vs load satisfaction
    
    V11/V11.1 Failure Analysis:
    - 问题：硬约束"负荷优先"导致缺电时无法制氢
    - 结果：春季/秋季完全无法制氢（75%训练时间浪费）
    - 根本原因：设计理念错误，限制了Agent学习空间
    
    V12 Key Changes:
    1. ✅ 移除"负荷优先"硬约束
    2. ✅ Agent自由分配功率（负荷、制氢、储能）
    3. ✅ 环境只负责总功率约束（按比例缩放）
    4. ✅ 负荷通过惩罚引导（软约束）
    5. ✅ 让Agent学习最优平衡策略
    
    Expected Results:
    - Agent可以学习"什么时候可以牺牲一点负荷来制氢"
    - 制氢量显著提升（>200 kg/天）
    - 训练稳定收敛
    - 所有季节都可以制氢
    """

    def __init__(self, df_weather, df_load, history_hours=3):
        super(H2RESEnv, self).__init__()

        # Data
        self.df_weather = df_weather.reset_index(drop=True)
        self.df_load = df_load.reset_index(drop=True)
        self.max_steps = len(self.df_weather)
        self.current_step = 0

        # --- System Parameters (Physical Units) ---
        self.P_WT_rated = 4000.0  # kW (风电额定功率)
        self.P_PV_rated = 3000.0  # kW (光伏额定功率)
        self.E_bat_rated = 1000.0  # kWh (电池额定容量)
        self.P_EL_rated = 3000.0  # kW (电解槽额定功率)
        self.P_FC_rated = 300.0  # kW (燃料电池额定功率)
        self.M_HS_max = 2000.0  # kg (氢气储罐最大容量)

        # --- Constraints Parameters ---
        self.SOC_min = 0.3
        self.SOC_max = 0.95
        self.SOCH_min = 0.2
        self.SOCH_max = 0.9
        self.P_fc_min = 0.1 * self.P_FC_rated  # 燃料电池最小运行功率约束

        # Efficiency Constants (效率参数)
        self.eta_bat_ch = 0.95  # 电池充电效率
        self.eta_bat_dis = 0.95  # 电池放电效率
        self.dt = 1.0  # 时间步长 (小时)

        # --- Fuel Cell Parameters ---
        self.FC_STARTUP_TIME = 1.0  # 启动时间（小时）
        
        # --- History Buffer ---
        self.history_hours = history_hours
        self.p_wt_history = deque(maxlen=history_hours)
        self.p_pv_history = deque(maxlen=history_hours)
        
        # --- Observation Space (10D) ---
        # [p_wt, p_pv, load, p_wt_hist_avg, p_pv_hist_avg, 
        #  SOC, SOCH, net_load, re_ratio, hour_norm]
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Internal Variables
        self.SOC = 0.5
        self.SOCH = 0.5
        self.H2_storage_kg = self.SOCH * self.M_HS_max

        # Raw tracking
        self.current_p_wt = 0.0
        self.current_p_pv = 0.0
        self.current_load = 0.0
        
        # 燃料电池状态追踪
        self.fc_is_running = False
        self.fc_startup_progress = 0.0  # 启动进度 [0, 1]

    def reset(self, start_step=0, init_soc=None, init_soch=None):
        """
        重置环境
        Args:
            start_step: 起始时间步（支持从任意时刻开始）
            init_soc: 初始电池SOC（None则使用默认值0.5）
            init_soch: 初始氢气储罐SOCH（None则使用默认值0.5）
        """
        self.current_step = start_step
        self.SOC = init_soc if init_soc is not None else 0.5
        self.SOCH = init_soch if init_soch is not None else 0.5
        self.H2_storage_kg = self.SOCH * self.M_HS_max
        
        # 重置燃料电池状态追踪
        self.fc_is_running = False
        self.fc_startup_progress = 0.0
        
        # 初始化历史缓冲区
        self.p_wt_history.clear()
        self.p_pv_history.clear()
        
        # 用初始值填充历史缓冲区
        step = min(start_step, len(self.df_weather) - 1)
        wind = self.df_weather.iloc[step]['wind_speed']
        irr = self.df_weather.iloc[step]['irradiance']
        temp = self.df_weather.iloc[step]['temperature']
        
        p_wt_init = self._calc_wind_power(wind)
        p_pv_init = self._calc_pv_power(irr, temp)
        
        for _ in range(self.history_hours):
            self.p_wt_history.append(p_wt_init)
            self.p_pv_history.append(p_pv_init)
        
        return self._get_obs()

    def _get_obs(self):
        """
        观测空间（10维）
        
        返回：
        [0] p_wt_norm: 当前风电功率（归一化）
        [1] p_pv_norm: 当前光伏功率（归一化）
        [2] load_norm: 当前负荷（归一化）
        [3] p_wt_hist_avg: 过去3h风电平均功率（归一化）
        [4] p_pv_hist_avg: 过去3h光伏平均功率（归一化）
        [5] SOC: 电池荷电状态
        [6] SOCH: 氢罐荷电状态
        [7] net_load_norm: 净负荷（load-gen，归一化）
        [8] re_ratio: 可再生能源占比
        [9] hour_norm: 小时归一化 [0,1]
        """
        step = min(self.current_step, len(self.df_weather) - 1)

        wind_speed = self.df_weather.iloc[step]['wind_speed']
        irradiance = self.df_weather.iloc[step]['irradiance']
        temp = self.df_weather.iloc[step]['temperature']
        load = self.df_load.iloc[step]['load']

        p_wt = self._calc_wind_power(wind_speed)
        p_pv = self._calc_pv_power(irradiance, temp)

        self.current_p_wt = p_wt
        self.current_p_pv = p_pv
        self.current_load = load
        
        # 更新历史缓冲区
        self.p_wt_history.append(p_wt)
        self.p_pv_history.append(p_pv)
        
        # 计算历史平均值
        p_wt_hist_avg = np.mean(self.p_wt_history) if len(self.p_wt_history) > 0 else p_wt
        p_pv_hist_avg = np.mean(self.p_pv_history) if len(self.p_pv_history) > 0 else p_pv
        
        # 时间特征
        try:
            hour = self.df_weather.index[step].hour
        except:
            hour = step % 24
        
        # 净负荷和可再生能源占比
        p_gen = p_wt + p_pv
        net_load = load - p_gen
        net_load_norm = np.clip(net_load / (self.P_WT_rated + self.P_PV_rated), -1, 1)
        
        re_ratio = p_gen / max(load, 1e-6) if load > 0 else 1.0
        re_ratio = np.clip(re_ratio, 0, 2.0) / 2.0  # 归一化到[0,1]

        obs = np.array([
            p_wt / self.P_WT_rated,                    # [0] 当前风电
            p_pv / self.P_PV_rated,                    # [1] 当前光伏
            load / (self.P_WT_rated + self.P_PV_rated), # [2] 当前负荷
            p_wt_hist_avg / self.P_WT_rated,           # [3] 历史风电均值
            p_pv_hist_avg / self.P_PV_rated,           # [4] 历史光伏均值
            self.SOC,                                   # [5] 电池SOC
            self.SOCH,                                  # [6] 氢罐SOCH
            (net_load_norm + 1) / 2.0,                 # [7] 净负荷（归一化到[0,1]）
            re_ratio,                                   # [8] 可再生能源占比
            hour / 24.0                                 # [9] 小时归一化
        ], dtype=np.float32)
        return obs

    # --- Physical Models ---
    def _calc_wind_power(self, v):
        """风电功率计算"""
        v_cut_in, v_rated, v_cut_out = 3.0, 12.0, 25.0
        if v < v_cut_in or v > v_cut_out:
            return 0.0
        elif v < v_rated:
            return self.P_WT_rated * ((v - v_cut_in) / (v_rated - v_cut_in)) ** 3
        else:
            return self.P_WT_rated

    def _calc_pv_power(self, G, T):
        """光伏功率计算"""
        k_t = -0.0037
        p_pv = self.P_PV_rated * (G / 1000.0) * (1 + k_t * (T - 25.0))
        return max(0.0, p_pv)

    def _calc_electrolyzer_h2(self, p_in):
        """电解槽氢气产量计算: ~55 kWh/kg"""
        if p_in <= 0: return 0.0
        return p_in / 55.0  # kg/h

    def _calc_fuel_cell_h2(self, p_out):
        """燃料电池氢气消耗计算: ~16 kWh/kg"""
        if p_out <= 0: return 0.0
        return p_out / 16.0  # kg/h

    # ========================================================================
    # V12: GEMINI'S APPROACH - AGENT-DRIVEN ALLOCATION
    # ========================================================================
    
    def _apply_proportional_scaling(self, p_avail, p_load_req, p_el_req, p_bat_charge_req):
        """
        V12核心：按比例缩放（Gemini建议）
        
        核心原则：
        1. Agent自由请求功率分配（负荷、制氢、储能）
        2. 环境只负责总功率约束
        3. 如果总需求超过可用功率，按比例缩放所有请求
        4. 负荷不再是硬约束，而是软约束（通过惩罚引导）
        
        Args:
            p_avail: 可用总功率 (kW, >=0)
            p_load_req: 负荷需求 (kW, >=0)
            p_el_req: 电解槽请求功率 (kW, >=0)
            p_bat_charge_req: 电池充电请求功率 (kW, >=0)
        
        Returns:
            (p_load_actual, p_el_actual, p_bat_actual): 实际分配的功率 (kW)
        """
        # 计算总需求
        p_total_demand = p_load_req + p_el_req + p_bat_charge_req
        
        if p_total_demand <= 0:
            return 0.0, 0.0, 0.0
        
        if p_total_demand <= p_avail:
            # 功率充足，完全满足所有请求
            return p_load_req, p_el_req, p_bat_charge_req
        else:
            # 功率不足，按比例缩放所有请求（包括负荷！）
            scale = p_avail / p_total_demand
            return (p_load_req * scale, 
                    p_el_req * scale, 
                    p_bat_charge_req * scale)

    def _apply_electrolyzer_constraints(self, p_el_alloc):
        """
        应用电解槽物理约束
        
        Args:
            p_el_alloc: 分配的电解槽功率 (kW, >=0)
        
        Returns:
            p_el_safe: 最终的电解槽功率 (kW)
        """
        if p_el_alloc <= 0:
            return 0.0
        
        # 检查氢气储罐剩余空间
        h2_max_allowed = self.SOCH_max * self.M_HS_max
        h2_room = max(0.0, h2_max_allowed - self.H2_storage_kg)
        
        if h2_room <= 0:
            # 储罐已满
            return 0.0
        
        # 计算氢气产量限制
        max_h2_prod = h2_room / self.dt  # kg/h
        p_el_limit_by_h2 = max_h2_prod * 55.0  # kW
        
        # 综合所有约束
        p_el_safe = min(p_el_alloc, p_el_limit_by_h2, self.P_EL_rated)
        
        return p_el_safe

    def _apply_battery_constraints(self, p_bat_charge_alloc, p_bat_discharge_req):
        """
        应用电池物理约束
        
        Args:
            p_bat_charge_alloc: 分配的充电功率 (kW, >=0)
            p_bat_discharge_req: 请求的放电功率 (kW, >=0)
        
        Returns:
            p_bat_safe: 最终的电池功率 (kW, 正=充电, 负=放电)
        """
        # 充电约束
        if p_bat_charge_alloc > 0:
            energy_room = (self.SOC_max - self.SOC) * self.E_bat_rated
            limit_by_capacity = energy_room / (self.dt * self.eta_bat_ch)
            p_bat_charge_safe = min(p_bat_charge_alloc, limit_by_capacity)
            return max(0.0, p_bat_charge_safe)
        
        # 放电约束
        elif p_bat_discharge_req > 0:
            energy_avail = (self.SOC - self.SOC_min) * self.E_bat_rated
            limit_by_capacity = energy_avail / self.dt
            p_bat_discharge_safe = min(p_bat_discharge_req, limit_by_capacity)
            return -max(0.0, p_bat_discharge_safe)
        
        else:
            return 0.0

    def _apply_fuel_cell_constraints(self, p_fc_req):
        """
        应用燃料电池约束
        
        Args:
            p_fc_req: 请求的燃料电池功率 (kW, >=0)
        
        Returns:
            p_fc_safe: 最终的燃料电池功率 (kW)
            fc_startup_cost: 启动成本（用于奖励计算）
        """
        fc_startup_cost = 0.0
        
        # 判断是否请求启动/运行
        agent_wants_fc = (p_fc_req >= self.P_fc_min)
        
        if agent_wants_fc:
            if not self.fc_is_running:
                # 开始启动
                self.fc_is_running = True
                self.fc_startup_progress = 0.0
                fc_startup_cost = -50.0  # 启动成本
            
            # 更新启动进度（线性爬坡）
            if self.fc_startup_progress < 1.0:
                self.fc_startup_progress += self.dt / self.FC_STARTUP_TIME
                self.fc_startup_progress = min(1.0, self.fc_startup_progress)
            
            # 计算可用功率（受启动进度限制）
            p_fc_max = self.P_FC_rated * self.fc_startup_progress
            
            # 氢气约束
            h2_available = self.H2_storage_kg - self.SOCH_min * self.M_HS_max
            max_h2_flow = max(0.0, h2_available) / self.dt
            p_fc_limit_by_h2 = max_h2_flow * 16.0
            
            # 最终功率
            p_fc_safe = min(p_fc_req, p_fc_max, p_fc_limit_by_h2)
            
            # 最小功率约束
            if p_fc_safe < self.P_fc_min:
                p_fc_safe = 0.0
        else:
            # 关闭
            self.fc_is_running = False
            self.fc_startup_progress = 0.0
            p_fc_safe = 0.0
        
        return p_fc_safe, fc_startup_cost

    def _update_soc(self, p_bat_safe):
        """更新电池SOC"""
        if p_bat_safe >= 0:  # 充电
            delta_soc = (p_bat_safe * self.dt * self.eta_bat_ch) / self.E_bat_rated
        else:  # 放电
            delta_soc = (p_bat_safe * self.dt) / (self.E_bat_rated * self.eta_bat_dis)
        
        self.SOC += delta_soc
        self.SOC = np.clip(self.SOC, self.SOC_min, self.SOC_max)

    def _update_soch(self, p_el_safe, p_fc_safe):
        """更新氢气储罐SOCH"""
        h2_prod = self._calc_electrolyzer_h2(p_el_safe) * self.dt  # kg
        h2_cons = self._calc_fuel_cell_h2(p_fc_safe) * self.dt  # kg
        
        self.H2_storage_kg += (h2_prod - h2_cons)
        
        # 严格限制在[SOCH_min, SOCH_max]范围内
        h2_vented = 0.0
        h2_max_allowed = self.SOCH_max * self.M_HS_max
        h2_min_allowed = self.SOCH_min * self.M_HS_max
        
        if self.H2_storage_kg > h2_max_allowed:
            h2_vented = self.H2_storage_kg - h2_max_allowed
            self.H2_storage_kg = h2_max_allowed
        elif self.H2_storage_kg < h2_min_allowed:
            self.H2_storage_kg = h2_min_allowed
        
        self.SOCH = self.H2_storage_kg / self.M_HS_max
        
        return h2_prod, h2_vented

    def _calculate_reward(self, p_load, p_load_actual, p_el_safe, p_bat_safe, 
                         p_fc_safe, p_dump, h2_vented, fc_startup_cost):
        """
        V12: 新的奖励函数（软约束设计）
        
        设计理念：
        1. 制氢是主要收益（100分/小时）
        2. 负荷满足也有奖励（50分/小时）
        3. 缺电有惩罚（20分/小时），但不压倒制氢奖励
        4. Agent会学习平衡：什么时候优先制氢，什么时候优先满足负荷
        
        关键改进：
        - 负荷不再是硬约束，而是软约束（通过惩罚引导）
        - 制氢奖励足够大，鼓励Agent积极制氢
        - 缺电惩罚适中，不会完全压倒制氢奖励
        """
        # 1. 制氢奖励（主要收益，0-100分）
        r_h2_production = (p_el_safe / self.P_EL_rated) * 100.0
        
        # 2. 负荷满足奖励（引导作用，0-50分）
        load_satisfaction_ratio = p_load_actual / max(p_load, 1e-6)
        r_load_satisfaction = load_satisfaction_ratio * 50.0
        
        # 3. 缺电惩罚（软约束，0-20分）
        p_unmet = p_load - p_load_actual
        penalty_unmet = (p_unmet / max(p_load, 1e-6)) * 20.0
        
        # 4. 其他小惩罚（引导作用）
        penalty_dump = (p_dump / (self.P_WT_rated + self.P_PV_rated)) * 5.0  # 弃电惩罚
        penalty_vented = h2_vented * 5.0  # 排氢惩罚
        penalty_fc_use = (p_fc_safe / self.P_FC_rated) * 2.0  # 燃料电池使用惩罚
        
        # 总奖励
        reward = (r_h2_production + r_load_satisfaction
                  - penalty_unmet - penalty_dump - penalty_vented - penalty_fc_use
                  + fc_startup_cost)
        
        return reward

    # ========================================================================
    # MAIN STEP FUNCTION
    # ========================================================================
    
    def step(self, action):
        """
        V12: Gemini方法的step函数
        
        核心改进：
        1. ✅ 移除"负荷优先"硬约束
        2. ✅ Agent自由分配功率（负荷、制氢、储能）
        3. ✅ 环境只负责总功率约束（按比例缩放）
        4. ✅ 负荷通过惩罚引导（软约束）
        
        流程：
        1. 获取当前状态（发电、负荷）
        2. 解析Agent动作（制氢、储能、燃料电池）
        3. 计算可用总功率（发电 + 燃料电池 + 电池放电）
        4. 按比例缩放所有请求（如果总需求超过可用）
        5. 应用物理约束（电解槽、电池、燃料电池）
        6. 状态更新（SOC、SOCH）
        7. 计算奖励（制氢奖励 + 负荷满足奖励 - 惩罚）
        8. 返回结果
        """
        # ========== 1. 获取当前状态 ==========
        p_gen = self.current_p_wt + self.current_p_pv
        p_load = self.current_load
        
        # ========== 2. 解析Agent动作 ==========
        raw_bat_action = action[0] * self.E_bat_rated  # [-E_bat_rated, +E_bat_rated]
        raw_el_action = (action[1] + 1) / 2 * self.P_EL_rated  # [0, P_EL_rated]
        raw_fc_action = (action[2] + 1) / 2 * self.P_FC_rated  # [0, P_FC_rated]
        
        # 分离充电和放电请求
        p_bat_charge_req = max(0, raw_bat_action)
        p_bat_discharge_req = max(0, -raw_bat_action)
        
        # ========== 3. 计算可用总功率 ==========
        # 先应用燃料电池约束（获取实际可用功率）
        p_fc_safe, fc_startup_cost = self._apply_fuel_cell_constraints(raw_fc_action)
        
        # 先应用电池放电约束（获取实际可用功率）
        p_bat_discharge_safe = self._apply_battery_constraints(0.0, p_bat_discharge_req)
        p_bat_discharge_actual = abs(p_bat_discharge_safe)  # 转为正值
        
        # 可用总功率 = 发电 + 燃料电池 + 电池放电
        p_avail = p_gen + p_fc_safe + p_bat_discharge_actual
        
        # ========== 4. V12核心：按比例缩放所有请求 ==========
        p_load_actual, p_el_alloc, p_bat_charge_alloc = self._apply_proportional_scaling(
            p_avail, p_load, raw_el_action, p_bat_charge_req
        )
        
        # ========== 5. 应用物理约束 ==========
        p_el_safe = self._apply_electrolyzer_constraints(p_el_alloc)
        
        # 电池充电约束
        p_bat_charge_safe = self._apply_battery_constraints(p_bat_charge_alloc, 0.0)
        
        # 最终电池功率（充电或放电）
        if p_bat_charge_alloc > 0:
            p_bat_safe = p_bat_charge_safe  # 充电
        else:
            p_bat_safe = p_bat_discharge_safe  # 放电（负值）
        
        # ========== 6. 状态更新 ==========
        self._update_soc(p_bat_safe)
        h2_prod, h2_vented = self._update_soch(p_el_safe, p_fc_safe)
        
        # ========== 7. 计算弃电 ==========
        # 供应侧：发电 + 燃料电池 + 电池放电
        p_supply_total = p_gen + p_fc_safe
        if p_bat_safe < 0:  # 放电
            p_supply_total += abs(p_bat_safe)
        
        # 需求侧：负荷实际 + 电解槽 + 电池充电
        p_demand_total = p_load_actual + p_el_safe
        if p_bat_safe > 0:  # 充电
            p_demand_total += p_bat_safe
        
        # 弃电
        p_dump = max(0.0, p_supply_total - p_demand_total)
        
        # ========== 8. 计算奖励 ==========
        reward = self._calculate_reward(
            p_load, p_load_actual, p_el_safe, p_bat_safe, p_fc_safe,
            p_dump, h2_vented, fc_startup_cost
        )
        
        # ========== 9. 返回结果 ==========
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        next_obs = self._get_obs() if not done else np.zeros(10)
        
        info = {
            'h2_prod': h2_prod,
            'h2_vented': h2_vented,
            'soc': self.SOC,
            'soch': self.SOCH,
            'p_dump': p_dump,
            'p_unmet': p_load - p_load_actual,
            'real_p_el': p_el_safe,
            'real_p_bat': p_bat_safe,
            'real_p_fc': p_fc_safe,
            'p_load_actual': p_load_actual
        }
        
        return next_obs, reward, done, info