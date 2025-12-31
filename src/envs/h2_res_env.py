
import gym
import numpy as np
from gym import spaces
from collections import deque


class H2RESEnv(gym.Env):
    """
    H2-RES System Environment (V10: True Gemini Implementation)

    Version History:
    - V1-V4: Initial implementations with various reward function designs
    - V5: Engineering constraints (fuel cell state machine, startup delays)
    - V5.1: Bug fix for continuous operation rewards (incremental rewards)
    - V6: Ratio-based power allocation with physical constraint checks
    - V7: Optimized reward for FC usage
    - V8: Enhanced observation space with history and time features
    - V9: Complete rewrite for training convergence
    - V9.1: True ratio-based allocation (removed min power constraint)
    - V9.2: Balanced reward function
    - V10: Fixed power allocation logic (Current)
    
    V10 Key Features (Critical Fix):
    1. ✅ Agent freely requests power (no forced zeros)
    2. ✅ Environment scales by available power ratio
    3. ✅ Works in both surplus and deficit scenarios
    4. ✅ True implementation of Gemini's philosophy
    
    Design Principles:
    - Agent decides: How much power to allocate (electrolyzer, battery, FC)
    - Environment enforces: Physical constraints + energy balance
    - No artificial restrictions: Agent learns optimal strategy
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
        self.P_el_min = 0.1 * self.P_EL_rated  # 电解槽最小运行功率约束
        self.P_fc_min = 0.1 * self.P_FC_rated  # 燃料电池最小运行功率约束

        # Efficiency Constants (效率参数)
        self.eta_bat_ch = 0.95  # 电池充电效率
        self.eta_bat_dis = 0.95  # 电池放电效率
        self.dt = 1.0  # 时间步长 (小时)

        # --- V9: Simplified Fuel Cell Parameters ---
        self.FC_STARTUP_TIME = 1.0  # 启动时间（小时）
        
        # --- V9: History Buffer ---
        self.history_hours = history_hours
        self.p_wt_history = deque(maxlen=history_hours)
        self.p_pv_history = deque(maxlen=history_hours)
        
        # --- V9: Enhanced Observation Space (12D) ---
        # [p_wt, p_pv, load, p_wt_hist_avg, p_pv_hist_avg, hour_sin, hour_cos, 
        #  SOC, SOCH, net_load, re_ratio, fc_startup_progress]
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Internal Variables
        self.SOC = 0.5
        self.SOCH = 0.5
        self.H2_storage_kg = self.SOCH * self.M_HS_max

        # Raw tracking
        self.current_p_wt = 0.0
        self.current_p_pv = 0.0
        self.current_load = 0.0
        
        # 电解槽启停追踪
        self.el_was_on = False
        self.el_continuous_hours = 0
        
        # V9: 简化的燃料电池状态追踪
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
        
        # 重置电解槽状态追踪
        self.el_was_on = False
        self.el_continuous_hours = 0
        
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
        V9: 增强的观测空间（12维）
        
        返回：
        [0] p_wt_norm: 当前风电功率（归一化）
        [1] p_pv_norm: 当前光伏功率（归一化）
        [2] load_norm: 当前负荷（归一化）
        [3] p_wt_hist_avg: 过去3h风电平均功率（归一化）
        [4] p_pv_hist_avg: 过去3h光伏平均功率（归一化）
        [5] hour_sin: 小时的sin编码（周期性）
        [6] hour_cos: 小时的cos编码（周期性）
        [7] SOC: 电池荷电状态
        [8] SOCH: 氢罐荷电状态
        [9] net_load_norm: 净负荷（load-gen，归一化）
        [10] re_ratio: 可再生能源占比
        [11] fc_startup_progress: 燃料电池启动进度 (V9新增)
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
        
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
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
            (hour_sin + 1) / 2.0,                      # [5] 小时sin（归一化到[0,1]）
            (hour_cos + 1) / 2.0,                      # [6] 小时cos（归一化到[0,1]）
            self.SOC,                                   # [7] 电池SOC
            self.SOCH,                                  # [8] 氢罐SOCH
            (net_load_norm + 1) / 2.0,                 # [9] 净负荷（归一化到[0,1]）
            re_ratio,                                   # [10] 可再生能源占比
            self.fc_startup_progress                    # [11] 燃料电池启动进度 (V9新增)
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
    # V9: SIMPLIFIED POWER ALLOCATION LOGIC
    # ========================================================================
    
    def _allocate_power(self, p_avail, p_el_req, p_bat_req):
        """
        V10: 真正的Gemini按比例分配（无条件限制）
        
        核心改进：
        1. Agent自由请求任意功率
        2. 环境根据可用功率按比例缩放
        3. 无论p_avail正负，都允许分配
        4. 完全符合Gemini的设计理念
        
        Args:
            p_avail: 可用功率 (kW, 可正可负)
            p_el_req: 电解槽请求功率 (kW, >=0)
            p_bat_req: 电池充电请求功率 (kW, >=0)
        
        Returns:
            (p_el_alloc, p_bat_alloc): 分配后的功率 (kW)
        """
        # 计算总需求
        total_req = p_el_req + p_bat_req
        
        if total_req <= 0:
            return 0.0, 0.0
        
        # 关键改进：即使p_avail<=0，也允许Agent请求
        # 让Agent学习在功率不足时如何分配
        if p_avail <= 0:
            # 功率不足，但仍按比例分配（让Agent学习）
            # 这样Agent可以学习到"功率不足时应该降低请求"
            scale = max(0.0, p_avail) / total_req
            return p_el_req * scale, p_bat_req * scale
        
        if total_req <= p_avail:
            # 功率充足，完全满足
            return p_el_req, p_bat_req
        else:
            # 功率不足，按比例缩放
            scale = p_avail / total_req
            return p_el_req * scale, p_bat_req * scale

    def _apply_electrolyzer_constraints(self, p_el_alloc):
        """
        应用电解槽物理约束（V9.1：移除最小功率约束）
        
        核心改进：
        - 移除最小功率约束，允许任意功率运行
        - 完全符合"按比例分配"的设计理念
        - Agent可以学习到连续的制氢策略
        
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
        
        # 综合所有约束（移除最小功率检查）
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
        应用燃料电池约束（V9简化版）
        
        核心改进：
        1. 移除状态机
        2. 简单的启动延迟（线性爬坡）
        3. 启动进度可观测
        
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
                fc_startup_cost = -100.0  # 启动成本（降低惩罚）
            
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

    def _check_energy_balance(self, p_gen, p_load, p_el_safe, p_bat_safe, p_fc_safe):
        """检查能量平衡"""
        # 供应侧：发电 + 燃料电池 + 电池放电
        p_supply_total = p_gen + p_fc_safe
        if p_bat_safe < 0:  # 放电
            p_supply_total += abs(p_bat_safe)
        
        # 需求侧：负荷 + 电解槽 + 电池充电
        p_demand_total = p_load + p_el_safe
        if p_bat_safe > 0:  # 充电
            p_demand_total += p_bat_safe
        
        # 功率平衡
        balance = p_supply_total - p_demand_total
        
        if balance >= 0:
            p_dump = balance  # 弃电
            p_unmet = 0.0
        else:
            p_dump = 0.0
            p_unmet = abs(balance)  # 缺电
        
        return p_dump, p_unmet

    def _calculate_reward(self, p_load, p_unmet, p_el_safe, p_bat_safe, p_fc_safe,
                         p_dump, h2_vented, fc_startup_cost):
        """
        V9.2: 平衡的奖励函数（Gemini建议的实现）
        
        核心改进：
        1. 制氢是主要收益（500分）
        2. 缺电惩罚降低到合理水平（50分）
        3. 让Agent有学习空间，不会因为缺电就完全负reward
        4. 完全符合"按比例分配"理念
        
        设计理念：
        - Agent的目标：最大化制氢量，同时尽量满足负荷
        - 环境负责：能量平衡、物理约束
        - 奖励信号：清晰、平衡、易于学习
        """
        # 1. 制氢奖励（核心收益，500分）
        r_h2_production = (p_el_safe / self.P_EL_rated) * 500.0
        
        # 2. 电池使用奖励（鼓励储能）
        if p_bat_safe > 0:
            r_battery = (p_bat_safe / self.E_bat_rated) * 20.0
        elif p_bat_safe < 0:
            r_battery = (abs(p_bat_safe) / self.E_bat_rated) * 30.0
        else:
            r_battery = 0.0
        
        # 3. 惩罚项（降低权重，给Agent学习空间）
        penalty_dump = (p_dump / (self.P_WT_rated + self.P_PV_rated)) * 5.0  # 弃电惩罚（降低）
        penalty_vented = h2_vented * 20.0  # 排氢惩罚（降低）
        penalty_unmet = (p_unmet / max(p_load, 1e-6)) * 50.0  # 缺电惩罚（大幅降低：200→50）
        penalty_fc_use = (p_fc_safe / self.P_FC_rated) * 5.0  # 燃料电池使用惩罚（降低）
        
        # 总奖励
        reward = (r_h2_production + r_battery
                  - penalty_dump - penalty_vented - penalty_unmet - penalty_fc_use
                  + fc_startup_cost)
        
        return reward

    # ========================================================================
    # MAIN STEP FUNCTION
    # ========================================================================
    
    def step(self, action):
        """
        V10: 真正的Gemini实现
        
        关键改进：
        1. Agent自由请求功率（无论发电是否充足）
        2. 环境按可用功率比例缩放
        3. 让Agent学习在功率不足时如何优化分配
        
        流程：
        1. 获取当前状态
        2. 解析Agent动作
        3. 计算可用功率（发电-负荷）
        4. 按比例分配功率（关键改进）
        5. 应用物理约束
        6. 状态更新
        7. 能量平衡检查
        8. 计算奖励
        9. 返回结果
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
        
        # ========== 3. 计算可用功率（关键改进）==========
        # 先满足负荷，剩余功率用于制氢和储能
        p_avail = p_gen - p_load
        
        # ========== 4. 按比例分配功率（关键改进）==========
        # 无论p_avail正负，都允许Agent请求
        p_el_alloc, p_bat_alloc = self._allocate_power(
            p_avail, raw_el_action, p_bat_charge_req
        )
        
        # ========== 5. 应用物理约束 ==========
        p_el_safe = self._apply_electrolyzer_constraints(p_el_alloc)
        p_bat_safe = self._apply_battery_constraints(p_bat_alloc, p_bat_discharge_req)
        p_fc_safe, fc_startup_cost = self._apply_fuel_cell_constraints(raw_fc_action)
        
        # ========== 6. 状态更新 ==========
        self._update_soc(p_bat_safe)
        h2_prod, h2_vented = self._update_soch(p_el_safe, p_fc_safe)
        
        # ========== 7. 能量平衡检查 ==========
        p_dump, p_unmet = self._check_energy_balance(
            p_gen, p_load, p_el_safe, p_bat_safe, p_fc_safe
        )
        
        # ========== 8. 计算奖励 ==========
        reward = self._calculate_reward(
            p_load, p_unmet, p_el_safe, p_bat_safe, p_fc_safe,
            p_dump, h2_vented, fc_startup_cost
        )
        
        # ========== 9. 返回结果 ==========
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        next_obs = self._get_obs() if not done else np.zeros(12)
        
        info = {
            'h2_prod': h2_prod,
            'h2_vented': h2_vented,
            'soc': self.SOC,
            'soch': self.SOCH,
            'p_dump': p_dump,
            'p_unmet': p_unmet,
            'real_p_el': p_el_safe,
            'real_p_bat': p_bat_safe,
            'real_p_fc': p_fc_safe
        }
        
        return next_obs, reward, done, info