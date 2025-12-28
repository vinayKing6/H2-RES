import gym
import numpy as np
from gym import spaces


class H2RESEnv(gym.Env):
    """
    H2-RES System Environment

    Strictly implements Physical Feasibility Constraints (Hard Constraints)
    to ensure Energy Conservation and Thermodynamic Consistency.
    """

    def __init__(self, df_weather, df_load):
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

        # State & Action Space
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Internal Variables
        self.SOC = 0.5
        self.SOCH = 0.5
        self.H2_storage_kg = self.SOCH * self.M_HS_max

        # Raw tracking
        self.current_p_wt = 0.0
        self.current_p_pv = 0.0
        self.current_load = 0.0
        
        # 电解槽启停追踪（防止频繁启停）
        self.el_was_on = False  # 上一步电解槽是否运行
        self.el_continuous_hours = 0  # 连续运行小时数

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
        
        return self._get_obs()

    def _get_obs(self):
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

        obs = np.array([
            p_wt / self.P_WT_rated,
            p_pv / self.P_PV_rated,
            load / (self.P_WT_rated + self.P_PV_rated),
            self.SOC,
            self.SOCH
        ], dtype=np.float32)
        return obs

    # --- Physical Models ---
    def _calc_wind_power(self, v):
        v_cut_in, v_rated, v_cut_out = 3.0, 12.0, 25.0
        if v < v_cut_in or v > v_cut_out:
            return 0.0
        elif v < v_rated:
            return self.P_WT_rated * ((v - v_cut_in) / (v_rated - v_cut_in)) ** 3
        else:
            return self.P_WT_rated

    def _calc_pv_power(self, G, T):
        k_t = -0.0037
        p_pv = self.P_PV_rated * (G / 1000.0) * (1 + k_t * (T - 25.0))
        return max(0.0, p_pv)

    def _calc_electrolyzer_h2(self, p_in):
        """电解槽氢气产量计算: ~55 kWh/kg (输入功率 -> 氢气质量)"""
        if p_in <= 0: return 0.0
        return p_in / 55.0  # kg/h

    def _calc_fuel_cell_h2(self, p_out):
        """燃料电池氢气消耗计算: ~16 kWh/kg (输出功率 -> 氢气质量)"""
        if p_out <= 0: return 0.0
        return p_out / 16.0  # kg/h

    # --- INTELLIGENT POWER ALLOCATION ---
    def _allocate_power_intelligently(self, p_surplus, raw_el, raw_bat_charge, current_soc):
        """
        智能功率分配算法：在满足物理约束的前提下，尊重Agent意图并考虑SOC状态
        
        核心原则：
        1. 电解槽有最小功率约束（300kW），电池没有
        2. 当SOC接近下限时，优先给电池充电（避免死锁）
        3. 否则按Agent请求比例分配
        
        Args:
            p_surplus: 剩余可用功率 (kW)
            raw_el: Agent请求的电解槽功率 (kW)
            raw_bat_charge: Agent请求的电池充电功率 (kW, 仅正值)
            current_soc: 当前电池SOC
        
        Returns:
            (p_el_allocated, p_bat_allocated): 分配后的功率 (kW)
        """
        # 情况1：电解槽请求不足最小功率 → 全部给电池
        if raw_el < self.P_el_min:
            return 0.0, min(raw_bat_charge, p_surplus)
        
        # 情况2：剩余功率不足以启动电解槽 → 全部给电池
        if p_surplus < self.P_el_min:
            return 0.0, min(raw_bat_charge, p_surplus)
        
        # 情况3：SOC危险低（<0.35）且电池请求充电 → 优先电池（避免死锁）
        if current_soc < 0.35 and raw_bat_charge > 0:
            # 给电池至少50%的剩余功率，或满足其请求
            bat_priority = min(raw_bat_charge, p_surplus * 0.5)
            remaining = p_surplus - bat_priority
            
            if remaining >= self.P_el_min:
                # 剩余功率足够启动电解槽
                el_allocated = min(raw_el, remaining)
                return el_allocated, bat_priority
            else:
                # 剩余功率不足，全部给电池
                return 0.0, min(raw_bat_charge, p_surplus)
        
        # 情况4：正常情况 → 按Agent请求比例分配
        total_request = raw_el + raw_bat_charge
        
        if total_request <= p_surplus:
            # 功率充足，完全满足Agent请求
            return raw_el, raw_bat_charge
        else:
            # 功率不足，按比例分配
            ratio_el = raw_el / total_request
            p_el_tentative = p_surplus * ratio_el
            
            if p_el_tentative >= self.P_el_min:
                # 按比例分配后仍满足最小功率
                p_bat_tentative = p_surplus - p_el_tentative
                return p_el_tentative, p_bat_tentative
            else:
                # 按比例分配后不满足最小功率
                # 检查是否可以给电解槽最小功率
                if p_surplus >= self.P_el_min:
                    p_el_allocated = self.P_el_min
                    p_bat_allocated = p_surplus - self.P_el_min
                    return p_el_allocated, p_bat_allocated
                else:
                    # 剩余功率不足，全部给电池
                    return 0.0, min(raw_bat_charge, p_surplus)

    # --- MAIN STEP FUNCTION WITH STRICT CONSTRAINTS ---
    def step(self, action):
        # 1. Physics Initialization
        p_gen = self.current_p_wt + self.current_p_pv
        p_load = self.current_load

        # Initial Power Balance
        p_surplus_initial = p_gen - p_load

        # Raw Actions (Intent)
        raw_bat_action = action[0] * self.E_bat_rated
        raw_el_action = (action[1] + 1) / 2 * self.P_EL_rated
        raw_fc_action = (action[2] + 1) / 2 * self.P_FC_rated

        # =================================================================
        # 2. Intelligent Power Allocation (New Layer)
        # =================================================================
        
        # 修复：正确处理充电和放电场景
        if p_surplus_initial > 0 and raw_bat_action > 0:
            # 充电场景：智能分配剩余功率（传入当前SOC）
            p_el_allocated, p_bat_allocated = self._allocate_power_intelligently(
                p_surplus_initial, raw_el_action, raw_bat_action, self.SOC
            )
        elif raw_bat_action < 0:
            # 放电场景：电池放电不受智能分配影响，直接使用原始请求
            p_el_allocated = 0.0  # 放电时不启动电解槽
            p_bat_allocated = raw_bat_action  # 保持负值（放电）
        else:
            # 其他情况（无剩余功率或电池不动作）
            p_el_allocated = raw_el_action if p_surplus_initial > 0 else 0.0
            p_bat_allocated = 0.0

        # =================================================================
        # 3. Physical Consistency Constraints (Safety Layer)
        # =================================================================

        # --- A. Electrolyzer Constraint ---
        p_el_safe = 0.0

        if p_el_allocated >= self.P_el_min and p_surplus_initial >= self.P_el_min:
            # 检查氢气储罐剩余空间
            h2_max_allowed = self.SOCH_max * self.M_HS_max
            h2_room = max(0.0, h2_max_allowed - self.H2_storage_kg)
            
            if h2_room > 0:  # 储罐有空间
                max_h2_prod = h2_room / self.dt  # kg/h
                p_el_limit_by_h2 = max_h2_prod * 55.0  # kW
                
                # 综合所有约束
                p_el_safe = min(p_el_allocated, p_el_limit_by_h2, self.P_EL_rated)
                
                # 再次检查最小功率
                if p_el_safe < self.P_el_min:
                    p_el_safe = 0.0
            else:
                p_el_safe = 0.0  # 储罐已满
        else:
            p_el_safe = 0.0

        # 计算电解槽实际消耗后的剩余功率
        p_surplus_after_el = p_surplus_initial - p_el_safe

        # --- B. Battery Constraint ---
        p_bat_safe = 0.0

        if p_bat_allocated > 0:  # 充电（修复：使用p_bat_allocated而非raw_bat_action）
            if p_surplus_after_el > 0:
                # 修复：使用分配后的功率作为基准，但仍需检查实际剩余
                limit_by_power = min(p_bat_allocated, p_surplus_after_el)

                # 容量限制（考虑充电效率）
                energy_room = (self.SOC_max - self.SOC) * self.E_bat_rated
                limit_by_capacity = energy_room / (self.dt * self.eta_bat_ch)

                p_bat_safe = min(limit_by_power, limit_by_capacity)
                p_bat_safe = max(0.0, p_bat_safe)
            else:
                p_bat_safe = 0.0

        elif p_bat_allocated < 0:  # 放电（修复：使用p_bat_allocated）
            req_discharge = abs(p_bat_allocated)

            # 可用能量直接转换为功率限制
            energy_avail = (self.SOC - self.SOC_min) * self.E_bat_rated
            limit_by_capacity = energy_avail / self.dt

            actual_discharge = min(req_discharge, limit_by_capacity)
            p_bat_safe = -max(0.0, actual_discharge)

        # --- C. Fuel Cell Constraint ---
        # 修复: 添加最小运行功率约束
        h2_available = self.H2_storage_kg - self.SOCH_min * self.M_HS_max
        max_h2_flow = max(0.0, h2_available) / self.dt  # kg/h
        p_fc_limit_by_h2 = max_h2_flow * 16.0  # kW

        p_fc_safe = min(raw_fc_action, p_fc_limit_by_h2, self.P_FC_rated)
        p_fc_safe = max(0.0, p_fc_safe)
        
        # 最小功率约束：低于最小功率则关闭
        if p_fc_safe < self.P_fc_min:
            p_fc_safe = 0.0

        # =================================================================
        # 3. State Update
        # =================================================================

        # SOC Update (修复: 正确应用充放电效率)
        if p_bat_safe >= 0:  # 充电
            # 输入功率 -> 存储能量（乘以效率）
            delta_soc = (p_bat_safe * self.dt * self.eta_bat_ch) / self.E_bat_rated
        else:  # 放电
            # 修复: 输出功率 -> 消耗能量（除以效率）
            # p_bat_safe是负值，所以delta_soc也是负值
            delta_soc = (p_bat_safe * self.dt) / (self.E_bat_rated * self.eta_bat_dis)

        self.SOC += delta_soc
        self.SOC = np.clip(self.SOC, self.SOC_min, self.SOC_max)

        # H2 Update (修复溢出处理逻辑)
        h2_prod = self._calc_electrolyzer_h2(p_el_safe) * self.dt  # kg
        h2_cons = self._calc_fuel_cell_h2(p_fc_safe) * self.dt  # kg

        self.H2_storage_kg += (h2_prod - h2_cons)

        # 修复: 严格限制在[SOCH_min, SOCH_max]范围内
        h2_vented = 0.0
        h2_max_allowed = self.SOCH_max * self.M_HS_max
        h2_min_allowed = self.SOCH_min * self.M_HS_max
        
        if self.H2_storage_kg > h2_max_allowed:
            h2_vented = self.H2_storage_kg - h2_max_allowed
            self.H2_storage_kg = h2_max_allowed
        elif self.H2_storage_kg < h2_min_allowed:
            self.H2_storage_kg = h2_min_allowed

        self.SOCH = self.H2_storage_kg / self.M_HS_max

        # =================================================================
        # 4. Final Balance Check (修复: 正确的能量守恒检查)
        # =================================================================
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
            p_unmet = abs(balance)  # 缺电（理论上不应出现）

        # =================================================================
        # 5. Reward (工程优化版：防止频繁启停 + 提高制氢积极性)
        # =================================================================
        
        # 核心设计原则：
        # 1. 所有奖励项归一化到相似量级（避免某项过大）
        # 2. 电解槽启停惩罚要足够大，让Agent不敢频繁启停
        # 3. 连续运行奖励要足够吸引人
        
        # 1. 负荷满足（最高优先级，归一化到0-100）
        load_satisfaction = (p_load - p_unmet) / max(p_load, 1e-6)
        r_load = load_satisfaction * 100.0
        
        # 2. 能量有效利用（归一化：按额定功率百分比计算）
        # 电池充电奖励：最高1000kW → 10分
        r_battery_storage = (max(0, p_bat_safe) / self.E_bat_rated) * 10.0
        # 制氢奖励：最高3000kW → 30分（3倍于电池）
        r_h2_production = (p_el_safe / self.P_EL_rated) * 30.0
        r_storage = r_battery_storage + r_h2_production
        
        # 3. 电解槽连续运行奖励（防止频繁启停）
        el_is_on = (p_el_safe >= self.P_el_min)
        
        # 初始化惩罚变量
        penalty_startup = 0.0
        penalty_shutdown = 0.0
        
        if el_is_on:
            self.el_continuous_hours += 1
            # 连续运行奖励：每小时+2分，最高40分（20小时）
            r_el_continuity = min(self.el_continuous_hours * 2.0, 40.0)
            
            # 如果刚启动（从关闭到开启），给予启动惩罚
            if not self.el_was_on:
                penalty_startup = -20.0  # 启动惩罚（相当于损失10小时连续运行奖励）
        else:
            self.el_continuous_hours = 0
            r_el_continuity = 0.0
            
            # 如果刚关闭（从开启到关闭），给予关闭惩罚
            if self.el_was_on:
                penalty_shutdown = -15.0  # 关闭惩罚（比启动轻一些）
        
        # 更新电解槽状态
        self.el_was_on = el_is_on
        
        # 4. SOC健康状态（归一化到±20）
        if 0.4 <= self.SOC <= 0.8:
            r_soc = 10.0  # 健康范围奖励
        elif self.SOC < 0.4:
            r_soc = -50.0 * (0.4 - self.SOC)  # 低SOC严重惩罚
        elif self.SOC > 0.85:
            r_soc = -20.0 * (self.SOC - 0.85)  # 过高轻微惩罚
        else:
            r_soc = 0.0
        
        # 5. SOCH健康状态（归一化到±20）
        if 0.3 <= self.SOCH <= 0.8:
            r_soch = 10.0  # 健康范围奖励
        elif self.SOCH < 0.3:
            r_soch = -40.0 * (0.3 - self.SOCH)  # 低SOCH惩罚
        else:
            r_soch = 0.0
        
        # 6. 惩罚项（归一化：按额定功率百分比）
        # 弃电惩罚：按总发电能力百分比，最高-10分
        penalty_dump = (p_dump / (self.P_WT_rated + self.P_PV_rated)) * 10.0
        # 氢气排放惩罚
        penalty_vented = h2_vented * 50.0  # 排放氢气非常不好
        # 缺电惩罚：最严重
        penalty_unmet = (p_unmet / max(p_load, 1e-6)) * 200.0
        
        # 7. 边界惩罚
        penalty_soc_boundary = 50.0 if self.SOC <= self.SOC_min + 0.01 else 0.0
        penalty_soch_boundary = 30.0 if self.SOCH <= self.SOCH_min + 0.01 else 0.0
        
        # 8. 运行成本（极低权重，可忽略）
        cost_op = 0.0  # 简化：不考虑运行成本
        
        # 总奖励（预期范围：-200 到 +200）
        reward = (r_load + r_storage + r_el_continuity + r_soc + r_soch
                  - penalty_dump - penalty_vented - penalty_unmet
                  - penalty_startup - penalty_shutdown
                  - penalty_soc_boundary - penalty_soch_boundary
                  - cost_op)

        self.current_step += 1
        done = self.current_step >= self.max_steps - 1

        next_obs = self._get_obs() if not done else np.zeros(5)

        # --- RETURN REAL PHYSICAL VALUES IN INFO FOR PLOTTING ---
        info = {
            'h2_prod': h2_prod,
            'h2_vented': h2_vented,  # 改名: sold -> vented
            'soc': self.SOC,
            'soch': self.SOCH,
            'p_dump': p_dump,
            'p_unmet': p_unmet,
            'real_p_el': p_el_safe,
            'real_p_bat': p_bat_safe,
            'real_p_fc': p_fc_safe
        }

        return next_obs, reward, done, info