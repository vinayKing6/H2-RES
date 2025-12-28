# -*- coding: utf-8 -*-
"""
物理一致性验证脚本
测试修复后的H2-RES环境是否满足物理约束
"""

import numpy as np
import pandas as pd
import sys
import io

# 修复Windows GBK编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.envs.h2_res_env import H2RESEnv

def test_battery_efficiency():
    """测试电池充放电效率"""
    print("\n" + "="*60)
    print("测试1: 电池充放电效率")
    print("="*60)
    
    # 创建简单测试数据
    df_test = pd.DataFrame({
        'wind_speed': [10.0] * 10,
        'irradiance': [800.0] * 10,
        'temperature': [25.0] * 10,
        'load': [2000.0] * 10
    })
    
    env = H2RESEnv(df_test, df_test)
    env.reset()
    
    # 测试充电
    print("\n[充电测试]")
    initial_soc = env.SOC
    print(f"初始SOC: {initial_soc:.4f}")
    
    # 充电100kW
    action = np.array([0.1, -1.0, -1.0])  # 只充电，不运行电解槽和燃料电池
    _, _, _, info = env.step(action)
    
    final_soc = env.SOC
    delta_soc = final_soc - initial_soc
    energy_stored = delta_soc * env.E_bat_rated
    
    print(f"充电功率: {info['real_p_bat']:.2f} kW")
    print(f"充电时间: 1 小时")
    print(f"理论存储能量: {info['real_p_bat'] * 1 * 0.95:.2f} kWh")
    print(f"实际存储能量: {energy_stored:.2f} kWh")
    print(f"最终SOC: {final_soc:.4f}")
    
    # 验证
    expected = info['real_p_bat'] * 1 * 0.95
    if abs(energy_stored - expected) < 1e-3:
        print("[OK] 充电效率正确")
    else:
        print(f"[ERROR] 充电效率错误: 期望{expected:.2f}, 实际{energy_stored:.2f}")
    
    # 测试放电
    print("\n[放电测试]")
    env.reset()
    env.SOC = 0.7  # 设置较高SOC以便放电
    initial_soc = env.SOC
    print(f"初始SOC: {initial_soc:.4f}")
    
    # 放电100kW
    action = np.array([-0.1, -1.0, -1.0])  # 只放电
    _, _, _, info = env.step(action)
    
    final_soc = env.SOC
    delta_soc = final_soc - initial_soc
    energy_consumed = abs(delta_soc * env.E_bat_rated)
    
    print(f"放电功率: {abs(info['real_p_bat']):.2f} kW")
    print(f"放电时间: 1 小时")
    print(f"理论消耗能量: {abs(info['real_p_bat']) / 0.95:.2f} kWh")
    print(f"实际消耗能量: {energy_consumed:.2f} kWh")
    print(f"最终SOC: {final_soc:.4f}")
    
    # 验证
    expected = abs(info['real_p_bat']) / 0.95
    if abs(energy_consumed - expected) < 1e-3:
        print("[OK] 放电效率正确")
    else:
        print(f"[ERROR] 放电效率错误: 期望{expected:.2f}, 实际{energy_consumed:.2f}")
    
    # 测试往返效率
    print("\n[往返效率测试]")
    env.reset()
    env.SOC = 0.5
    initial_soc = env.SOC
    
    # 充电
    action = np.array([0.1, -1.0, -1.0])
    _, _, _, info1 = env.step(action)
    soc_after_charge = env.SOC
    
    # 放电
    action = np.array([-0.1, -1.0, -1.0])
    _, _, _, info2 = env.step(action)
    final_soc = env.SOC
    
    round_trip_efficiency = (initial_soc - final_soc) / (soc_after_charge - initial_soc)
    expected_efficiency = 0.95 * 0.95  # 0.9025
    
    print(f"充电后SOC: {soc_after_charge:.4f}")
    print(f"放电后SOC: {final_soc:.4f}")
    print(f"往返效率: {round_trip_efficiency:.4f}")
    print(f"理论效率: {expected_efficiency:.4f}")
    
    if abs(round_trip_efficiency - expected_efficiency) < 0.01:
        print("[OK] 往返效率正确")
    else:
        print(f"[ERROR] 往返效率错误")


def test_hydrogen_system():
    """测试氢能系统约束"""
    print("\n" + "="*60)
    print("测试2: 氢能系统约束")
    print("="*60)
    
    df_test = pd.DataFrame({
        'wind_speed': [15.0] * 10,
        'irradiance': [1000.0] * 10,
        'temperature': [25.0] * 10,
        'load': [1000.0] * 10
    })
    
    env = H2RESEnv(df_test, df_test)
    env.reset()
    
    # 测试电解槽
    print("\n[电解槽测试]")
    action = np.array([0.0, 1.0, -1.0])  # 最大功率运行电解槽
    _, _, _, info = env.step(action)
    
    p_el = info['real_p_el']
    h2_prod = info['h2_prod']
    expected_h2 = p_el / 55.0
    
    print(f"电解槽功率: {p_el:.2f} kW")
    print(f"产氢量: {h2_prod:.4f} kg")
    print(f"理论产氢: {expected_h2:.4f} kg")
    
    if abs(h2_prod - expected_h2) < 1e-6:
        print("[OK] 电解槽产氢计算正确")
    else:
        print(f"[ERROR] 电解槽产氢错误")
    
    # 测试燃料电池最小功率约束
    print("\n[燃料电池最小功率约束测试]")
    env.reset()
    env.SOCH = 0.5  # 确保有足够氢气
    env.H2_storage_kg = env.SOCH * env.M_HS_max
    
    # 尝试低于最小功率运行
    action = np.array([0.0, -1.0, 0.05])  # 约15kW，低于30kW最小功率
    _, _, _, info = env.step(action)
    
    p_fc = info['real_p_fc']
    print(f"请求功率: ~15 kW")
    print(f"最小功率: {env.P_fc_min:.2f} kW")
    print(f"实际功率: {p_fc:.2f} kW")
    
    if p_fc == 0.0:
        print("[OK] 最小功率约束生效（低于最小功率则关闭）")
    else:
        print(f"[ERROR] 最小功率约束失效")
    
    # 测试高于最小功率
    action = np.array([0.0, -1.0, 0.5])  # 约150kW，高于最小功率
    _, _, _, info = env.step(action)
    p_fc = info['real_p_fc']
    
    print(f"\n请求功率: ~150 kW")
    print(f"实际功率: {p_fc:.2f} kW")
    
    if p_fc > env.P_fc_min:
        print("[OK] 高于最小功率时正常运行")
    else:
        print(f"[ERROR] 功率约束错误")


def test_energy_balance():
    """测试能量平衡"""
    print("\n" + "="*60)
    print("测试3: 能量平衡")
    print("="*60)
    
    df_test = pd.DataFrame({
        'wind_speed': [12.0] * 10,
        'irradiance': [800.0] * 10,
        'temperature': [25.0] * 10,
        'load': [3000.0] * 10
    })
    
    env = H2RESEnv(df_test, df_test)
    state = env.reset()
    
    print("\n[能量平衡测试]")
    action = np.array([0.1, 0.5, 0.3])  # 混合运行
    _, _, _, info = env.step(action)
    
    # 计算供应和需求
    p_gen = env.current_p_wt + env.current_p_pv
    p_supply = p_gen + info['real_p_fc']
    if info['real_p_bat'] < 0:
        p_supply += abs(info['real_p_bat'])
    
    p_demand = env.current_load + info['real_p_el']
    if info['real_p_bat'] > 0:
        p_demand += info['real_p_bat']
    
    balance = p_supply - p_demand
    p_dump = info['p_dump']
    p_unmet = info['p_unmet']
    
    print(f"发电功率: {p_gen:.2f} kW")
    print(f"  - 风电: {env.current_p_wt:.2f} kW")
    print(f"  - 光伏: {env.current_p_pv:.2f} kW")
    print(f"燃料电池: {info['real_p_fc']:.2f} kW")
    print(f"电池: {info['real_p_bat']:.2f} kW")
    print(f"\n负荷: {env.current_load:.2f} kW")
    print(f"电解槽: {info['real_p_el']:.2f} kW")
    print(f"\n供应总计: {p_supply:.2f} kW")
    print(f"需求总计: {p_demand:.2f} kW")
    print(f"平衡: {balance:.2f} kW")
    print(f"弃电: {p_dump:.2f} kW")
    print(f"缺电: {p_unmet:.2f} kW")
    
    # 验证
    if abs(balance - p_dump + p_unmet) < 1e-3:
        print("[OK] 能量平衡正确")
    else:
        print(f"[ERROR] 能量平衡错误")


def test_constraints():
    """测试约束满足"""
    print("\n" + "="*60)
    print("测试4: 约束满足")
    print("="*60)
    
    df_test = pd.DataFrame({
        'wind_speed': [12.0] * 100,
        'irradiance': [800.0] * 100,
        'temperature': [25.0] * 100,
        'load': [3000.0] * 100
    })
    
    env = H2RESEnv(df_test, df_test)
    state = env.reset()
    
    violations = {
        'soc_min': 0,
        'soc_max': 0,
        'soch_min': 0,
        'soch_max': 0
    }
    
    print("\n[运行50步测试约束]")
    for i in range(50):
        action = np.random.uniform(-1, 1, 3)
        state, reward, done, info = env.step(action)
        
        if info['soc'] < env.SOC_min - 1e-6:
            violations['soc_min'] += 1
        if info['soc'] > env.SOC_max + 1e-6:
            violations['soc_max'] += 1
        if info['soch'] < env.SOCH_min - 1e-6:
            violations['soch_min'] += 1
        if info['soch'] > env.SOCH_max + 1e-6:
            violations['soch_max'] += 1
    
    print(f"\nSOC范围: [{env.SOC_min}, {env.SOC_max}]")
    print(f"SOCH范围: [{env.SOCH_min}, {env.SOCH_max}]")
    print(f"\n约束违反统计:")
    print(f"  SOC < {env.SOC_min}: {violations['soc_min']} 次")
    print(f"  SOC > {env.SOC_max}: {violations['soc_max']} 次")
    print(f"  SOCH < {env.SOCH_min}: {violations['soch_min']} 次")
    print(f"  SOCH > {env.SOCH_max}: {violations['soch_max']} 次")
    
    total_violations = sum(violations.values())
    if total_violations == 0:
        print("\n[OK] 所有约束均满足")
    else:
        print(f"\n[ERROR] 发现 {total_violations} 次约束违反")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("H2-RES 物理一致性验证")
    print("="*60)
    
    try:
        test_battery_efficiency()
        test_hydrogen_system()
        test_energy_balance()
        test_constraints()
        
        print("\n" + "="*60)
        print("✅ 所有物理验证测试完成")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()