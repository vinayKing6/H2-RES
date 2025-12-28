"""
测试电池充放电逻辑是否正确
"""
import sys
import os
import numpy as np
import pandas as pd

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.envs.h2_res_env import H2RESEnv

def test_battery_discharge():
    """测试电池放电功能"""
    print("\n" + "="*60)
    print("测试1: 电池放电逻辑")
    print("="*60)
    
    # 创建简单测试数据
    df_test = pd.DataFrame({
        'wind_speed': [0.0] * 24,  # 无风
        'irradiance': [0.0] * 24,  # 无光
        'temperature': [25.0] * 24,
        'load': [2000.0] * 24  # 恒定负荷2000kW
    })
    
    env = H2RESEnv(df_test, df_test)
    
    # 初始化：高SOC，可以放电
    state = env.reset(init_soc=0.8, init_soch=0.5)
    print(f"初始状态: SOC={env.SOC:.3f}, SOCH={env.SOCH:.3f}")
    
    # 测试放电动作
    # action[0] = -0.5 表示请求放电 500kW
    # action[1] = -1.0 表示不启动电解槽
    # action[2] = 0.5 表示启动燃料电池
    action = np.array([-0.5, -1.0, 0.5])
    
    next_state, reward, done, info = env.step(action)
    
    print(f"\n动作: 电池={action[0]:.2f}, 电解槽={action[1]:.2f}, 燃料电池={action[2]:.2f}")
    print(f"实际执行:")
    print(f"  - 电池功率: {info['real_p_bat']:.2f} kW (应为负值表示放电)")
    print(f"  - 燃料电池: {info['real_p_fc']:.2f} kW")
    print(f"  - 电解槽: {info['real_p_el']:.2f} kW")
    print(f"  - 缺电: {info['p_unmet']:.2f} kW")
    print(f"结果: SOC={env.SOC:.3f} (应该下降)")
    
    if info['real_p_bat'] < 0:
        print("[OK] 电池放电成功")
        return True
    else:
        print("[FAIL] 电池放电失败！")
        return False


def test_battery_charge():
    """测试电池充电功能"""
    print("\n" + "="*60)
    print("测试2: 电池充电逻辑")
    print("="*60)
    
    # 创建测试数据：高可再生能源
    df_test = pd.DataFrame({
        'wind_speed': [15.0] * 24,  # 强风
        'irradiance': [800.0] * 24,  # 强光
        'temperature': [25.0] * 24,
        'load': [1000.0] * 24  # 低负荷
    })
    
    env = H2RESEnv(df_test, df_test)
    
    # 初始化：低SOC，可以充电
    state = env.reset(init_soc=0.4, init_soch=0.5)
    print(f"初始状态: SOC={env.SOC:.3f}, SOCH={env.SOCH:.3f}")
    print(f"可再生能源: 风电={env.current_p_wt:.0f}kW, 光伏={env.current_p_pv:.0f}kW")
    print(f"剩余功率: {env.current_p_wt + env.current_p_pv - env.current_load:.0f}kW")
    
    # 测试充电动作
    # action[0] = 0.5 表示请求充电 500kW
    # action[1] = 0.5 表示电解槽请求1500kW
    # action[2] = -1.0 表示不启动燃料电池
    action = np.array([0.5, 0.5, -1.0])
    
    next_state, reward, done, info = env.step(action)
    
    print(f"\n动作: 电池={action[0]:.2f}, 电解槽={action[1]:.2f}, 燃料电池={action[2]:.2f}")
    print(f"实际执行:")
    print(f"  - 电池功率: {info['real_p_bat']:.2f} kW (应为正值表示充电)")
    print(f"  - 电解槽: {info['real_p_el']:.2f} kW")
    print(f"  - 燃料电池: {info['real_p_fc']:.2f} kW")
    print(f"  - 弃电: {info['p_dump']:.2f} kW")
    print(f"结果: SOC={env.SOC:.3f} (应该上升)")
    
    if info['real_p_bat'] > 0 and env.SOC > 0.4:
        print("[OK] 电池充电成功")
        return True
    else:
        print("[FAIL] 电池充电失败！")
        return False


def test_soc_dynamics():
    """测试SOC动态变化"""
    print("\n" + "="*60)
    print("测试3: SOC动态变化（24小时）")
    print("="*60)
    
    # 创建昼夜循环数据
    hours = np.arange(24)
    df_test = pd.DataFrame({
        'wind_speed': 8.0 + 4.0 * np.sin(hours * np.pi / 12),
        'irradiance': np.maximum(0, 800 * np.sin((hours - 6) * np.pi / 12)),
        'temperature': [25.0] * 24,
        'load': 2000 + 500 * np.sin(hours * np.pi / 12)
    })
    
    env = H2RESEnv(df_test, df_test)
    state = env.reset(init_soc=0.5, init_soch=0.5)
    
    soc_history = [env.SOC]
    bat_power_history = []
    
    print(f"初始SOC: {env.SOC:.3f}")
    
    for hour in range(24):
        # 简单策略：白天充电，晚上放电
        p_surplus = env.current_p_wt + env.current_p_pv - env.current_load
        
        if p_surplus > 500:  # 剩余功率充足
            action = np.array([0.3, 0.3, -1.0])  # 充电
        elif p_surplus < -500:  # 功率不足
            action = np.array([-0.3, -1.0, 0.5])  # 放电
        else:
            action = np.array([0.0, 0.0, 0.0])  # 不动作
        
        next_state, reward, done, info = env.step(action)
        
        soc_history.append(env.SOC)
        bat_power_history.append(info['real_p_bat'])
        
        state = next_state
    
    print(f"最终SOC: {env.SOC:.3f}")
    print(f"SOC变化范围: [{min(soc_history):.3f}, {max(soc_history):.3f}]")
    print(f"电池功率范围: [{min(bat_power_history):.0f}, {max(bat_power_history):.0f}] kW")
    
    # 检查SOC是否有变化
    soc_changed = max(soc_history) - min(soc_history) > 0.05
    has_charge = any(p > 0 for p in bat_power_history)
    has_discharge = any(p < 0 for p in bat_power_history)
    
    if soc_changed and has_charge and has_discharge:
        print("[OK] SOC动态变化正常")
        return True
    else:
        print("[FAIL] SOC动态异常！")
        print(f"  - SOC变化: {soc_changed}")
        print(f"  - 有充电: {has_charge}")
        print(f"  - 有放电: {has_discharge}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  电池充放电逻辑测试")
    print("="*60)
    
    results = []
    
    results.append(("电池放电", test_battery_discharge()))
    results.append(("电池充电", test_battery_charge()))
    results.append(("SOC动态", test_soc_dynamics()))
    
    print("\n" + "="*60)
    print("  测试结果汇总")
    print("="*60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n[SUCCESS] 所有测试通过！电池逻辑修复成功。")
        print("建议：删除旧的训练结果，重新训练模型。")
        print("命令：")
        print("  del results\\ddpg_checkpoint.pth")
        print("  python train.py")
    else:
        print("\n[ERROR] 部分测试未通过，需要进一步调试。")
    
    print("="*60 + "\n")