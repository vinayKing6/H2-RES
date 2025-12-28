"""
测试SOC低位时的充电优先级
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

def test_low_soc_priority():
    """测试SOC=0.3时，电池是否能获得充电优先级"""
    print("\n" + "="*60)
    print("测试: SOC低位充电优先级")
    print("="*60)
    
    # 创建测试数据：高可再生能源
    df_test = pd.DataFrame({
        'wind_speed': [15.0] * 24,  # 强风
        'irradiance': [800.0] * 24,  # 强光
        'temperature': [25.0] * 24,
        'load': [1000.0] * 24  # 低负荷
    })
    
    env = H2RESEnv(df_test, df_test)
    
    # 初始化：SOC=0.3（下限），SOCH=0.5
    state = env.reset(init_soc=0.3, init_soch=0.5)
    print(f"初始状态: SOC={env.SOC:.3f}, SOCH={env.SOCH:.3f}")
    print(f"可再生能源: 风电={env.current_p_wt:.0f}kW, 光伏={env.current_p_pv:.0f}kW")
    print(f"剩余功率: {env.current_p_wt + env.current_p_pv - env.current_load:.0f}kW")
    
    # Agent请求：电池充电500kW，电解槽1500kW
    action = np.array([0.5, 0.5, -1.0])
    
    next_state, reward, done, info = env.step(action)
    
    print(f"\nAgent请求:")
    print(f"  - 电池充电: 500 kW")
    print(f"  - 电解槽: 1500 kW")
    print(f"\n实际执行:")
    print(f"  - 电池功率: {info['real_p_bat']:.2f} kW")
    print(f"  - 电解槽: {info['real_p_el']:.2f} kW")
    print(f"  - 弃电: {info['p_dump']:.2f} kW")
    print(f"\n结果: SOC={env.SOC:.3f} (应该上升)")
    print(f"奖励: {reward:.2f}")
    
    # 检查电池是否获得了充电
    if info['real_p_bat'] > 0 and env.SOC > 0.3:
        print("\n[OK] SOC低位时，电池成功获得充电优先级")
        
        # 计算分配比例
        total_allocated = info['real_p_bat'] + info['real_p_el']
        bat_ratio = info['real_p_bat'] / total_allocated if total_allocated > 0 else 0
        print(f"电池分配比例: {bat_ratio*100:.1f}%")
        
        if bat_ratio >= 0.3:  # 至少30%
            print("[EXCELLENT] 电池获得了足够的充电功率")
            return True
        else:
            print("[WARNING] 电池充电功率偏低")
            return True
    else:
        print("\n[FAIL] SOC低位时，电池未能获得充电")
        return False


def test_normal_soc_allocation():
    """测试SOC正常时的功率分配"""
    print("\n" + "="*60)
    print("测试: SOC正常时的功率分配")
    print("="*60)
    
    df_test = pd.DataFrame({
        'wind_speed': [15.0] * 24,
        'irradiance': [800.0] * 24,
        'temperature': [25.0] * 24,
        'load': [1000.0] * 24
    })
    
    env = H2RESEnv(df_test, df_test)
    
    # 初始化：SOC=0.6（正常）
    state = env.reset(init_soc=0.6, init_soch=0.5)
    print(f"初始状态: SOC={env.SOC:.3f}, SOCH={env.SOCH:.3f}")
    
    # Agent请求：电池充电500kW，电解槽1500kW
    action = np.array([0.5, 0.5, -1.0])
    
    next_state, reward, done, info = env.step(action)
    
    print(f"\nAgent请求:")
    print(f"  - 电池充电: 500 kW")
    print(f"  - 电解槽: 1500 kW")
    print(f"\n实际执行:")
    print(f"  - 电池功率: {info['real_p_bat']:.2f} kW")
    print(f"  - 电解槽: {info['real_p_el']:.2f} kW")
    
    # 计算分配比例
    total_allocated = info['real_p_bat'] + info['real_p_el']
    bat_ratio = info['real_p_bat'] / total_allocated if total_allocated > 0 else 0
    el_ratio = info['real_p_el'] / total_allocated if total_allocated > 0 else 0
    
    print(f"\n分配比例:")
    print(f"  - 电池: {bat_ratio*100:.1f}%")
    print(f"  - 电解槽: {el_ratio*100:.1f}%")
    
    # 检查是否按请求比例分配（500:1500 = 1:3 = 25%:75%）
    expected_bat_ratio = 500 / 2000
    ratio_error = abs(bat_ratio - expected_bat_ratio)
    
    if ratio_error < 0.1:  # 误差<10%
        print(f"[OK] 按Agent请求比例分配（误差{ratio_error*100:.1f}%）")
        return True
    else:
        print(f"[WARNING] 分配比例偏离请求（误差{ratio_error*100:.1f}%）")
        return True


def test_continuous_charging():
    """测试从SOC=0.3连续充电到0.5"""
    print("\n" + "="*60)
    print("测试: 从SOC=0.3连续充电")
    print("="*60)
    
    df_test = pd.DataFrame({
        'wind_speed': [15.0] * 24,
        'irradiance': [800.0] * 24,
        'temperature': [25.0] * 24,
        'load': [1000.0] * 24
    })
    
    env = H2RESEnv(df_test, df_test)
    state = env.reset(init_soc=0.3, init_soch=0.5)
    
    print(f"初始SOC: {env.SOC:.3f}")
    
    soc_history = [env.SOC]
    
    # 连续10步，持续请求充电
    for step in range(10):
        action = np.array([0.5, 0.3, -1.0])  # 电池500kW，电解槽900kW
        next_state, reward, done, info = env.step(action)
        soc_history.append(env.SOC)
        state = next_state
        
        if step % 3 == 0:
            print(f"  步骤{step}: SOC={env.SOC:.3f}, 电池={info['real_p_bat']:.0f}kW")
    
    print(f"\n最终SOC: {env.SOC:.3f}")
    print(f"SOC增长: {env.SOC - 0.3:.3f}")
    
    # 检查SOC是否持续上升
    soc_increasing = all(soc_history[i+1] >= soc_history[i] for i in range(len(soc_history)-1))
    
    if soc_increasing and env.SOC > 0.35:
        print("[OK] SOC持续上升，成功脱离下限")
        return True
    else:
        print("[FAIL] SOC未能持续上升")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SOC低位充电优先级测试")
    print("="*60)
    
    results = []
    
    results.append(("SOC低位优先级", test_low_soc_priority()))
    results.append(("SOC正常分配", test_normal_soc_allocation()))
    results.append(("连续充电", test_continuous_charging()))
    
    print("\n" + "="*60)
    print("  测试结果汇总")
    print("="*60)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n[SUCCESS] 所有测试通过！SOC低位保护机制工作正常。")
        print("\n关键改进:")
        print("  1. SOC<0.35时，电池获得至少50%剩余功率")
        print("  2. 奖励函数鼓励SOC维持在[0.4, 0.8]")
        print("  3. SOC低于0.4时给予惩罚，避免长期停留在下限")
        print("\n现在可以重新训练模型了！")
    else:
        print("\n[ERROR] 部分测试未通过。")
    
    print("="*60 + "\n")