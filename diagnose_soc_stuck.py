"""
诊断SOC卡在0.3的原因
"""
import sys
import os
import numpy as np
import pandas as pd
import torch

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.envs.h2_res_env import H2RESEnv
from src.algos.ddpg import DDPGAgent

def diagnose_with_trained_model():
    """使用训练好的模型诊断"""
    print("\n" + "="*60)
    print("诊断：使用训练好的模型")
    print("="*60)
    
    # 创建测试数据：高可再生能源
    df_test = pd.DataFrame({
        'wind_speed': [15.0] * 48,  # 强风
        'irradiance': [0, 0, 0, 0, 0, 0, 200, 400, 600, 800, 900, 1000,
                       1000, 900, 800, 600, 400, 200, 0, 0, 0, 0, 0, 0] * 2,  # 昼夜循环
        'temperature': [25.0] * 48,
        'load': [2000.0] * 48
    })
    
    env = H2RESEnv(df_test, df_test)
    
    # 加载训练好的模型
    agent = DDPGAgent(env.observation_space.shape[0],
                     env.action_space.shape[0])
    
    if os.path.exists('results/ddpg_checkpoint.pth'):
        agent.load('results/ddpg_checkpoint.pth')
        print("[OK] 已加载训练好的模型\n")
    else:
        print("[ERROR] 未找到模型文件！")
        return
    
    # 从SOC=0.3开始
    state = env.reset(init_soc=0.3, init_soch=0.5)
    
    print(f"初始状态: SOC={env.SOC:.3f}, SOCH={env.SOCH:.3f}\n")
    print("="*80)
    print(f"{'步骤':<4} {'时刻':<6} {'风电':<6} {'光伏':<6} {'剩余':<7} {'动作':<25} {'实际':<30} {'SOC':<6}")
    print("="*80)
    
    for step in range(24):
        # 使用训练好的策略
        action = agent.select_action(state, noise=False)
        
        # 记录当前状态
        p_wt = env.current_p_wt
        p_pv = env.current_p_pv
        p_surplus = p_wt + p_pv - env.current_load
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 解析动作
        raw_bat = action[0] * 1000
        raw_el = (action[1] + 1) / 2 * 3000
        raw_fc = (action[2] + 1) / 2 * 300
        
        action_str = f"B:{raw_bat:+5.0f} E:{raw_el:4.0f} F:{raw_fc:3.0f}"
        actual_str = f"B:{info['real_p_bat']:+5.0f} E:{info['real_p_el']:4.0f} F:{info['real_p_fc']:3.0f}"
        
        print(f"{step:<4} {step:02d}:00  {p_wt:5.0f}  {p_pv:5.0f}  {p_surplus:+6.0f}  {action_str}  {actual_str}  {env.SOC:.3f}")
        
        state = next_state
        
        if done:
            break
    
    print("="*80)
    print(f"\n最终SOC: {env.SOC:.3f}")
    
    if env.SOC <= 0.31:
        print("\n[CRITICAL] SOC仍然卡在0.3附近！")
        print("\n可能原因分析:")
        print("1. Agent学到了错误的策略（一直请求大功率充电但实际无法充电）")
        print("2. 电池充电被电解槽完全抢占")
        print("3. 奖励函数设计问题导致Agent不关心SOC")
        print("4. 训练不充分（只有2200个episode）")
    else:
        print(f"\n[OK] SOC成功上升到 {env.SOC:.3f}")


def diagnose_manual_actions():
    """使用手动动作诊断"""
    print("\n" + "="*60)
    print("诊断：使用手动动作（强制电池充电）")
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
    print(f"可再生能源: 风电={env.current_p_wt:.0f}kW, 光伏={env.current_p_pv:.0f}kW")
    print(f"剩余功率: {env.current_p_wt + env.current_p_pv - env.current_load:.0f}kW\n")
    
    # 测试1：只充电池，不启动电解槽
    print("测试1: 只充电池（电池=+800kW, 电解槽=0kW）")
    action = np.array([0.8, -1.0, -1.0])  # 电池800kW，电解槽0，燃料电池0
    next_state, reward, done, info = env.step(action)
    print(f"  实际: 电池={info['real_p_bat']:.0f}kW, 电解槽={info['real_p_el']:.0f}kW")
    print(f"  结果: SOC={env.SOC:.3f} (变化={env.SOC-0.3:.3f})")
    print(f"  奖励: {reward:.2f}\n")
    
    # 重置
    env.reset(init_soc=0.3, init_soch=0.5)
    
    # 测试2：电池和电解槽都请求
    print("测试2: 电池+电解槽（电池=+500kW, 电解槽=1500kW）")
    action = np.array([0.5, 0.5, -1.0])
    next_state, reward, done, info = env.step(action)
    print(f"  实际: 电池={info['real_p_bat']:.0f}kW, 电解槽={info['real_p_el']:.0f}kW")
    print(f"  结果: SOC={env.SOC:.3f} (变化={env.SOC-0.3:.3f})")
    print(f"  奖励: {reward:.2f}\n")
    
    # 重置
    env.reset(init_soc=0.3, init_soch=0.5)
    
    # 测试3：小功率充电
    print("测试3: 小功率充电（电池=+200kW, 电解槽=0kW）")
    action = np.array([0.2, -1.0, -1.0])
    next_state, reward, done, info = env.step(action)
    print(f"  实际: 电池={info['real_p_bat']:.0f}kW, 电解槽={info['real_p_el']:.0f}kW")
    print(f"  结果: SOC={env.SOC:.3f} (变化={env.SOC-0.3:.3f})")
    print(f"  奖励: {reward:.2f}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SOC卡在0.3的诊断工具")
    print("="*60)
    
    # 先测试手动动作
    diagnose_manual_actions()
    
    # 再测试训练好的模型
    diagnose_with_trained_model()
    
    print("\n" + "="*60)
    print("  诊断完成")
    print("="*60 + "\n")