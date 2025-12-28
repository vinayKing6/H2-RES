# H2-RES DDPG 训练指南

## 问题诊断与修复总结

### 原始问题
用户报告训练后的可视化结果显示：
- **电解槽功率恒为0**
- **电池功率恒定在300kW**（实际是-300kW放电，但SOC=0.3无法继续放电）
- **SOC始终保持在0.3**（下限约束）

### 根本原因
1. **电池放电逻辑缺陷**：放电请求被错误处理，导致电池无法正常放电
2. **智能分配算法过于偏向电解槽**：当SOC=0.3时，电池无法获得充电功率
3. **死锁状态**：SOC=0.3 → 无法放电 → 请求充电 → 电解槽抢走所有功率 → SOC永远=0.3

### 修复方案

#### 1. 修复电池充放电逻辑
**文件**: `src/envs/h2_res_env.py` (行182-199)

```python
# 修复前：放电请求被忽略
if p_surplus_initial > 0 and raw_bat_action > 0:
    # 充电场景
    ...
else:
    # 放电场景被错误处理
    p_bat_allocated = raw_bat_action if raw_bat_action > 0 else 0.0  # ❌ 放电被设为0

# 修复后：正确处理放电
elif raw_bat_action < 0:
    # 放电场景：直接使用原始请求
    p_el_allocated = 0.0
    p_bat_allocated = raw_bat_action  # ✓ 保持负值
```

#### 2. 添加SOC低位保护机制
**文件**: `src/envs/h2_res_env.py` (行122-176)

```python
def _allocate_power_intelligently(self, p_surplus, raw_el, raw_bat_charge, current_soc):
    # 新增：SOC危险低（<0.35）时，优先给电池充电
    if current_soc < 0.35 and raw_bat_charge > 0:
        # 给电池至少50%的剩余功率
        bat_priority = min(raw_bat_charge, p_surplus * 0.5)
        remaining = p_surplus - bat_priority
        
        if remaining >= self.P_el_min:
            el_allocated = min(raw_el, remaining)
            return el_allocated, bat_priority
        else:
            return 0.0, min(raw_bat_charge, p_surplus)
```

#### 3. 改进奖励函数
**文件**: `src/envs/h2_res_env.py` (行333-380)

```python
# 新增：SOC状态奖励
if 0.4 <= self.SOC <= 0.8:
    r_soc = 2.0  # 健康范围内给予正奖励
elif self.SOC < 0.4:
    r_soc = -3.0 * (0.4 - self.SOC)  # 低于0.4惩罚加重
else:
    r_soc = 0.0

# 增强SOC低位惩罚
penalty_soc_low = 10.0 if self.SOC <= self.SOC_min + 0.01 else 0.0
```

## 训练步骤

### 1. 删除旧模型
```bash
del results\ddpg_checkpoint.pth
del results\training_rewards.npy
```

### 2. 运行训练
```bash
python train.py
```

**建议训练参数**（已在`train.py`中配置）：
- `MAX_EPISODES`: 10000-20000
- `MAX_STEPS`: 24-48小时
- `BATCH_SIZE`: 64
- `LR_ACTOR`: 1e-4
- `LR_CRITIC`: 1e-3
- `NOISE`: 0.2

### 3. 监控训练
观察控制台输出，奖励应该：
- 初期：负值（-50 ~ -20）
- 中期：逐渐上升（-10 ~ 0）
- 后期：稳定在正值（5 ~ 15）

### 4. 生成可视化
```bash
python advanced_visualization.py
```

## 预期结果

### 正常的可视化特征

#### 子图(a) - 可再生能源发电与负荷
- 风电和光伏曲线应该有明显的日夜变化
- 负荷曲线呈现双峰特征（早晚高峰）

#### 子图(b) - 储能系统响应
- **电解槽**：在光伏高峰期启动（功率≥300kW或为0）
- **燃料电池**：在负荷高峰且可再生能源不足时启动
- **电池功率**：
  - 白天：正值（充电），范围0-1000kW
  - 晚上：负值（放电），范围-1000-0kW
  - 应该呈现波动，不应恒定

#### 子图(c) - 储能状态
- **电池SOC**：
  - 应在[0.4, 0.8]范围内波动
  - 白天上升，晚上下降
  - **不应长期停留在0.3**
- **氢罐SOCH**：
  - 应在[0.3, 0.8]范围内缓慢变化
  - 长期趋势取决于制氢和用氢的平衡

#### 子图(d) - 功率流动热力图
- 应该看到明显的昼夜模式
- 白天：电解槽和电池充电活跃
- 晚上：燃料电池和电池放电活跃

#### 子图(e) - 日能量平衡
- 供应侧和需求侧应该基本平衡
- 弃电应该较少（<10%总发电量）

#### 子图(f) - 系统性能指标
- 可再生能源利用率：>80%
- 负荷满足率：>95%
- 制氢速率：20-40%
- 储能效率：60-80%

## 故障排查

### 如果SOC仍然恒为0.3
1. 检查是否删除了旧模型
2. 检查训练是否收敛（查看`results/training_rewards.npy`）
3. 运行测试脚本验证环境：
   ```bash
   python test_battery_logic.py
   python test_soc_priority.py
   ```

### 如果电解槽一直为0
- 可能原因：氢罐已满（SOCH=0.9）
- 解决方案：增加燃料电池使用，或降低电解槽优先级

### 如果训练不收敛
- 增加训练轮数（MAX_EPISODES）
- 调整学习率（降低LR_ACTOR和LR_CRITIC）
- 检查奖励函数权重是否合理

## 技术细节

### 关键约束
- **电解槽最小功率**：300kW（10% × 3000kW）
- **燃料电池最小功率**：30kW（10% × 300kW）
- **SOC范围**：[0.3, 0.95]
- **SOCH范围**：[0.2, 0.9]

### 智能分配优先级
1. **SOC < 0.35**：电池获得≥50%剩余功率
2. **SOC ≥ 0.35**：按Agent请求比例分配
3. **电解槽最小功率约束**：必须≥300kW或为0

### 奖励函数权重
- 可再生能源利用率：×10（最高优先级）
- 负荷满足率：×8
- SOC健康奖励：+2（0.4-0.8范围）
- SOC低位惩罚：-3×(0.4-SOC)
- 弃电惩罚：×0.5
- 缺电惩罚：×50（最严重）

## 参考文献
- DDPG算法：Lillicrap et al. (2015) "Continuous control with deep reinforcement learning"
- H2-RES系统：相关氢能-可再生能源系统论文

## 联系与支持
如有问题，请检查：
1. 环境配置是否正确（Python 3.8+, PyTorch, Gym）
2. 数据文件是否存在（`src/data/*.xlsx`）
3. 测试脚本是否全部通过