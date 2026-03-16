# SRΨ-v2.0 Field-Aware Training - Colab Experiment Manifest

**实验名称**: 动态场状态感知训练验证
**实验日期**: 2026-03-16
**实验者**: Genesis-OS Research Team
**算力节点**: Google Colab (GPU)

---

## 🎯 实验目标

验证 TRAE 的动态场状态感知训练机制是否能够：
1. **自动调整优化策略**：根据 Resonance 动态切换"数据拟合"与"物理一致性"
2. **提升物理守恒性**：相比静态权重，Energy Drift 和 Momentum Drift 是否更低
3. **稳定训练过程**：Resonance 是否随训练逐渐收敛到稳定状态

---

## 📍 当前场状态

```yaml
项目: srpsi-engine-tiny
版本: v2.0-dynamic-field-aware
阶段: dynamic_field_aware_training
关键突破: Loss 函数从静态约束进化到动态场状态感知

核心机制:
  - Phase-Aware Training: 每个 epoch 读取场状态
  - IntegratedConstraint: 根据场状态动态调整权重
  - get_coupling_weights(): 感知 Resonance 并自适应

动态权重公式:
  fitting_weight = 1.0 - (resonance * 0.5)
  consistency_scale = 1.0 + resonance

行为谱系:
  resonance_0.0: 纯数据拟合 (fitting=1.0, consistency=1.0)
  resonance_0.5: 平衡模式 (fitting=0.75, consistency=1.5)
  resonance_1.0: 物理一致性优先 (fitting=0.5, consistency=2.0)
```

---

## 🔧 实验配置

```yaml
模型: SRΨ v2.0 Hybrid
任务: 1D Burgers 方程场演化预测
数据集: Burgers 1D (4800 samples)

训练配置:
  epochs: 80
  batch_size: 32
  learning_rate: 0.0001
  scheduler: CosineAnnealingLR

数据划分:
  train: 4000 samples
  val: 400 samples
  test: 400 samples

输入/输出:
  tin: 16 (历史时间步)
  tout: 32 (预测时间步)
  nx: 128 (空间点数)
```

---

## 📊 关键观察指标

### 1. **场状态指标**
- **Resonance** (每个 epoch):
  - 目标范围: 0.5 → 0.85
  - 预期: 随训练逐渐上升并稳定
  - 含义: 空间算子与时间算子的对齐度

- **Phase**:
  - 预期: 从 evolving → stable
  - 含义: 场稳定性状态

### 2. **训练指标**
- **Train Loss**: 总体损失
- **Val Loss**: 验证损失
- **Learning Rate**: 学习率变化

### 3. **物理守恒指标** (核心!)
- **Energy Drift**:
  - v1.0 基线: 10.883
  - 目标: < 10.0 (相比 v1.0 改善)
  - 含义: 能量守恒保持能力

- **Momentum Drift**:
  - 目标: < 5.0
  - 含义: 动量守恒保持能力

### 4. **动态权重指标**
- **Fitting Weight**:
  - 预期范围: 0.5 ~ 0.75
  - 行为: Resonance 高时下降

- **Consistency Scale**:
  - 预期范围: 1.5 ~ 2.0
  - 行为: Resonance 高时上升

---

## ✅ 成功判据

### 最小成功条件:
1. ✅ 训练能够运行 80 epochs 不崩溃
2. ✅ Resonance 能够被正确计算和记录
3. ✅ 动态权重机制能够正常工作

### 预期成功条件:
1. ✅ Resonance 随训练收敛到 > 0.7
2. ✅ Energy Drift < 10.0 (优于 v1.0 的 10.883)
3. ✅ Phase 最终稳定为 "stable"

### 理想成功条件:
1. ✅ Resonance 收敛到 > 0.85
2. ✅ Energy Drift < 9.0 (显著优于 v1.0)
3. ✅ 能够观察到 Resonance 驱动的策略切换

---

## ❌ 失败判据

### 训练失败:
1. ❌ 训练在中途崩溃或 Loss 爆炸
2. ❌ Resonance 计算错误或始终为 0
3. ❌ 动态权重机制未生效

### 性能失败:
1. ❌ Energy Drift > 12.0 (劣于 v1.0)
2. ❌ Resonance 始终 < 0.5 (场状态不稳定)
3. ❌ Phase 始终为 "evolving" (未能收敛)

---

## 📝 实验步骤

### Phase 1: 环境准备 (5 min)
```python
# 1. 挂载 Google Drive
# 2. 克隆仓库
# 3. 安装依赖
# 4. 检查 GPU 可用性
```

### Phase 2: 数据准备 (5 min)
```python
# 1. 生成 Burgers 1D 数据集
# 2. 验证数据格式
# 3. 创建 DataLoader
```

### Phase 3: 模型训练 (60-90 min)
```python
# 1. 创建 SRΨ v2.0 Hybrid 模型
# 2. 创建 IntegratedConstraint (动态权重)
# 3. 运行 80 epochs 训练
# 4. 每个 epoch 记录:
#    - Resonance
#    - Phase
#    - Loss components
#    - 动态权重
```

### Phase 4: 结果分析 (10 min)
```python
# 1. 绘制 Resonance 曲线
# 2. 绘制动态权重曲线
# 3. 绘制 Energy Drift 对比
# 4. 生成实验报告
```

---

## 📤 输出要求

### 必须输出:
1. **训练日志**: 每个 epoch 的完整指标
2. **Resonance 曲线**: 如何随训练变化
3. **动态权重曲线**: fitting_weight 和 consistency_scale 的变化
4. **物理守恒指标**: Energy Drift 和 Momentum Drift 最终值
5. **Checkpoints**: 最佳模型和最终模型

### 可选输出:
1. **Loss 曲线**: Train/Val Loss 变化
2. **权重切换可视化**: 何时从"拟合优先"切换到"物理优先"
3. **与 v1.0 对比**: 性能提升百分比

---

## 🔍 关键验证点

### Epoch 0-20 (探索阶段):
- [ ] Resonance 能够正确计算
- [ ] 动态权重机制开始工作
- [ ] Loss 能够正常下降

### Epoch 20-50 (收敛阶段):
- [ ] Resonance 开始上升 (> 0.5)
- [ ] Phase 从 evolving → stable
- [ ] 能看到权重策略切换

### Epoch 50-80 (稳定阶段):
- [ ] Resonance 稳定在 > 0.7
- [ ] Energy Drift < 10.0
- [ ] Phase 最终为 "stable"

---

## 💾 数据保存

### 保存到 Google Drive:
```
/srpsi-engine-tiny/colab_runs/
  ├── run_2026-03-16/
  │   ├── checkpoints/
  │   │   ├── best_model.pt
  │   │   └── final_model.pt
  │   ├── logs/
  │   │   ├── training_log.txt
  │   │   └── metrics.json
  │   └── figures/
  │       ├── resonance_curve.png
  │       ├── dynamic_weights.png
  │       └── energy_drift.png
```

---

## 🎓 实验背景阅读

### 必读文档:
1. **EXPERIMENT_LOG.md** - Entry 0009, 0010
2. **TRAE_INSIGHTS.md** - 完整阅读
3. **train_v2_hybrid.py** - 重点关注 get_field_reading()
4. **src/training/physical_loss.py** - 重点关注 get_coupling_weights()

### 核心理解:
- **Phase-Aware Training**: 每个 epoch 读取场状态
- **动态权重机制**: fitting_weight 和 consistency_scale 根据 Resonance 调整
- **场驱动优化**: 优化方向根据场状态自适应，而非固定

---

## 🚨 注意事项

1. **不要修改训练脚本** - 使用提供的 train_v2_hybrid.py
2. **保持数据一致性** - 使用相同的 random seed (42)
3. **记录所有异常** - 如果训练失败，记录错误信息
4. **观察 Resonance** - 这是本次实验的核心指标
5. **耐心等待** - 80 epochs 可能需要 60-90 分钟

---

## 📮 实验完成后

1. **保存所有结果** 到 Google Drive
2. **生成实验报告** (简要总结)
3. **回写 EXPERIMENT_LOG.md** (如果有新的发现)
4. **通知团队** 实验完成

---

**Status**: Ready to Run
**Priority**: Highest
**Expected Duration**: 90-120 minutes
**Success Probability**: High (基于 v1.0 的成功基础)

---

**备注**: 这是一个安静的算力节点，但不是"瞎跑"。每一个 epoch 都在感知场状态，每一个权重都在动态调整。这就是"场状态感知训练"的真正实现。🌟
