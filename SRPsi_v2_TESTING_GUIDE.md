# SRΨ-v2.0 物理测试快速启动指南

**目标**: 验证真实数据训练的 v2.0 模型是否达成 TRAE 设定的指标

---

## 📋 **测试清单**

在 Colab 中完成以下测试：

### **1. Phase 1A: Shift Robustness Test** (约 5-10 分钟)

**测试内容**:
- 空间平移鲁棒性 (s ∈ [0, 64])
- 能量守恒
- 动量守恒

**运行方法**:
```python
# 在 Colab 中运行
!python test_v2_physical.py
```

**预期输出**:
- `outputs/v2_physical_tests_real/physical_metrics_v2_real.json`
- `outputs/v2_physical_tests_real/physical_test_plots_v2_real.png`

---

### **2. Phase 1C: Zero-Shot Extrapolation Test** (约 10-15 分钟)

**测试内容**:
- T=200 零样本外推（训练时只见过 T=48）
- 长期物理守恒性
- 预测稳定性

**运行方法**:
```python
# 在 Colab 中运行
!python test_v2_extrapolation.py
```

**预期输出**:
- `outputs/v2_extrapolation_tests_real/extrapolation_metrics_v2_real.json`
- `outputs/v2_extrapolation_tests_real/extrapolation_plots_v2_real.png`
- `outputs/v2_extrapolation_tests_real/predictions_T200.pt`

---

## 🚀 **推荐方案：使用 Notebook**

**最简单的方式**: 直接运行 `SRPsi_v2_Physical_Tests.ipynb`

### **操作步骤**:

1. **在 Colab 中打开 notebook**:
   ```python
   # 上传 SRPsi_v2_Physical_Tests.ipynb 到 Colab
   ```

2. **按顺序运行所有 cells**:
   - Step 0: 环境检查
   - Step 1: 加载模型
   - Step 2: Phase 1A 测试
   - Step 3: Phase 1C 测试
   - Step 4: 结果对比
   - Step 5: 下载报告

3. **查看结果**:
   - 所有结果会自动可视化
   - 最后会打包下载 `results_v2_real.zip`

---

## 📊 **成功标准**

| 指标 | Transformer | SRΨ v1.0 | **v2.0 目标** |
|------|-------------|----------|---------------|
| **Shift Growth** | 0.97x | 0.74x | **< 0.8x** |
| **Energy Drift** | 0.24 | 1.65 | **< 0.3** |
| **Momentum Drift** | 1.18 | 18.49 | **< 2.0** |

**最低期望**:
- ✅ 至少在 **1 项指标**上超过 Transformer
- ✅ 至少在 **2 项指标**上超过 SRΨ v1.0

**理想目标** (TRAE 的期望):
- ✅ **Momentum Drift < 2.0** (已经从 288K 降到 172！)
- ✅ **Shift Growth < 0.8x** (验证 SpatialOperator 的能力)
- ⚠️  **Energy Drift < 0.3** (可能需要 Phase 3 优化)

---

## 🔍 **关键观察点**

### **Phase 1A 测试中**:

1. **Shift Growth 曲线**:
   - ✅ 理想：平缓上升，SGE < 0.8
   - ⚠️  警告：陡峭上升，SGE > 1.0

2. **Energy Drift vs Shift**:
   - ✅ 理想：稳定在 0.3 以下
   - ⚠️  警告：随 shift 增加而恶化

3. **Momentum Drift vs Shift**:
   - ✅ 理想：稳定在 2.0 以下
   - ⚠️  警告：随 shift 增加而爆炸

---

### **Phase 1C 测试中**:

1. **Energy Evolution (T=0 to T=200)**:
   - ✅ 理想：轻微波动，漂移 < 50%
   - ⚠️  警告：单调上升/下降，漂移 > 100%

2. **Momentum Evolution**:
   - ✅ 理想：稳定在初始值附近
   - ⚠️  警告：发散或振荡

3. **Sample Trajectory**:
   - ✅ 理想：物理结构清晰，无爆炸
   - ⚠️  警告：数值不稳定，出现 NaN 或 Inf

---

## 📁 **输出文件说明**

### **Phase 1A 输出**:

```
outputs/v2_physical_tests_real/
├── physical_metrics_v2_real.json       # 关键指标
└── physical_test_plots_v2_real.png     # 4 张可视化图
```

**JSON 文件包含**:
- `shift_growth`: Shift Growth Error
- `avg_energy_drift`: 平均能量漂移
- `avg_momentum_drift`: 平均动量漂移
- `shift_tests`: 每个 shift 值的详细数据

---

### **Phase 1C 输出**:

```
outputs/v2_extrapolation_tests_real/
├── extrapolation_metrics_v2_real.json  # 关键指标
├── extrapolation_plots_v2_real.png     # 4 张可视化图
└── predictions_T200.pt                 # 完整预测数据
```

**JSON 文件包含**:
- `short_term`: T=0-48 验证结果
- `long_term`: T=0-200 守恒性分析
- `stability`: 稳定性指标
- `trajectories`: 能量和动量的时间序列

---

## ⏱️ **时间估算**

| 测试 | 预计时间 | 说明 |
|------|---------|------|
| **Phase 1A** | 5-10 分钟 | 测试 17 个 shift 值 |
| **Phase 1C** | 10-15 分钟 | 自回归预测 T=200 |
| **总计** | 15-25 分钟 | 约 0.5 小时 |

---

## 🎯 **测试完成后**

1. **下载所有结果**:
   ```python
   !zip -r results_v2_real.zip outputs/
   ```

2. **本地保存 checkpoint**:
   ```python
   from google.colab import files
   files.download('checkpoints/v2_hybrid/checkpoint_best.pt')
   ```

3. **准备 TRAE 分析**:
   - 将 `results_v2_real.zip` 发送给 TRAE
   - 等待 TRAE 的评价和 Phase 3 建议

---

## 🔧 **故障排查**

### **问题 1: CUDA Out of Memory**

**解决方案**:
```python
# 减少测试样本数
NUM_SAMPLES = 50  # 从 100 减少到 50
```

---

### **问题 2: 测试运行太慢**

**解决方案**:
```python
# 减少 shift 测试点
SHIFT_VALUES = list(range(0, 65, 8))  # 从 4 改为 8
```

---

### **问题 3: 找不到 checkpoint**

**检查**:
```bash
!ls -lh checkpoints/v2_hybrid/
```

**如果没有**:
```bash
# 重新上传训练时下载的 checkpoint_best.pt
```

---

## ✅ **准备就绪？**

**在 Colab 中运行以下命令开始**:

```python
# 快速检查
!ls -lh checkpoints/v2_hybrid/checkpoint_best.pt
!ls -lh data/processed/*.pt
!python test_v2_physical.py  # Phase 1A
!python test_v2_extrapolation.py  # Phase 1C
```

---

**🚀 祝 SRΨ-v2.0 在物理测试中大放异彩！**

---
