# SRΨ-v2.0 "Field-Transformer" Quick Guide

**版本**: 2.0-Hybrid
**生成日期**: 2026-03-15

---

## 🎯 核心理念：分维协同

> **"不同的物理维度需要不同算子"**

```
空间维度 → SRΨ 的 S 算子 (0.74x Shift Growth)
时间维度 → Transformer 的 Attention (0.24x Extrapolation)
物理守恒 → 显式物理损失 (Energy + Momentum)
```

---

## 🏗️ 架构组件

### **1. SpatialOperator (SRΨ's S)**
- 结构化卷积捕捉局部空间关系
- Group Normalization 提供尺度不变性
- **优势**: Shift Growth = 0.74x (最佳)

### **2. TemporalOperator (Transformer's Attention)**
- Multi-head Attention 捕捉全局时间关系
- 相对位置编码允许外推
- **优势**: Extrapolation Ratio = 0.24x (最佳)

### **3. HybridFieldBlock**
- 可学习的融合权重
- 自动平衡空间与时间算子
- **目标**: 结合两者优势

---

## 🔧 物理损失函数

```python
Total_Loss = MSE_Loss
           + λ_energy × Energy_Drift
           + λ_momentum × Momentum_Drift
```

**关键设计**:
- **Energy Conservation**: `dE/dt = 0`
- **Momentum Conservation**: `dP/dt = 0`
- **Adaptive Weighting**: 前 10 epochs warm-up，之后逐渐引入物理约束

---

## 📊 目标指标

| 指标 | Transformer | SRΨ Real | **v2.0 目标** |
|------|-------------|----------|---------------|
| **Extrapolation Ratio** | 0.24x | 0.41x | **< 0.3x** |
| **Shift Growth** | 0.97x | 0.74x | **< 0.8x** |
| **Energy Drift** | 0.24 | 1.65 | **< 0.3** |
| **Momentum Drift** | 1.18 | 18.49 | **< 2.0** |

---

## 🚀 使用指南

### **在 Colab 上训练**

1. **打开**: `SRPsi_v2_Training.ipynb`
2. **选择 Runtime**: A100 GPU
3. **按顺序运行**: Step 0 → Step 8

### **训练时间**: 30-40 分钟 (A100)

---

## 📝 文件清单

- `src/models/srpsi_v2_hybrid.py` - 核心模型
- `src/training/physical_loss.py` - 物理损失
- `config/burgers_v2.yaml` - 配置文件
- `train_v2_hybrid.py` - 训练脚本
- `SRPsi_v2_Training.ipynb` - Colab notebook

---

**详细文档**: 参见 Phase 1A + 1C 报告
