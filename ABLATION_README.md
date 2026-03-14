# Ablation Study 快速启动指南

## 📋 实验组概览

| 实验组 | 模型 | 关键差异 | 测试假设 |
|--------|------|---------|---------|
| **Exp1** | SRΨ Full | 完整实现（已有结果） | 基线 |
| **Exp2** | SRΨ Real-only | 移除虚部（单通道） | 相位表示的必要性 |
| **Exp3** | SRΨ w/o R | 移除 R 算子 | R 算子的独立贡献 |
| **Exp4** | Conv Baseline | 纯卷积架构 | 卷积偏置的基线水平 |
| **Exp5** | Transformer Rel-PE | 相对位置编码 | 更公平的 Transformer |

## 🚀 快速启动

### 方法 1：一键启动所有实验

```bash
./run_ablation_study.sh
```

这将并行启动 4 个实验（Exp1 已完成）：
- Exp2: SRΨ Real-only
- Exp3: SRΨ w/o R
- Exp4: Conv Baseline
- Exp5: Transformer Rel-PE

### 方法 2：手动启动单个实验

```bash
# Exp2: SRΨ Real-only
venv/bin/python src/train.py \
  --config config/burgers.yaml \
  --model srpsi_real \
  --data data/burgers_1d.npy \
  --output outputs/ablation_study/srpsi_real \
  --epochs 80

# Exp3: SRΨ w/o R
venv/bin/python src/train.py \
  --config config/burgers.yaml \
  --model srpsi_no_r \
  --data data/burgers_1d.npy \
  --output outputs/ablation_study/srpsi_no_r \
  --epochs 80

# Exp4: Conv Baseline
venv/bin/python src/train.py \
  --config config/burgers.yaml \
  --model conv_baseline \
  --data data/burgers_1d.npy \
  --output outputs/ablation_study/conv_baseline \
  --epochs 80

# Exp5: Transformer Rel-PE
venv/bin/python src/train.py \
  --config config/burgers.yaml \
  --model transformer_rel_pe \
  --data data/burgers_1d.npy \
  --output outputs/ablation_study/transformer_rel_pe \
  --epochs 80
```

## 📊 监控训练进度

### 查看所有日志
```bash
tail -f logs/ablation_*.log
```

### 查看单个实验日志
```bash
tail -f logs/ablation_srpsi_real.log
tail -f logs/ablation_srpsi_no_r.log
tail -f logs/ablation_conv_baseline.log
tail -f logs/ablation_transformer_rel_pe.log
```

### 使用 TensorBoard
```bash
tensorboard --logdir outputs/ablation_study
```

## ⏱️ 预期时间

- 单个实验：~2-3 小时（CPU）
- 4 个实验并行：~2-3 小时
- 总时间：**2-3 小时**（不是 8-12 小时，因为并行运行）

## ✅ 训练完成后

### 评估所有模型

```bash
venv/bin/python src/eval.py \
  --config config/burgers.yaml \
  --output_dir outputs/ablation_study \
  --data data/burgers_1d.npy
```

### 查看结果

对比图表将保存在 `outputs/ablation_study/comparison/`

## 🎯 假设检验

### 主要对比

| 对比 | 关键指标 | 预期结果 |
|------|---------|---------|
| **Exp1 vs Exp2** | Shift Robustness | Exp1 << Exp2（相位表示贡献） |
| **Exp1 vs Exp3** | Late MSE, Drift | Exp1 < Exp3（R 算子贡献） |
| **Exp1 vs Exp4** | 所有指标 | Exp1 ≤ Exp4（卷积基线） |
| **Exp1 vs Exp5** | Shift Robustness | Exp1 < Exp5（vs 公平 Transformer） |
| **Exp2 vs Exp4** | 所有指标 | Exp2 < Exp4（实值 vs 卷积） |

## 📁 文件结构

```
srpsi-engine-tiny/
├── outputs/
│   └── ablation_study/
│       ├── srpsi_real/         # Exp2
│       ├── srpsi_no_r/         # Exp3
│       ├── conv_baseline/      # Exp4
│       ├── transformer_rel_pe/ # Exp5
│       └── comparison/         # 对比图表
├── logs/
│   ├── ablation_srpsi_real.log
│   ├── ablation_srpsi_no_r.log
│   ├── ablation_conv_baseline.log
│   └── ablation_transformer_rel_pe.log
└── run_ablation_study.sh       # 启动脚本
```

## 🔍 故障排查

### 问题：找不到模块

```bash
# 确保在项目根目录
cd /Users/luxiangrong/ClaudeCode/my-project/GenCLI+Claude/projects/srpsi-engine-tiny
```

### 问题：CUDA 不可用

这是正常的，脚本会自动回退到 CPU。

### 问题：训练太慢

考虑减少 epoch 数量（修改 `run_ablation_study.sh` 中的 `EPOCHS` 变量）。

## 📝 下一步

1. 等待训练完成（2-3 小时）
2. 运行评估脚本
3. 分析结果
4. 撰写 Ablation Study 结论

---

**准备好启动了吗？运行 `./run_ablation_study.sh` 即可！**
