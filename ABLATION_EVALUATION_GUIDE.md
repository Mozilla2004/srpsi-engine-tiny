# Ablation Study Evaluation Guide

## 等待训练完成

目前 4 个 ablation 实验正在并行训练中，预计需要 2-3 小时。

你可以用以下命令检查进度：

```bash
# 查看所有实验的最新日志
tail -20 logs/ablation_*.log

# 检查是否完成（完成后会显示 "Training complete"）
grep -E "(Epoch 80|Training complete)" logs/ablation_*.log
```

## 训练完成后的步骤

### 1. 运行评估脚本

训练完成后，运行：

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 运行评估（会自动加载所有模型并测试）
python src/evaluate_ablation.py --config config/burgers.yaml
```

这会：
- 加载所有 6 个模型（SRΨ Full + 4 ablation + Transformer baseline）
- 在测试集上评估所有模型
- 生成对比表格
- 保存结果到 `results/ablation/ablation_results.json`

### 2. 生成可视化报告

评估完成后，运行：

```bash
python scripts/analyze_ablation_results.py
```

这会生成：
- 4 个对比柱状图（每个指标一个）
- 1 个雷达图（多指标对比）
- 1 个 Markdown 分析报告

文件保存在 `results/ablation/` 目录。

## 预期结果

### 主要对比指标

| 指标 | 说明 | 越低越好 |
|------|------|----------|
| **Rollout MSE** | 整个预测时域的平均误差 | ✅ |
| **Late Horizon MSE** | 后半段预测误差（稳定性测试） | ✅ |
| **Energy Drift** | 能量漂移（守恒性测试） | ✅ |
| **Shift Robustness** | 平移等变误差（平移不变性测试） | ✅ |

### 关键假设验证

1. **复值表示的贡献**：SRΨ Real vs SRΨ Full
   - 如果 SRΨ Full 更好 → 复值表示重要
   - 特别关注 Shift Robustness 的差异

2. **R 算子的贡献**：SRΨ w/o R vs SRΨ Full
   - 如果 SRΨ Full 更好 → R 算子提供稳定性
   - 特别关注 Energy Drift 的差异

3. **架构优势**：Conv Baseline vs SRΨ Full
   - 验证 SRΨ 是否真的比简单卷积更好
   - 关注所有指标

4. **位置编码影响**：Transformer Rel-PE vs Transformer Baseline
   - 相对位置是否真的有帮助
   - 与 SRΨ 对比

## 输出文件

完成后的文件结构：

```
results/ablation/
├── ablation_results.json        # 原始数据
├── ablation_rollup_mse.png      # Rollout MSE 对比图
├── ablation_late_mse.png        # Late Horizon MSE 对比图
├── ablation_energy_drift.png    # Energy Drift 对比图
├── ablation_shift_robustness.png # Shift Robustness 对比图
├── ablation_radar_chart.png     # 多指标雷达图
└── ABLATION_RESULTS.md          # 分析报告
```

## 快速检查命令

```bash
# 一键检查所有实验状态
for log in logs/ablation_*.log; do
    echo "=== $(basename $log) ==="
    tail -3 "$log" | grep -E "(Epoch|loss|Training complete)"
    echo ""
done

# 统计完成的 epoch 数
for log in logs/ablation_*.log; do
    echo "$(basename $log): $(grep -c "Epoch [0-9]*/80" "$log")/80 epochs"
done
```

---

**Note**: 所有脚本都已经准备好了，只需要等待 TRAE 的训练任务完成！
