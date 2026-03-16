# EXPERIMENT PHASES - SRΨ-Engine Tiny

**版本**: v0.1
**项目**: SRΨ-Engine Tiny
**更新日期**: 2026-03-16

---

## 实验阶段定义

本项目的实验分为四个主要阶段,每个阶段有明确的目标、标准和交付物。

---

## Phase 1: Setup (搭建) ✅

**时间**: 2026-03-13
**状态**: ✅ 已完成

### 目标
建立项目骨架,实现三个模型架构,搭建训练/评估流程。

### 关键任务
- [x] 实现 SRΨ-Engine Tiny (461 行核心代码)
- [x] 实现 Baseline Transformer
- [x] 实现 Baseline MLP
- [x] 数据生成管道 (Burgers 1D)
- [x] 四组分损失函数
- [x] 四项核心评估指标
- [x] 训练/评估脚本

### 交付物
- `src/models/srpsi_engine_tiny.py`
- `src/train.py`
- `src/losses.py`
- `src/metrics.py`
- `DELIVERY_SUMMARY.md`

### 完成标准
- 三个模型可以训练
- 数据生成成功
- 评估脚本可运行

### 结果
✅ **Milestone 1 达成**: Can Run

---

## Phase 2: Training (训练) ✅

**时间**: 2026-03-13 - 2026-03-14
**状态**: ✅ 已完成

### 目标
训练所有模型,对比性能,验证核心假设。

### 关键任务
- [x] 训练 SRΨ-Engine (80 epochs)
- [x] 训练 Baseline Transformer (80 epochs)
- [x] 四项指标对比测试
- [x] 生成可视化图表
- [x] 撰写实验报告

### 核心发现
- **Rollout MSE**: SRΨ ↓ 6.5%
- **Late Horizon MSE**: SRΨ ↓ 1.7%
- **Energy Drift**: SRΨ ↓ 22.2% ✅
- **Shift Robustness**: SRΨ ↓ 98.3% ✅✅✅

### 交付物
- `outputs/burgers_1d/comparison/` (所有对比图表)
- `EXPERIMENT_REPORT_ZH.md`
- `EXPERIMENT_REPORT_EN.md`

### 完成标准
- 所有模型训练完成
- 四项指标全部对比
- 实验报告撰写完成

### 结果
✅ **Milestone 2 达成**: Can Compare

---

## Phase 3: Analysis (分析) 🔄

**时间**: 2026-03-14 - 2026-03-16
**状态**: 🔄 进行中

### 目标
解耦组件贡献,理解架构优势,建立因果链条。

### 关键任务
- [ ] Ablation Study (5 组对比)
  - [ ] SRΨ Full vs w/o Complex State
  - [ ] SRΨ Full vs w/o Rhythm Operator
  - [ ] SRΨ vs Conv Baseline
  - [ ] SRΨ vs Transformer w/o PE
- [ ] 内部状态检查
  - [ ] ψ 范数分布
  - [ ] 相位门激活
  - [ ] 能量漂移轨迹
- [ ] Shift Robustness 机制分析
- [ ] Energy Drift 机制分析

### TRAE 洞察
> "智能即物理学,智能体现在不变性而非拟合度"

**关键评估维度**:
- Training Loss ← 次要,拟合度
- Shift Robustness ← 核心,不变性
- Energy Drift ← 核心,守恒律
- Field-State Coherence ← 核心,相干性

### 交付物
- `ABLATION_STUDY_REPORT.md`
- `TRAE_INSIGHTS.md` (已完成)
- 内部状态可视化

### 完成标准
- 5 组 ablation 实验完成
- 因果链条建立
- 架构优势解释清楚

### 结果
🔄 **Milestone 3 进行中**: Can Explain (60% 完成)

---

## Phase 4: Validation (验证) 🔄

**时间**: 2026-03-15 - 2026-03-17
**状态**: 🔄 进行中

### 目标
验证物理外推能力,测试分布外泛化,准备开源发布。

### 关键任务
- [ ] Phase 1C Zero-Shot Test ✅
  - [x] 不同粘性系数 ν
  - [x] 不同时间步长 dt
  - [x] 不同初始条件
- [ ] Physical Tests v2.0 (进行中)
  - [ ] 真实数据测试
  - [ ] 外推比例验证 (目标 < 0.24x)
  - [ ] 物理一致性检查
- [ ] v2.0 Hybrid 架构测试
  - [ ] SRΨ + Attention 性能
  - [ ] 与 v1.0 对比
  - [ ] Hybrid 架构报告

### 核心假设
**SRΨ-Engine 能从"统计插值"进化到"物理推演"**

### 交付物
- `SRPSI_Phase_1C_Zero_Shot_Report.md` ✅
- `test_v2_physical.py`
- `test_v2_extrapolation.py`
- `v2.0_EXPERIMENT_REPORT.md`

### 完成标准
- 真实数据测试完成
- 外推比例 < 0.24x
- v2.0 架构验证完成

### 结果
🔄 **进行中**: Physical Tests v2.0

---

## Phase 5: Synthesis (综合) ⏳

**时间**: 2026-03-17 - 2026-03-20
**状态**: ⏳ 待开始

### 目标
综合所有实验结果,撰写完整报告,准备开源发布。

### 关键任务
- [ ] 撰写 v2.0 完整实验报告
  - [ ] Hybrid 架构性能
  - [ ] 物理外推分析
  - [ ] 与 v1.0 对比
  - [ ] TRAE 洞察整合
- [ ] 准备开源发布
  - [ ] 代码整理与注释
  - [ ] 文档完善 (README, CONTRIBUTING)
  - [ ] 示例脚本
  - [ ] Colab Notebook
- [ ] 录像 / 演示准备

### 交付物
- `SRPSI_v2.0_FINAL_REPORT.md`
- `README.md` (更新)
- `examples/` (示例脚本)
- Colab Notebook

### 完成标准
- 完整实验报告撰写完成
- 代码开源就绪
- 示例可复现

---

## 关键转折点

### T1: 从"拟合度"到"不变性" (2026-03-14)
**Entry 0004**: TRAE 的洞察重新定义了智能评估标准

**范式转换**:
```
v1.0-Raw:      数据拟合视角
v2.0-Field:    物理重构视角
```

### T2: 从"纯 SRΨ"到"Hybrid" (2026-03-15)
**Entry 0006**: 架构升级,结合物理约束与长程关联

**架构演进**:
```
v1.0: 纯 SRΨ (Complex-valued + S/R/N/Φ)
v2.0: Hybrid (SRΨ + Attention + Interaction)
```

### T3: 从"物理模拟"到"场自治" (2026-03-16)
**Entry 0007**: 同步 spirit-coding-lab 后的认知升级

**理念升级**:
> "模型不再是被动地逼近目标值,而是通过其内部的物理约束层,
>  主动维持场状态的逻辑自洽"

---

## 实验假设验证状态

| 假设 | 描述 | 状态 | 证据强度 |
|------|------|------|---------|
| **H1: 长期稳定性** | SRΨ 的稳定投影器能有效控制误差累积 | ✅ 验证通过 | 强 |
| **H2: 守恒律控制** | Complex-valued State 能更好地编码物理守恒律 | ✅ 验证通过 | 强 |
| **H3: 平移鲁棒性** | 相位表示 + 卷积算子提供平移不变性 | ✅✅ 压倒性验证 | 极强 |
| **H4: 扰动恢复** | 节律算子增强系统的动态平衡能力 | 🔲 测试中 | - |

---

## Milestone 完成度

```
Milestone 1: Can Run        [████████████████████] 100% ✅
Milestone 2: Can Compare    [████████████████████] 100% ✅
Milestone 3: Can Explain    [██████████░░░░░░░░░░]  60% 🔄
```

---

## 下一步行动

### 立即 (今天)
1. 完成 Physical Tests v2.0 的真实数据测试
2. 验证 Extrapolation Ratio < 0.24x

### 短期 (本周)
3. 对比 v1.0 vs v2.0 性能
4. 撰写 v2.0 实验报告

### 中期 (下周)
5. 准备开源发布
6. 创建示例脚本和 Colab Notebook

---

**维护说明**:
- 每完成一个任务,更新对应的状态 `[ ]` → `[x]`
- 每个 Phase 完成后,更新进度条
- 关键发现及时记录到 EXPERIMENT_LOG.md
- 状态变化同步更新 EXPERIMENT_STATE.json
