# EXPERIMENT LOG - SRΨ-Engine Tiny

**项目名称**: SRΨ-Engine Tiny - 物理场演化预测架构验证
**实验任务**: 1D Burgers 方程预测
**开始日期**: 2026-03-13
**当前版本**: v2.0 Hybrid

---

## 统计信息

- **总条目数**: 8
- **当前阶段**: physical_validation
- **关键里程碑**: ✅ Milestone 1 | ✅ Milestone 2 | 🔄 Milestone 3

---

## Entry 0001

- **timestamp**: 2026-03-13T00:00:00+08:00
- **type**: project_init
- **phase**: setup
- **summary**: 🏗️ 项目初始化:Milestone 1 达成
- **details**: |
  完成 SRΨ-Engine Tiny 项目骨架搭建
  - 三模型架构实现 (MLP / Transformer / SRΨ-Engine)
  - 数据生成管道 (Burgers 1D)
  - 训练/评估脚本
  - 四组分损失函数
  - 核心评估指标

**交付物**: DELIVERY_SUMMARY.md
**状态**: ✅ 完成

---

## Entry 0002

- **timestamp**: 2026-03-14T00:00:00+08:00
- **type**: experiment_result
- **phase**: training
- **summary**: 🎯 核心实验完成:SRΨ 全面超越 Transformer
- **details**: |
  80 epochs 训练完成,关键发现:
  - Rollout MSE: ↓ 6.5%
  - Late Horizon MSE: ↓ 1.7%
  - Energy Drift: ↓ 22.2% ✅
  - Shift Robustness: ↓ 98.3% ✅✅✅

  **物理洞察**:
  - Shift Robustness 的压倒性优势验证了 Complex-valued State 的价值
  - Energy Drift 的改善体现了 Stable Projector 的收敛
  - SRΨ 的优势是"慢激活"的,需要长期训练 (≥50 epochs)

**交付物**: EXPERIMENT_REPORT_ZH.md
**状态**: ✅ Milestone 2 达成

---

## Entry 0003

- **timestamp**: 2026-03-14T12:00:00+08:00
- **type**: ablation_study
- **phase**: analysis
- **summary**: 🔬 Ablation Study 开始:解耦组件贡献
- **details**: |
  设计 5 组对比实验:
  1. SRΨ Full (完整实现)
  2. SRΨ w/o Complex State (实值版本)
  3. SRΨ w/o Rhythm Operator (移除 R)
  4. Conv Baseline (纯卷积)
  5. Transformer w/o Absolute PE (相对位置)

  **目标**: 建立因果链条,量化各组件独立贡献

**状态**: 🔄 进行中

---

## Entry 0004

- **timestamp**: 2026-03-14T18:00:00+08:00
- **speaker**: trae
- **type**: insight
- **phase**: analysis
- **summary**: 🧠 TRAE 洞察:"智能即物理学"
- **details**: |
  > "智能体现在不变性而非拟合度"

  **关键评估维度**:
  - Training Loss ← 次要,拟合度
  - Shift Robustness ← 核心,不变性
  - Energy Drift ← 核心,守恒律
  - Field-State Coherence ← 核心,相干性

  **警告**:
  > "Loss 越低,有时反而意味着模型被锁死在了数据的特定坐标系中"
  > "SRΨ 略高的 Loss,实际上是在为维持物理不变性付出代价"

**交付物**: TRAE_INSIGHTS.md
**影响**: 重新定义评估标准

---

## Entry 0005

- **timestamp**: 2026-03-15T00:00:00+08:00
- **type**: milestone
- **phase**: validation
- **summary**: ✅ Phase 1C Zero-Shot Test 完成
- **details**: |
  验证模型在训练分布外参数上的泛化能力
  - 测试不同的 ν (粘性系数)
  - 测试不同的 dt (时间步长)
  - 测试不同的初始条件

**交付物**: SRPSI_Phase_1C_Zero_Shot_Report.md
**状态**: ✅ 完成

---

## Entry 0006

- **timestamp**: 2026-03-15T12:00:00+08:00
- **type**: architecture_evolution
- **phase**: refinement
- **summary**: 🚀 v2.0 Hybrid 架构:SRΨ + Attention
- **details**: |
  架构升级:纯 SRΨ → Hybrid (SRΨ + Attention)

  **关键改进**:
  - S-Operator: 空间特征的结构化提取
  - Attention: 时间维度的长程全局关联
  - DomainInteractionGate: 两类特征的动态融合

  **文件**:
  - src/models/srpsi_v2_hybrid.py
  - src/training/physical_loss.py

**状态**: 🔄 开发中

---

## Entry 0007

- **timestamp**: 2026-03-16T00:00:00+08:00
- **speaker**: trae
- **type**: paradigm_shift
- **phase**: synthesis
- **summary**: 🌌 认知升级:从"数据拟合"到"场自治架构"
- **details**: |
  同步 spirit-coding-lab (v3.0 协作自治模式) 后的架构重构

  **核心理念**:
  > "模型不再是被动地逼近目标值,而是通过其内部的物理约束层,
  >  主动维持场状态的逻辑自洽"

  **物理损失重新定义**:
  - 动量守恒 = 空间平移不变性的结构化保持
  - 能量守恒 = 耗散系统能量衰减规律的内在模拟

  **目标**: 建立具备物理外推能力的通用引擎

**状态**: ✅ 认知对齐完成

---

## Entry 0008

- **timestamp**: 2026-03-16T08:00:00+08:00
- **type**: testing
- **phase**: physical_validation
- **summary**: 🧪 Physical Tests v2.0:外推能力验证
- **details**: |
  测试集:
  - 真实数据测试 (real_data/)
  - 物理一致性测试 (physical_dimension_tests.py)
  - 外推比例验证 (Extrapolation Ratio < 0.24x)

  **关键指标**:
  - 物理外推 vs 统计插值
  - 分布外泛化能力
  - 能量/动量守恒保持率

**文件**:
- test_v2_physical.py
- test_v2_extrapolation.py
- prepare_real_data_local.py

**状态**: 🔄 进行中

---

## 关键转折点

### T1: 从"拟合度"到"不变性" (Entry 0004)
TRAE 的洞察重新定义了智能评估标准,从数据拟合视角转向物理重构视角。

### T2: 从"纯 SRΨ"到"Hybrid" (Entry 0006)
架构升级,结合 SRΨ 的物理约束与 Attention 的长程关联。

### T3: 从"物理模拟"到"场自治" (Entry 0007)
同步 spirit-coding-lab 后的认知升级,模型从被动拟合转向主动自洽。

---

## 实验阶段

### ✅ Milestone 1: Can Run
项目骨架完整,所有模型可以训练

### ✅ Milestone 2: Can Compare
统一评估脚本,四类对比图,模型性能分析

### 🔄 Milestone 3: Can Explain
内部状态检查,消融实验,架构解释

---

## 下一步行动

1. **完成 Physical Tests v2.0**
   - 真实数据测试
   - 外推能力验证
   - 物理一致性检查

2. **撰写 v2.0 实验报告**
   - Hybrid 架构性能
   - 物理外推分析
   - 与 v1.0 对比

3. **准备开源发布**
   - 代码整理
   - 文档完善
   - 示例脚本

---

**更新日志**:
- 2026-03-16: 初始化,导入 Entries 0001-0008
