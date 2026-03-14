# SRΨ-Engine Tiny: 项目交付总结

**生成时间**: 2026-03-13
**状态**: ✅ Milestone 1 完成 - 项目骨架已就绪
**版本**: v0.1.0

---

## 核心目标

验证 **SRΨ-Engine**（带有结构-节律-稳定投影归纳偏置的动力学算子）是否能在 1D 场演化预测任务上提升：
1. 长期 rollout 稳定性
2. 守恒控制
3. 位移扰动鲁棒性
4. 小噪声恢复能力

---

## 交付文件清单

### 1. 项目结构 ✅

```
srpsi-engine-tiny/
├── config/              # 配置文件
│   ├── default.yaml     # 默认配置
│   └── burgers.yaml     # Burgers 方程配置
├── data/                # 数据目录
├── outputs/             # 输出目录
├── src/
│   ├── data_gen.py      # 数据生成 (Burgers 1D)
│   ├── datasets.py      # PyTorch Dataset
│   ├── utils.py         # 工具函数
│   ├── losses.py        # 损失函数
│   ├── metrics.py       # 评估指标
│   ├── train.py         # 训练脚本
│   ├── eval.py          # 评估脚本
│   ├── plot.py          # 可视化
│   └── models/
│       ├── baseline_mlp.py
│       ├── baseline_transformer.py
│       └── srpsi_engine_tiny.py
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_ablation.sh
├── requirements.txt
└── README.md
```

### 2. 核心实现 ✅

#### SRΨ-Engine Tiny 完整实现 (461 行)

**关键组件**:
- ✅ `InputEncoder`: 历史场 → 复值状态 ψ₀ [B, X, D, 2]
- ✅ `StructureOperatorS`: 局部空间耦合（卷积）
- ✅ `RhythmOperatorR`: 相位旋转动力学（相位门控）
- ✅ `NonlinearOperatorN`: 非线性调制
- ✅ `StableProjector`: 稳定投影（范数控制）
- ✅ `SRPsiBlock`: S+R+N+Φ 组合
- ✅ `OutputDecoder`: ψ → 实场
- ✅ `SRPsiEngineTiny`: 完整模型

**设计特点**:
```python
# 复值状态表示
psi.shape = [B, X, D, 2]  # [batch, space, hidden, real/imag]

# 单步演化
Δψ = S(ψ) + R(ψ) + N(ψ)
ψ_{t+1} = ψ_t + dt · Δψ
ψ_{t+1} = Φ(ψ_{t+1})
```

#### 三模型对比实现

| 模型 | 文件 | 参数量 | 特点 |
|------|------|--------|------|
| Baseline MLP | `baseline_mlp.py` | ~200K | Flatten → FC → Reshape |
| Baseline Transformer | `baseline_transformer.py` | ~300K | 时间帧作为 token |
| SRΨ-Engine Tiny | `srpsi_engine_tiny.py` | ~150K | 复值动力学算子 |

### 3. 损失函数 ✅

**四组分损失** (252 行):
```python
L_total = L_pred
    + λ_cons · L_conservation      # 能量守恒
    + λ_phase · L_shift_consistency # 位移一致性
    + λ_smooth · L_smoothness       # 时间平滑
```

### 4. 评估指标 ✅

**四项核心指标** (234 行):
1. **Rollout MSE**: 整体预测误差
2. **Late-Horizon MSE**: 后半段误差（长期稳定性）
3. **Energy Drift**: 能量漂移
4. **Shift Robustness**: 位移等变误差

### 5. 训练与评估 ✅

**训练脚本** (400 行):
- 支持 `--model` 选择模型类型
- TensorBoard 日志
- Checkpoint 自动保存
- 多损失组件跟踪

**评估脚本**:
- 加载所有三个模型
- 生成对比图表
- 输出性能汇总表

### 6. 可视化 ✅

**四类对比图**:
1. Truth vs Prediction（多时间点对比）
2. Temporal Error Growth（误差增长曲线）
3. Energy Drift Trajectories（能量漂移）
4. Model Comparison Bar Charts（性能对比）

---

## 快速开始

### 环境安装

```bash
cd srpsi-engine-tiny
pip install -r requirements.txt
```

### 生成数据

```bash
python src/data_gen.py \
    --task burgers_1d \
    --num_samples 4800 \
    --total_steps 48 \
    --nx 128 \
    --output data/burgers_1d.npy
```

### 训练模型

**方式 1: 批量训练所有模型**
```bash
bash scripts/run_train.sh
```

**方式 2: 单独训练**
```bash
# SRΨ-Engine Tiny
python src/train.py \
    --config config/burgers.yaml \
    --model srpsi_engine \
    --output outputs/burgers_1d/srpsi_engine
```

### 评估对比

```bash
bash scripts/run_eval.sh
```

---

## 代码质量

### 统计数据

| 模块 | 行数 | 功能 |
|------|------|------|
| `srpsi_engine_tiny.py` | 461 | SRΨ 核心实现 |
| `train.py` | 400 | 训练循环 |
| `losses.py` | 252 | 损失函数 |
| `metrics.py` | 234 | 评估指标 |
| **总计** | **~2000+** | **核心代码** |

### 代码特性

- ✅ 模块化设计
- ✅ 完整文档字符串
- ✅ 类型提示
- ✅ 可测试（每个模块有 `if __name__ == "__main__"` 测试）
- ✅ 可配置（YAML 配置文件）
- ✅ 可扩展（易于添加新算子/损失/指标）

---

## Milestone 1 状态

### ✅ 已完成

- [x] 项目骨架搭建
- [x] 三个模型实现
- [x] 数据生成（Burgers 1D）
- [x] 训练/评估脚本
- [x] 损失函数（四组分）
- [x] 评估指标（四项）
- [x] 可视化工具
- [x] 运行脚本
- [x] README 文档

### 🔄 Milestone 2 待完成

- [ ] 统一评估脚本集成测试
- [ ] 四类对比图生成验证
- [ ] 模型性能对比分析

### 📋 Milestone 3 待完成

- [ ] 内部状态检查（ψ 范数、相位门分布）
- [ ] 消融实验（S/R/N/Φ 贡献分析）
- [ ] 架构解释（"为什么更稳"）

---

## 预期结果

### 我们期望看到的格局

| 模型 | 短期拟合 | 长期稳定性 | 能量漂移 | 位移鲁棒性 |
|------|---------|-----------|---------|-----------|
| MLP | 良好 | 差 | 高 | 差 |
| Transformer | 优秀 | 中等 | 中等 | 中等 |
| **SRΨ** | 良好 | **最佳** | **最低** | **最佳** |

### 关键问题

**SRΨ-Engine 能否在不牺牲准确性的前提下实现更好的稳定性？**

---

## 技术亮点

### 1. 复值状态表示

```python
# 传统方法
u.shape = [B, X]  # 实值场

# SRΨ-Engine
psi.shape = [B, X, D, 2]  # 复值场（双通道实值近似）
```

**优势**:
- 自然编码相位信息
- 支持旋转动力学
- 更丰富的表达能力

### 2. 算子分解

```python
Δψ = S(ψ) + R(ψ) + N(ψ)
```

- **S (Structure)**: 局部空间耦合 → 编码空间结构
- **R (Rhythm)**: 相位旋转 → 编码振荡/节律
- **N (Nonlinear)**: 非线性调制 → 编码非线性交互

### 3. 稳定投影

```python
ψ ← ψ / max(1.0, ||ψ||)
```

**作用**: 防止数值爆炸，控制能量漂移

### 4. 物理约束损失

```python
L_conservation: E_pred ≈ E_true
L_shift_consistency: model(shift(x)) ≈ shift(model(x))
```

---

## 下一步行动

### 立即可做

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

2. **测试数据生成**:
   ```bash
   python src/data_gen.py --task burgers_1d --num_samples 100 --output test_data.npy
   ```

3. **小规模训练测试**:
   ```bash
   # 修改 config/burgers.yaml: samples_train=100, epochs=2
   python src/train.py --config config/burgers.yaml --model srpsi_engine
   ```

### 完整实验流程

1. 生成完整数据集（4800 samples）
2. 训练三个模型（80 epochs）
3. 运行评估脚本
4. 生成对比图表
5. 分析结果，撰写实验报告

---

## 审计清单

### 核心文件

- [x] `srpsi_engine_tiny.py` - SRΨ 核心实现
- [x] `train.py` - 训练脚本
- [x] `losses.py` - 损失函数
- [x] `metrics.py` - 评估指标

### 关键组件

- [x] InputEncoder ✅
- [x] StructureOperatorS ✅
- [x] RhythmOperatorR ✅
- [x] NonlinearOperatorN ✅
- [x] StableProjector ✅
- [x] SRPsiBlock ✅
- [x] OutputDecoder ✅

### 文档

- [x] README.md ✅
- [x] 代码内文档字符串 ✅
- [x] 本交付总结 ✅

---

## 交付状态

| 项目 | 状态 |
|------|------|
| 代码实现 | ✅ 完成 |
| 文档 | ✅ 完成 |
| 测试 | ⚠️ 需运行验证 |
| 训练 | ⚠️ 待执行 |
| 结果分析 | ⏳ 待训练后完成 |

**整体状态**: ✅ **Milestone 1 达成 - 项目骨架完整，可开始实验**

---

**作者**: Claude Code (陆队版)
**项目**: SRΨ-Engine Tiny v0.1.0
**交付日期**: 2026-03-13

---

## 附录：关键代码片段

### SRΨ Block 核心逻辑

```python
class SRPsiBlock(nn.Module):
    def forward(self, psi):
        # 计算三个算子的贡献
        delta = self.S(psi) + self.R(psi) + self.N(psi)

        # 欧拉积分
        psi_next = psi + self.dt * delta

        # 稳定投影
        psi_next = self.P(psi_next)

        return psi_next
```

### 自回归 Rollout

```python
def forward(self, x):
    psi = self.encoder(x)  # 编码初始状态
    preds = []

    for _ in range(self.tout):
        psi = self.step(psi)  # 每步应用 K 个 SRΨ blocks
        y = self.decoder(psi)
        preds.append(y)

    return torch.stack(preds, dim=1)  # [B, Tout, X]
```

### 能量守恒损失

```python
def conservation_loss(pred, target):
    e_pred = (pred ** 2).sum(dim=-1)  # [B, T]
    e_true = (target ** 2).sum(dim=-1)
    return F.mse_loss(e_pred, e_true)
```

---

**准备好了吗？让我们开始验证 SRΨ 理论！**
