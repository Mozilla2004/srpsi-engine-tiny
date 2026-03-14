# Ablation Study 实验计划
# 目标：解耦 SRΨ 各组件的贡献

## 实验设计

### 对照组设置

| 实验组 | 架构 | 关键差异 | 测试假设 |
|--------|------|---------|---------|
| **Exp1: SRΨ Full** | S+R+N+Φ, Complex | 完整实现 | 基线 |
| **Exp2: Real-only** | S+R+N+Φ, Real | 移除虚部通道 | 相位表示的必要性 |
| **Exp3: No Rhythm** | S+N+Φ, Complex | 移除 R 算子 | R 的独立贡献 |
| **Exp4: Conv Baseline** | Conv+MLP, Real | 无相位、无 R | 卷积偏置基线 |
| **Exp5: Rel-Attn** | Transformer, Rel PE | 移除绝对位置 | 公平 Transformer |

### 实现路径

#### Exp2: Real-only SRΨ
```python
# 修改 InputEncoder
class InputEncoderReal(nn.Module):
    def __init__(self, tin, nx, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tin, hidden_dim),  # 单通道
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # ... normalize ...
        z = self.net(x)
        z = torch.clamp(z, -10, 10)
        # 返回 [B, X, D] 而非 [B, X, D, 2]
        return z
```

#### Exp3: SRΨ w/o R
```python
# 修改 SRPsiBlock
class SRPsiBlockNoR(nn.Module):
    def __init__(self, hidden_dim, kernel_size, dt):
        super().__init__()
        self.S = StructureOperatorS(hidden_dim, kernel_size)
        # 移除 self.R
        self.N = NonlinearOperatorN(hidden_dim)
        self.P = StableProjector(hidden_dim)
        self.dt = dt

    def forward(self, psi):
        delta = self.S(psi) + self.N(psi)  # 无 R
        # ... rest same ...
```

#### Exp4: Conv Baseline
```python
# 简化架构
class ConvBaseline(nn.Module):
    def __init__(self, tin, nx, hidden_dim, depth, kernel_size):
        super().__init__()
        self.encoder = nn.Linear(tin, hidden_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
            for _ in range(depth)
        ])
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2))  # [B, D, X]
        for conv in self.convs:
            h = F.gelu(conv(h))
        h = h.transpose(1, 2)  # [B, X, D]
        return self.decoder(h).unsqueeze(1).repeat(1, 32, 1)
```

#### Exp5: Transformer w/o Absolute PE
```python
# 修改 BaselineTransformer
class BaselineTransformerRelPE(nn.Module):
    def __init__(self, tin, tout, nx, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.tin = tin
        self.tout = tout
        self.nx = nx

        # 移除绝对位置编码
        self.input_projection = nn.Linear(tin, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 使用相对位置注意力（需要自定义）
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.shape[0]
        h = self.input_projection(x)  # [B, Tin, D]

        # Autoregressive rollout
        preds = []
        for _ in range(self.tout):
            h = self.transformer(h)[:, -1:, :]  # 取最后一步
            out = self.output_projection(h).squeeze(-1)  # [B, X]
            preds.append(out)
            # 更新 h（移位）
            h = torch.cat([h[:, 1:, :], out.unsqueeze(1)], dim=1)

        return torch.stack(preds, dim=1)
```

### 训练配置

所有实验使用相同的超参数：
- Epochs: 80（快速收敛）
- Batch size: 32
- Learning rate: 0.0001
- Loss: 仅 prediction loss（移除 conservation/smoothness 以公平对比）

### 评估指标

核心对比：
1. **Rollout MSE** - 整体预测精度
2. **Late Horizon MSE** - 长期稳定性
3. **Energy Drift** - 守恒律控制
4. **Shift Robustness** - 平移不变性

### 预期结果

假设检验矩阵：

| 对比 | Shift Robustness | Rollout MSE | Late MSE | Drift |
|------|-----------------|-------------|----------|-------|
| Exp1 vs Exp2 | Exp1 << Exp2 | ≈ | ≈ | ≈ |
| Exp1 vs Exp3 | Exp1 < Exp3 | ≈ | Exp1 < Exp3 | ≈ |
| Exp1 vs Exp4 | Exp1 << Exp4 | ≈ | ≈ | ≈ |
| Exp1 vs Exp5 | Exp1 < Exp5 | ≈ | ≈ | ≈ |
| Exp2 vs Exp4 | Exp2 < Exp4 | ≈ | ≈ | ≈ |

### 成功标准

**强结论**：
- Exp1 在 Shift Robustness 上显著优于所有对照组
- Exp3 vs Exp2 显示 R 的独立贡献
- Exp2 vs Exp4 显示相位表示的优势

**弱结论**：
- Exp1 仅在部分指标上优于对照组
- 对照组之间差异不显著

### 时间估算

- 单个实验训练：~2-3 小时（CPU）
- 5 个实验总时间：~10-15 小时
- 可并行运行（如果有多台机器）

### 下一步

1. 实现 Exp2-5 的模型代码
2. 创建训练脚本（复用 train.py，修改 --model 参数）
3. 启动并行训练
4. 生成对比报告（热图、柱状图）
5. 撰写 Ablation Study 结论

---

## 输出格式

最终报告将包括：
- 各模型的性能对比表
- Shift Robustness 对比图（关键）
- 组件贡献度分析
- 因果链条验证结论
- 对未来工作的启示
