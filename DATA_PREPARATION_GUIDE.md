# 真实数据准备指南

**目标**: 准备真实的 Burgers 数据用于 SRΨ-v2.0 训练

---

## 📁 **现有数据**

已发现完整数据集：
```
data/burgers_1d.npy (117 MB)
- Shape: (4800, 48, 128)
- Format: [num_samples, total_steps, nx]
- Range: [-10, 10]
- Mean: 0.046, Std: 0.806
```

**完全符合要求！** ✅

---

## 🚀 **两种准备方式**

### **方式 A: 在本地准备（推荐）**

```bash
# 1. 进入项目目录
cd srpsi-engine-tiny

# 2. 运行数据准备脚本
python3 -c "
import numpy as np
import torch
from pathlib import Path

# Load data
u_data = np.load('data/burgers_1d.npy')
print(f'Loaded: {u_data.shape}')

# Split
num_train = int(0.8 * 4800)
num_val = int(0.1 * 4800)

u_train = u_data[:num_train]
u_val = u_data[num_train:num_train+num_val]
u_test = u_data[num_train+num_val:]

# Transform (关键: transpose!)
tin, tout = 16, 32
train_x = torch.tensor(u_train[:, :tin, :]).transpose(1, 2)
train_y = torch.tensor(u_train[:, tin:tin+tout, :]).transpose(1, 2)

val_x = torch.tensor(u_val[:, :tin, :]).transpose(1, 2)
val_y = torch.tensor(u_val[:, tin:tin+tout, :]).transpose(1, 2)

# Save
data_dir = Path('data/processed')
data_dir.mkdir(parents=True, exist_ok=True)

torch.save(train_x, data_dir / 'train_x.pt')
torch.save(train_y, data_dir / 'train_y.pt')
torch.save(val_x, data_dir / 'val_x.pt')
torch.save(val_y, data_dir / 'val_y.pt')

print('✅ Data saved to data/processed/')
print(f'Train X: {train_x.shape}')
print(f'Train Y: {train_y.shape}')
"
```

---

### **方式 B: 在 Colab 中准备**

1. **打开 `Prepare_Real_Data.ipynb`**
2. **按顺序运行所有 cells**
3. **下载 `data/processed/` 文件夹**

---

## 📊 **准备后的数据格式**

```
data/processed/
├── train_x.pt  [3840, 128, 16]  - 输入 (前 16 步)
├── train_y.pt  [3840, 128, 32]  - 输出 (后 32 步)
├── val_x.pt    [480, 128, 16]
├── val_y.pt    [480, 128, 32]
└── metadata.json
```

**关键**: 维度顺序是 `[batch, nx, tin]`（已 transpose）

---

## 🎯 **下一步**

数据准备完成后：

1. **创建新的训练 notebook**（使用真实数据）
2. **修改 Step 5 的数据加载**：
   ```python
   # 加载预处理的数据
   train_x = torch.load('data/processed/train_x.pt')
   train_y = torch.load('data/processed/train_y.pt')
   # ...
   ```

3. **开始训练！**

---

## ⏱️ **预期训练时间**

- **Dummy data**: 2-3 分钟 (80 epochs)
- **Real data**: 10-15 分钟 (80 epochs on H100)
- **原因**: 真实数据计算量更大（物理演化更复杂）

---

## 🎓 **数据说明**

### **Burgers 方程**

```
∂u/∂t + u·∂u/∂x = ν·∂²u/∂²x
```

- **雷诺数**: R = 10
- **粘度**: ν = 0.01
- **边界条件**: 周期边界
- **初始条件**: 随机初始化

### **物理特性**

- **能量守恒**: E = 0.5·∫u²dx ≈ constant
- **动量守恒**: P = ∫u·dx ≈ constant
- **激波形成**: 高雷诺数下的非线性效应

---

## ✅ **验证清单**

数据准备完成后检查：

- [ ] train_x.shape = (3840, 128, 16)
- [ ] train_y.shape = (3840, 128, 32)
- [ ] 所有文件都在 `data/processed/` 目录
- [ ] 可以成功加载 torch.load()

---

**准备开始！** 🚀
