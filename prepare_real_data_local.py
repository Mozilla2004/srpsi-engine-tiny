"""
Prepare Real Burgers Data (Local)
==================================

在本地运行此脚本，生成 data/processed/ 文件夹，
然后直接上传到 Colab。

Usage:
    python prepare_real_data_local.py
"""

import numpy as np
import torch
from pathlib import Path

print("=" * 70)
print(" " * 20 + "PREPARING REAL BURGERS DATA")
print("=" * 70)
print()

# 1. 加载原始数据
data_path = Path('data/burgers_1d.npy')

if not data_path.exists():
    print(f"❌ Error: {data_path} not found!")
    print(f"   Please ensure data/burgers_1d.npy exists.")
    exit(1)

print(f"📁 Loading original data...")
u_data = np.load(data_path)
print(f"✅ Loaded: {u_data.shape}")
print(f"   Range: [{u_data.min():.3f}, {u_data.max():.3f}]")
print(f"   Mean: {u_data.mean():.3f}, Std: {u_data.std():.3f}")
print()

# 2. 分割数据集
print("🔪 Splitting dataset...")
num_samples = u_data.shape[0]
num_train = int(0.8 * num_samples)
num_val = int(0.1 * num_samples)

print(f"   Train: {num_train} samples")
print(f"   Val: {num_val} samples")
print(f"   Test: {num_samples - num_train - num_val} samples")
print()

# 3. 转换数据格式
print("🔄 Transforming data format...")
tin, tout = 16, 32

# Train
u_train = u_data[:num_train]
train_x = torch.tensor(u_train[:, :tin, :], dtype=torch.float32).transpose(1, 2)
train_y = torch.tensor(u_train[:, tin:tin+tout, :], dtype=torch.float32).transpose(1, 2)

# Val
u_val = u_data[num_train:num_train+num_val]
val_x = torch.tensor(u_val[:, :tin, :], dtype=torch.float32).transpose(1, 2)
val_y = torch.tensor(u_val[:, tin:tin+tout, :], dtype=torch.float32).transpose(1, 2)

# Test
u_test = u_data[num_train+num_val:]
test_x = torch.tensor(u_test[:, :tin, :], dtype=torch.float32).transpose(1, 2)
test_y = torch.tensor(u_test[:, tin:tin+tout, :], dtype=torch.float32).transpose(1, 2)

print(f"✅ Transform complete:")
print(f"   Train X: {train_x.shape} (expected: [{num_train}, 128, 16])")
print(f"   Train Y: {train_y.shape} (expected: [{num_train}, 128, 32])")
print(f"   Val X:   {val_x.shape} (expected: [{num_val}, 128, 16])")
print(f"   Val Y:   {val_y.shape} (expected: [{num_val}, 128, 32])")
print()

# 4. 保存处理后的数据
print("💾 Saving processed data...")
data_dir = Path('data/processed')
data_dir.mkdir(parents=True, exist_ok=True)

torch.save(train_x, data_dir / 'train_x.pt')
torch.save(train_y, data_dir / 'train_y.pt')
torch.save(val_x, data_dir / 'val_x.pt')
torch.save(val_y, data_dir / 'val_y.pt')
torch.save(test_x, data_dir / 'test_x.pt')
torch.save(test_y, data_dir / 'test_y.pt')

print(f"✅ Data saved to {data_dir}/")
print()

# 5. 创建元数据
metadata = {
    'num_samples': num_samples,
    'num_train': num_train,
    'num_val': num_val,
    'num_test': num_samples - num_train - num_val,
    'nx': 128,
    'tin': 16,
    'tout': 32,
    'total_steps': 48,
    'data_range': [float(u_data.min()), float(u_data.max())],
    'data_mean': float(u_data.mean()),
    'data_std': float(u_data.std())
}

import json
with open(data_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"📊 Metadata:")
for key, value in metadata.items():
    print(f"   {key}: {value}")
print()

# 6. 文件大小
print("📁 File sizes:")
for file in sorted(data_dir.glob('*')):
    size_mb = file.stat().st_size / (1024 * 1024)
    print(f"   {file.name}: {size_mb:.2f} MB")
print()

print("=" * 70)
print(" " * 15 + "✅ DATA PREPARATION COMPLETE!")
print("=" * 70)
print()
print("🚀 Next steps:")
print("   1. 在 Colab 中创建 data/processed/ 目录")
print("   2. 上传以下文件到 Colab:")
print("      - data/processed/train_x.pt")
print("      - data/processed/train_y.pt")
print("      - data/processed/val_x.pt")
print("      - data/processed/val_y.pt")
print("   3. 继续运行 Step 6 (Training)")
print()
