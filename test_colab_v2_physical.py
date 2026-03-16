"""
SRΨ-v2.0 Physical Tests (Colab-trained Model)
=============================================

Test the model trained in Colab on Burgers 1D data:
- Energy Conservation
- Momentum Conservation
- Prediction Quality

Author: Claude Code + TRAE
Version: 1.0-Colab-Test
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, '/content/srpsi-engine-tiny')

from src.models.srpsi_v2_hybrid import create_srpsi_v2_model
from src.datasets import create_dataloaders

# ===============================================================
#                    CONFIGURATION
# ===============================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = '/content/drive/MyDrive/srpsi-engine-tiny/colab_runs/run_2026-03-16-v4/checkpoints/checkpoint_best.pt'
DATA_PATH = 'data/burgers_1d.npy'
OUTPUT_DIR = Path('/content/drive/MyDrive/srpsi-engine-tiny/colab_runs/run_2026-03-16-v4/physical_tests')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(" " * 20 + "SRΨ-v2.0 PHYSICAL TESTS")
print("=" * 70)
print(f"📁 Checkpoint: {CHECKPOINT_PATH}")
print(f"📊 Device: {DEVICE}")
print(f"📁 Output: {OUTPUT_DIR}")
print()

# ===============================================================
#                    LOAD MODEL
# ===============================================================

print("🔧 Loading model...")
if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
    print("Please update the path!")
    sys.exit(1)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
cfg = checkpoint['config']

# Re-create model
model = create_srpsi_v2_model(cfg, DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
print(f"   Best Val Loss: {checkpoint['val_loss']:.4f}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# ===============================================================
#                    LOAD TEST DATA
# ===============================================================

print("📊 Loading test data...")
_, _, test_loader = create_dataloaders(
    data_path=DATA_PATH,
    tin=16,
    tout=32,
    batch_size=32,
    num_train=4000,
    num_val=400,
    num_test=400,
    num_workers=0,
    seed=42
)
print(f"✅ Test data loaded: {len(test_loader)} batches")
print()

# ===============================================================
#                    PHYSICAL METRICS FUNCTIONS
# ===============================================================

def compute_energy_drift(pred, target):
    """
    Energy Drift: |E(pred) - E(target)|
    where E = 0.5 * sum(u^2)
    """
    energy_pred = 0.5 * torch.sum(pred ** 2, dim=[1, 2])  # [batch]
    energy_target = 0.5 * torch.sum(target ** 2, dim=[1, 2])  # [batch]
    drift = torch.abs(energy_pred - energy_target).mean()
    return drift.item()

def compute_momentum_drift(pred, target):
    """
    Momentum Drift: |P(pred) - P(target)|
    where P = sum(u)
    """
    momentum_pred = torch.sum(pred, dim=[1, 2])  # [batch]
    momentum_target = torch.sum(target, dim=[1, 2])  # [batch]
    drift = torch.abs(momentum_pred - momentum_target).mean()
    return drift.item()

# ===============================================================
#                    RUN TESTS
# ===============================================================

print("=" * 70)
print(" " * 25 + "TESTING")
print("=" * 70)
print()

all_mse = []
all_energy_drift = []
all_momentum_drift = []

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        # Get data
        batch_x = batch['x'].to(DEVICE)
        batch_y = batch['y'].to(DEVICE)

        # Transpose: [batch, tin, nx] -> [batch, nx, tin]
        batch_x = batch_x.transpose(1, 2)
        batch_y = batch_y.transpose(1, 2)

        # Predict
        pred_y = model(batch_x)

        # Compute metrics
        mse = torch.mean((pred_y - batch_y) ** 2).item()
        energy_drift = compute_energy_drift(pred_y, batch_y)
        momentum_drift = compute_momentum_drift(pred_y, batch_y)

        all_mse.append(mse)
        all_energy_drift.append(energy_drift)
        all_momentum_drift.append(momentum_drift)

# ===============================================================
#                    RESULTS
# ===============================================================

avg_mse = np.mean(all_mse)
avg_energy_drift = np.mean(all_energy_drift)
avg_momentum_drift = np.mean(all_momentum_drift)

print()
print("=" * 70)
print(" " * 20 + "TEST RESULTS")
print("=" * 70)
print()
print(f"📊 MSE:               {avg_mse:.4f}")
print(f"🔥 Energy Drift:      {avg_energy_drift:.4f}")
print(f"⚡ Momentum Drift:    {avg_momentum_drift:.4f}")
print()

# Compare with baselines
print("🎯 Comparisons:")
if avg_energy_drift < 10.88:
    print(f"   v1.0 Energy Drift (10.88):    ✅ BEATS v1.0!")
else:
    print(f"   v1.0 Energy Drift (10.88):    {avg_energy_drift:.4f} (was 10.88)")

if avg_momentum_drift < 18.49:
    print(f"   v1.0 Momentum Drift (18.49):  ✅ BEATS v1.0!")
else:
    print(f"   v1.0 Momentum Drift (18.49):  {avg_momentum_drift:.4f} (was 18.49)")

print()

# Save results
results = {
    'model': 'SRΨ-v2.0-Hybrid (Colab-trained)',
    'checkpoint_epoch': int(checkpoint['epoch']),
    'checkpoint_val_loss': float(checkpoint['val_loss']),
    'test_metrics': {
        'mse': float(avg_mse),
        'energy_drift': float(avg_energy_drift),
        'momentum_drift': float(avg_momentum_drift)
    },
    'baselines': {
        'v1.0': {
            'energy_drift': 10.883,
            'momentum_drift': 18.49
        },
        'transformer': {
            'energy_drift': 13.99,
            'momentum_drift': 1.38
        }
    },
    'improvements': {
        'energy_drift_vs_v1': f"{((10.883 - avg_energy_drift) / 10.883 * 100):.1f}%",
        'momentum_drift_vs_v1': f"{((18.49 - avg_momentum_drift) / 18.49 * 100):.1f}%"
    }
}

with open(OUTPUT_DIR / 'physical_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"💾 Results saved to {OUTPUT_DIR}/physical_test_results.json")
print()
print("=" * 70)
print(" " * 20 + "TEST COMPLETED ✅")
print("=" * 70)
print()
print("🚀 Next: Run extrapolation test (test_v2_extrapolation.py)")
print()
