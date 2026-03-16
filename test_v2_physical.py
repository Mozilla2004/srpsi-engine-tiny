"""
SRΨ-v2.0 Physical Dimension Tests (Phase 1A)
==============================================

针对真实数据训练的 v2.0 模型进行完整的物理维度测试：
- Shift Robustness Test (s ∈ [0, 64])
- Energy Conservation
- Momentum Conservation
- Prediction Quality

对比基准：
- Transformer (Extrapolation Ratio: 0.24x, Shift Growth: 0.97x)
- SRΨ v1.0 (Extrapolation Ratio: 0.41x, Shift Growth: 0.74x)

Target:
- Extrapolation Ratio: < 0.3x
- Shift Growth: < 0.8x
- Energy Drift: < 0.3
- Momentum Drift: < 2.0

Author: TRAE + Claude Code
Version: 2.0-Real-Data-Test
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parent))

from src.models.srpsi_v2_hybrid import create_srpsi_v2_model
from src.training.physical_loss import PhysicalLoss

# ===============================================================
#                    CONFIGURATION
# ===============================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'checkpoints/v2_hybrid/checkpoint_best.pt'
DATA_PATH = 'data/processed'  # 真实数据
OUTPUT_DIR = Path('outputs/v2_physical_tests_real')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test configuration
NUM_SAMPLES = 100  # 测试样本数
SHIFT_VALUES = list(range(0, 65, 4))  # [0, 4, 8, ..., 64]
TIN = 16
TOUT = 32
NX = 128

print("=" * 70)
print(" " * 15 + "SRΨ-v2.0 PHYSICAL TESTS (REAL DATA)")
print("=" * 70)
print(f"📁 Checkpoint: {CHECKPOINT_PATH}")
print(f"📊 Device: {DEVICE}")
print(f"🎯 Shift values: {SHIFT_VALUES}")
print()

# ===============================================================
#                    LOAD MODEL & DATA
# ===============================================================

print("🔧 Loading model...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
cfg = checkpoint['config']

# Re-create model
model = create_srpsi_v2_model(cfg, DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
print(f"   Best Val Loss: {checkpoint['val_loss']:.4f}")
print()

# Load test data
print("📊 Loading test data...")
test_x = torch.load(f'{DATA_PATH}/test_x.pt').to(DEVICE)
test_y = torch.load(f'{DATA_PATH}/test_y.pt').to(DEVICE)
print(f"✅ Test data loaded: {test_x.shape} → {test_y.shape}")
print()

# Create physical loss function
loss_fn = PhysicalLoss(
    lambda_energy=0.1,
    lambda_momentum=0.1
)

# ===============================================================
#                    SHIFT ROBUSTNESS TEST
# ===============================================================

print("=" * 70)
print(" " * 20 + "SHIFT ROBUSTNESS TEST")
print("=" * 70)
print()

results = {
    'shift_values': [],
    'mse_errors': [],
    'energy_drifts': [],
    'momentum_drifts': [],
    'predictions': []
}

for shift in tqdm(SHIFT_VALUES, desc="Testing shifts"):
    # Apply spatial shift
    if shift == 0:
        shifted_x = test_x[:NUM_SAMPLES]
        shifted_y = test_y[:NUM_SAMPLES]
    else:
        # Circular shift along spatial dimension
        shifted_x = torch.roll(test_x[:NUM_SAMPLES], shifts=shift, dims=1)
        shifted_y = torch.roll(test_y[:NUM_SAMPLES], shifts=shift, dims=1)

    # Predict
    with torch.no_grad():
        pred_y = model(shifted_x)

    # Compute metrics
    mse_error = torch.mean((pred_y - shifted_y) ** 2).item()

    # Physical metrics
    energy_drift = loss_fn.energy_drift(pred_y, shifted_y).item()
    momentum_drift = loss_fn.momentum_drift(pred_y, shifted_y).item()

    # Store results
    results['shift_values'].append(shift)
    results['mse_errors'].append(mse_error)
    results['energy_drifts'].append(energy_drift)
    results['momentum_drifts'].append(momentum_drift)

    # Store one prediction for visualization
    results['predictions'].append({
        'shift': shift,
        'input': shifted_x[0].cpu().numpy(),
        'target': shifted_y[0].cpu().numpy(),
        'prediction': pred_y[0].cpu().numpy()
    })

    print(f"Shift={shift:2d}: MSE={mse_error:.4f}, E={energy_drift:.4f}, M={momentum_drift:.4f}")

print()

# ===============================================================
#                    ANALYSIS & METRICS
# ===============================================================

print("=" * 70)
print(" " * 20 + "METRICS ANALYSIS")
print("=" * 70)
print()

# 1. Shift Growth Error (SGE)
# SGE = (Error at shift=64) / (Error at shift=0)
mse_0 = results['mse_errors'][0]
mse_64 = results['mse_errors'][-1]
shift_growth = mse_64 / mse_0 if mse_0 > 0 else float('inf')

print(f"📊 Shift Growth Error:")
print(f"   MSE (shift=0):  {mse_0:.4f}")
print(f"   MSE (shift=64): {mse_64:.4f}")
print(f"   SGE Ratio:      {shift_growth:.4f}x")
print()

# 2. Baseline comparisons
print("🎯 Target Comparisons:")
print(f"   Target SGE:       < 0.8x (SRΨ v1.0)")
print(f"   Actual SGE:       {shift_growth:.4f}x")
if shift_growth < 0.8:
    print(f"   Status:           ✅ BEATS SRΨ v1.0!")
elif shift_growth < 0.97:
    print(f"   Status:           ✅ BEATS Transformer!")
else:
    print(f"   Status:           ⚠️  Needs improvement")
print()

# 3. Energy Drift Analysis
avg_energy_drift = np.mean(results['energy_drifts'])
print(f"🔥 Energy Drift:")
print(f"   Average:          {avg_energy_drift:.4f}")
print(f"   Target:           < 0.3 (Transformer)")
if avg_energy_drift < 0.3:
    print(f"   Status:           ✅ TARGET MET!")
else:
    print(f"   Status:           ⚠️  {avg_energy_drift / 0.3:.1f}x above target")
print()

# 4. Momentum Drift Analysis
avg_momentum_drift = np.mean(results['momentum_drifts'])
print(f"⚡ Momentum Drift:")
print(f"   Average:          {avg_momentum_drift:.4f}")
print(f"   Target:           < 2.0 (Transformer)")
if avg_momentum_drift < 2.0:
    print(f"   Status:           ✅ TARGET MET!")
else:
    print(f"   Status:           ⚠️  {avg_momentum_drift / 2.0:.1f}x above target")
print()

# ===============================================================
#                    SAVE RESULTS
# ===============================================================

print("💾 Saving results...")

# 1. Summary metrics
summary = {
    'model': 'SRΨ-v2.0-Hybrid',
    'data_type': 'real',
    'checkpoint_epoch': checkpoint['epoch'],
    'checkpoint_val_loss': float(checkpoint['val_loss']),
    'metrics': {
        'shift_growth': float(shift_growth),
        'mse_shift_0': float(mse_0),
        'mse_shift_64': float(mse_64),
        'avg_energy_drift': float(avg_energy_drift),
        'avg_momentum_drift': float(avg_momentum_drift)
    },
    'targets': {
        'shift_growth': 0.8,
        'energy_drift': 0.3,
        'momentum_drift': 2.0
    },
    'shift_tests': {
        'shift_values': results['shift_values'],
        'mse_errors': [float(x) for x in results['mse_errors']],
        'energy_drifts': [float(x) for x in results['energy_drifts']],
        'momentum_drifts': [float(x) for x in results['momentum_drifts']]
    }
}

with open(OUTPUT_DIR / 'physical_metrics_v2_real.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"   ✅ Metrics saved to {OUTPUT_DIR / 'physical_metrics_v2_real.json'}")

# 2. Save visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: MSE vs Shift
axes[0, 0].plot(results['shift_values'], results['mse_errors'], 'b-o', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Shift (s)', fontsize=12)
axes[0, 0].set_ylabel('MSE Error', fontsize=12)
axes[0, 0].set_title('Shift Robustness: MSE vs Shift', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=mse_0, color='r', linestyle='--', label=f'Baseline (shift=0): {mse_0:.4f}')
axes[0, 0].legend()

# Plot 2: Energy Drift vs Shift
axes[0, 1].plot(results['shift_values'], results['energy_drifts'], 'r-o', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Shift (s)', fontsize=12)
axes[0, 1].set_ylabel('Energy Drift', fontsize=12)
axes[0, 1].set_title('Energy Conservation vs Shift', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0.3, color='g', linestyle='--', label='Target: 0.3')
axes[0, 1].legend()

# Plot 3: Momentum Drift vs Shift
axes[1, 0].plot(results['shift_values'], results['momentum_drifts'], 'g-o', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Shift (s)', fontsize=12)
axes[1, 0].set_ylabel('Momentum Drift', fontsize=12)
axes[1, 0].set_title('Momentum Conservation vs Shift', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=2.0, color='orange', linestyle='--', label='Target: 2.0')
axes[1, 0].legend()

# Plot 4: Sample prediction (shift=32)
idx = SHIFT_VALUES.index(32) if 32 in SHIFT_VALUES else len(SHIFT_VALUES) // 2
pred_data = results['predictions'][idx]

# Visualize last timestep
axes[1, 1].plot(pred_data['target'][:, -1], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
axes[1, 1].plot(pred_data['prediction'][:, -1], 'r--', linewidth=2, label='Prediction', alpha=0.7)
axes[1, 1].set_xlabel('Spatial Index (x)', fontsize=12)
axes[1, 1].set_ylabel('Amplitude (u)', fontsize=12)
axes[1, 1].set_title(f'Sample Prediction (shift={SHIFT_VALUES[idx]})', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'physical_test_plots_v2_real.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Plots saved to {OUTPUT_DIR / 'physical_test_plots_v2_real.png'}")
print()

# ===============================================================
#                    FINAL SUMMARY
# ===============================================================

print("=" * 70)
print(" " * 20 + "TEST COMPLETED ✅")
print("=" * 70)
print()
print("📁 Results saved to:")
print(f"   {OUTPUT_DIR}/")
print()
print("🎯 Key Findings:")
print(f"   1. Shift Growth: {shift_growth:.4f}x (target: < 0.8x)")
print(f"   2. Energy Drift: {avg_energy_drift:.4f} (target: < 0.3)")
print(f"   3. Momentum Drift: {avg_momentum_drift:.4f} (target: < 2.0)")
print()
print("🚀 Ready for Phase 1C: Extrapolation Test (T=200)")
print()
