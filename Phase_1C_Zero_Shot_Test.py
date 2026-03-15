# Phase 1C: Zero-Shot Extrapolation Test (T=200)
#
# 目标: 验证模型在 2x 训练长度的预测能力
# 训练: 48 步 (tin=16, tout=32)
# 测试: 200 步 (4 倍训练长度)

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

print('\n' + '='*70)
print(' ' * 10 + 'PHASE 1C: ZERO-SHOT EXTRAPOLATION TEST (T=200)')
print('='*70)

print('\n🎯 Testing: 2x training length prediction (T=200)')
print('   Hypothesis: Conv fails, Transformer maintains stability\n')

# Load data
data = load_burgers_data()

# Test parameters
rollout_steps = 200  # 4x training length!
test_samples = 10

print(f'📊 Rollout steps: {rollout_steps} (4x training length)')
print(f'📊 Samples: {test_samples}')
print(f'⏱️  Estimated time: 8-12 minutes\\n')

# Initialize results
results_zeroshot = {
    name: {
        'short_term_error': [],  # Steps 0-48 (within training)
        'long_term_error': [],    # Steps 48-200 (extrapolation)
        'explosion_step': None    # When error > 10x initial
    }
    for name in models.keys()
}

# Ground truth limited (we only have 48 steps)
u_init = torch.tensor(data['u_test'][:test_samples, :16, :], dtype=torch.float32)
u_true_short = torch.tensor(data['u_test'][:test_samples, 16:48, :], dtype=torch.float32)

print('🔄 Running zero-shot extrapolation...\\n')

for name, model_info in models.items():
    print(f'🧪 Testing {name}...', end=' ', flush=True)

    model = model_info['model']
    u_current = u_init.clone()

    # Track errors
    errors_short = []
    errors_long = []
    initial_error = None

    # Auto-regressive rollout
    for step in range(0, rollout_steps, 32):  # Predict 32 steps at a time
        with torch.no_grad():
            u_pred = model(u_current.to(device)).cpu()

        # Compute error for steps within training distribution (0-48)
        if step < 48:
            # Compare with available ground truth
            gt_start = max(0, step)
            gt_end = min(48, step + 32)
            pred_start = 0
            pred_end = gt_end - gt_start

            if gt_end > gt_start:
                gt_chunk = u_true_short[:, gt_start:gt_end, :]
                pred_chunk = u_pred[:, pred_start:pred_end, :]

                error = torch.mean((pred_chunk - gt_chunk) ** 2).item()
                errors_short.append(error)

                # Record initial error
                if step == 0:
                    initial_error = error

        # For extrapolation (48-200), use prediction magnitude as error metric
        pred_magnitude = torch.mean(u_pred ** 2).item()
        errors_long.append(pred_magnitude)

        # Check for explosion (error > 10x initial)
        if initial_error is not None and pred_magnitude > 10 * initial_error:
            results_zeroshot[name]['explosion_step'] = step
            print(f'\\n   ⚠️  Exploded at step {step}!')
            break

        # Shift window: keep last 16 steps as next input
        u_current = torch.cat([u_current[:, 16:, :], u_pred[:, -16:, :]], dim=1)

    # Store results
    results_zeroshot[name]['short_term_error'] = errors_short
    results_zeroshot[name]['long_term_error'] = errors_long

    # Compute metrics
    if errors_short:
        short_term_avg = np.mean(errors_short)
    else:
        short_term_avg = 0

    if errors_long:
        long_term_avg = np.mean(errors_long)
    else:
        long_term_avg = 0

    # Extrapolation ratio: how much error grows in extrapolation
    extrapolation_ratio = long_term_avg / (short_term_avg + 1e-10)

    results_zeroshot[name]['short_term_avg'] = short_term_avg
    results_zeroshot[name]['long_term_avg'] = long_term_avg
    results_zeroshot[name]['extrapolation_ratio'] = extrapolation_ratio

    print('✅')

print('\\n' + '-'*70)
print(' ' * 20 + 'ZERO-SHOT EXTRAPOLATION RESULTS')
print('-'*70)

print(f'\\n{"Model":<20} {"Short (0-48)":<18} {"Long (48-200)":<18} {"Ratio":<12} {"Explosion":<12}')
print('-'*80)

for name in models.keys():
    r = results_zeroshot[name]
    short = r['short_term_avg']
    long = r['long_term_avg']
    ratio = r['extrapolation_ratio']
    explosion = r['explosion_step']

    explosion_str = f"Step {explosion}" if explosion else "✅ Stable"
    print(f'{name:<20} {short:<18.4f} {long:<18.4f} {ratio:<12.2f}x {explosion_str:<12}')

print('\\n' + '='*70)
print(' ' * 25 + 'KEY INSIGHTS')
print('='*70)

print('\\n🔍 Analysis:')
print('\\n1. Extrapolation Ratio (lower = better stability):')
for name in models.keys():
    ratio = results_zeroshot[name]['extrapolation_ratio']
    print(f'   {name:<20}: {ratio:.2f}x')

print('\\n2. Explosion Detection:')
exploded = [name for name, r in results_zeroshot.items() if r['explosion_step'] is not None]
if exploded:
    print(f'   ❌ Exploded models: {", ".join(exploded)}')
else:
    print('   ✅ All models remained stable!')

print('\\n3. TRAE Hypothesis Verification:')
conv_ratio = results_zeroshot['Exp4_Conv']['extrapolation_ratio']
srpsi_ratio = results_zeroshot['Exp2_SRΨ_Real']['extrapolation_ratio']
transformer_ratio = results_zeroshot['Exp5_Transformer']['extrapolation_ratio']

print(f'   Conv: {conv_ratio:.2f}x')
print(f'   SRΨ Real: {srpsi_ratio:.2f}x')
print(f'   Transformer: {transformer_ratio:.2f}x')

print('\\n' + '='*70)
print('✅ Phase 1C completed!')
print('='*70)
