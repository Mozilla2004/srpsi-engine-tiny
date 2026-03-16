#!/usr/bin/env python3
"""
Analyze Exp2 and Exp3 Training Results
=======================================

Compare the training metrics from checkpoints.

Author: SRΨ-Engine Tiny Experiment
"""

import torch
import json
from pathlib import Path
import numpy as np


def analyze_checkpoint(checkpoint_path: str, model_name: str):
    """Extract training metrics from checkpoint"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*70}")

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Extract information
    info = {
        'model': model_name,
        'checkpoint_path': checkpoint_path,
        'final_epoch': ckpt.get('epoch', 'N/A'),
        'train_loss': ckpt.get('loss', 'N/A'),
        'val_loss': ckpt.get('val_loss', 'N/A'),
        'best_val_loss': ckpt.get('best_val_loss', 'N/A'),
        'best_val_mse': ckpt.get('best_val_mse', 'N/A'),
        'best_val_drift': ckpt.get('best_val_drift', 'N/A'),
    }

    # Model parameters
    state_dict = ckpt.get('model_state_dict', {})
    total_params = sum(p.numel() for p in state_dict.values())
    info['total_params'] = total_params

    # Checkpoint file info
    file_path = Path(checkpoint_path)
    info['file_size_mb'] = file_path.stat().st_size / (1024*1024)

    # Print summary
    print(f"\n📦 Checkpoint: {checkpoint_path}")
    print(f"📁 File Size:   {info['file_size_mb']:.2f} MB")

    print(f"\n🏗️  Model Architecture:")
    print(f"  Parameters:  {total_params:,}")

    print(f"\n📊 Training Progress:")
    print(f"  Final Epoch:          {info['final_epoch']}/80")
    print(f"  Completion:           {info['final_epoch']/80*100:.1f}%")

    print(f"\n📈 Training Metrics:")
    print(f"  Final Loss:           {info['train_loss'] if isinstance(info['train_loss'], (int, float)) else 'N/A'}")
    print(f"  Best Val Loss:        {info['best_val_loss'] if isinstance(info['best_val_loss'], (int, float)) else 'N/A'}")
    print(f"  Best Val MSE:         {info['best_val_mse'] if isinstance(info['best_val_mse'], (int, float)) else 'N/A'}")
    print(f"  Best Val Drift:       {info['best_val_drift'] if isinstance(info['best_val_drift'], (int, float)) else 'N/A'}")

    # Show checkpoint keys
    print(f"\n🔑 Checkpoint Keys:")
    for key in ckpt.keys():
        value = ckpt[key]
        if isinstance(value, dict):
            print(f"  • {key}: dict with {len(value)} items")
        elif isinstance(value, torch.Tensor):
            print(f"  • {key}: tensor {tuple(value.shape)}")
        elif isinstance(value, int):
            print(f"  • {key}: {value}")
        else:
            print(f"  • {key}: {type(value).__name__}")

    return info


def main():
    """Analyze and compare Exp2 and Exp3"""
    print("\n" + "="*70)
    print(" " * 15 + "EXP2 vs EXP3: COMPARATIVE ANALYSIS")
    print("="*70)

    # Checkpoint paths
    exp2_ckpt = "outputs/ablation_study/srpsi_real/srpsi_real/checkpoints/final.pt"
    exp3_ckpt = "outputs/ablation_study/srpsi_no_r/srpsi_no_r/checkpoints/final.pt"

    # Analyze both
    exp2_info = analyze_checkpoint(exp2_ckpt, "Exp2: SRΨ Real-only")
    exp3_info = analyze_checkpoint(exp3_ckpt, "Exp3: SRΨ w/o Rhythm")

    if exp2_info is None or exp3_info is None:
        print("\n❌ Could not analyze one or both checkpoints")
        return

    # Comparison Report
    print("\n" + "="*70)
    print(" " * 20 + "COMPARISON REPORT")
    print("="*70)

    print("\n📊 Validation Metrics Comparison:")
    print(f"\n{'Metric':<25} {'Exp2 (Real)':<20} {'Exp3 (w/o R)':<20} {'Difference':<20}")
    print("-" * 85)

    # Best Val Loss
    if isinstance(exp2_info['best_val_loss'], (int, float)) and isinstance(exp3_info['best_val_loss'], (int, float)):
        loss_diff = ((exp3_info['best_val_loss'] - exp2_info['best_val_loss'])
                    / exp2_info['best_val_loss'] * 100)
        print(f"{'Best Val Loss':<25} {exp2_info['best_val_loss']:<20.2f} {exp3_info['best_val_loss']:<20.2f} {loss_diff:+.2f}%")

    # Best Val MSE
    if isinstance(exp2_info['best_val_mse'], (int, float)) and isinstance(exp3_info['best_val_mse'], (int, float)):
        mse_diff = ((exp3_info['best_val_mse'] - exp2_info['best_val_mse'])
                   / exp2_info['best_val_mse'] * 100)
        print(f"{'Best Val MSE':<25} {exp2_info['best_val_mse']:<20.6f} {exp3_info['best_val_mse']:<20.6f} {mse_diff:+.2f}%")

    # Best Val Drift
    if isinstance(exp2_info['best_val_drift'], (int, float)) and isinstance(exp3_info['best_val_drift'], (int, float)):
        drift_diff = ((exp3_info['best_val_drift'] - exp2_info['best_val_drift'])
                     / exp2_info['best_val_drift'] * 100)
        print(f"{'Best Val Drift':<25} {exp2_info['best_val_drift']:<20.6f} {exp3_info['best_val_drift']:<20.6f} {drift_diff:+.2f}%")

    # Parameters
    param_diff = ((exp3_info['total_params'] - exp2_info['total_params'])
                 / exp2_info['total_params'] * 100)
    print(f"{'Total Parameters':<25} {exp2_info['total_params']:<20,} {exp3_info['total_params']:<20,} {param_diff:+.2f}%")

    # File size
    size_diff = ((exp3_info['file_size_mb'] - exp2_info['file_size_mb'])
                / exp2_info['file_size_mb'] * 100)
    print(f"{'Checkpoint Size (MB)':<25} {exp2_info['file_size_mb']:<20.2f} {exp3_info['file_size_mb']:<20.2f} {size_diff:+.2f}%")

    # Key Findings
    print("\n" + "="*70)
    print(" " * 20 + "KEY FINDINGS")
    print("="*70)

    print("\n🔍 Impact of Removing Rhythm Operator (R):")

    if isinstance(exp2_info['best_val_loss'], (int, float)) and isinstance(exp3_info['best_val_loss'], (int, float)):
        if exp3_info['best_val_loss'] > exp2_info['best_val_loss']:
            loss_increase = loss_diff
            print(f"\n  ❌ Validation Loss INCREASED by {loss_increase:.2f}%")
            print(f"     Exp2: {exp2_info['best_val_loss']:.2f} → Exp3: {exp3_info['best_val_loss']:.2f}")
            print(f"     → R operator is CRITICAL for performance")
        else:
            print(f"\n  ✓ Validation Loss DECREASED by {abs(loss_diff):.2f}%")
            print(f"     Exp2: {exp2_info['best_val_loss']:.2f} → Exp3: {exp3_info['best_val_loss']:.2f}")
            print(f"     → Unexpected: R operator removal helped?")

    if isinstance(exp2_info['best_val_mse'], (int, float)) and isinstance(exp3_info['best_val_mse'], (int, float)):
        if exp3_info['best_val_mse'] > exp2_info['best_val_mse']:
            mse_increase = mse_diff
            print(f"\n  ❌ Val MSE INCREASED by {mse_increase:.2f}%")
            print(f"     Exp2: {exp2_info['best_val_mse']:.6f} → Exp3: {exp3_info['best_val_mse']:.6f}")
            print(f"     → R operator improves prediction accuracy")
        else:
            print(f"\n  ✓ Val MSE DECREASED by {abs(mse_diff):.2f}%")

    # Model complexity
    print(f"\n🏗️  Model Complexity:")
    print(f"  • Exp2 parameters:  {exp2_info['total_params']:,}")
    print(f"  • Exp3 parameters:  {exp3_info['total_params']:,}")
    print(f"  • Difference:       {exp3_info['total_params'] - exp2_info['total_params']:,} params ({param_diff:+.2f}%)")

    if exp3_info['total_params'] > exp2_info['total_params']:
        print(f"  → Exp3 has MORE parameters despite removing R operator")
        print(f"  → This suggests architectural differences beyond just R")

    print("\n" + "="*70)
    print(" " * 18 + "ABLATION STUDY CONCLUSIONS")
    print("="*70)

    print("\n🎯 Question: Does the Rhythm Operator (R) matter?")

    if isinstance(exp2_info['best_val_loss'], (int, float)) and isinstance(exp3_info['best_val_loss'], (int, float)):
        if exp3_info['best_val_loss'] > exp2_info['best_val_loss']:
            print("\n✅ Answer: YES, R operator is crucial")
            print(f"\n  Evidence:")
            print(f"  • Removing R caused {loss_diff:.1f}% higher validation loss")
            print(f"  • Val MSE increased by {mse_diff:.1f}%")
            print(f"  • R operator provides significant performance gain")
            print(f"\n  Interpretation:")
            print(f"  • R operator captures temporal rhythm/oscillation patterns")
            print(f"  • Without R, model loses ability to model time-dependent dynamics")
            print(f"  • This validates the design choice of including R operator")
        else:
            print("\n⚠️  Answer: R operator removal IMPROVED performance?")
            print(f"\n  This is unexpected and requires investigation:")
            print(f"  • Validation loss decreased by {abs(loss_diff):.1f}%")
            print(f"  • Possible reasons:")
            print(f"    - Overfitting in Exp2 (with R)")
            print(f"    - Regularization effect of removing R")
            print(f"    - Need to check test set performance")

    print("\n" + "="*70)
    print("✅ Analysis Complete")
    print("="*70)

    # Save results
    results = {
        "Exp2_SRΨ_Real": {
            "final_epoch": int(exp2_info['final_epoch']),
            "best_val_loss": float(exp2_info['best_val_loss']) if isinstance(exp2_info['best_val_loss'], (int, float)) else None,
            "best_val_mse": float(exp2_info['best_val_mse']) if isinstance(exp2_info['best_val_mse'], (int, float)) else None,
            "best_val_drift": float(exp2_info['best_val_drift']) if isinstance(exp2_info['best_val_drift'], (int, float)) else None,
            "parameters": int(exp2_info['total_params']),
            "file_size_mb": float(exp2_info['file_size_mb']),
        },
        "Exp3_SRΨ_w/o_R": {
            "final_epoch": int(exp3_info['final_epoch']),
            "best_val_loss": float(exp3_info['best_val_loss']) if isinstance(exp3_info['best_val_loss'], (int, float)) else None,
            "best_val_mse": float(exp3_info['best_val_mse']) if isinstance(exp3_info['best_val_mse'], (int, float)) else None,
            "best_val_drift": float(exp3_info['best_val_drift']) if isinstance(exp3_info['best_val_drift'], (int, float)) else None,
            "parameters": int(exp3_info['total_params']),
            "file_size_mb": float(exp3_info['file_size_mb']),
        },
        "comparison": {
            "val_loss_diff_pct": float(loss_diff) if isinstance(exp2_info['best_val_loss'], (int, float)) and isinstance(exp3_info['best_val_loss'], (int, float)) else None,
            "val_mse_diff_pct": float(mse_diff) if isinstance(exp2_info['best_val_mse'], (int, float)) and isinstance(exp3_info['best_val_mse'], (int, float)) else None,
            "param_diff_pct": float(param_diff),
        }
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "exp2_vs_exp3_analysis.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {results_path}")


if __name__ == "__main__":
    main()
