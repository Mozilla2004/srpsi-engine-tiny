#!/usr/bin/env python3
"""
Evaluate Exp2 (SRΨ Real-only) and Exp3 (SRΨ w/o R)
===================================================

Analyze the two completed experiments from Windows TRAE.

Author: SRΨ-Engine Tiny Experiment
"""

import torch
import yaml
import json
from pathlib import Path
import numpy as np
import sys

# Add src to path
sys.path.insert(0, 'src')

from models import SRPsiEngineReal, SRPsiEngineNoR
from datasets import get_dataloader
from metrics import compute_all_metrics


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint and extract training metrics"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    info = {
        'epoch': ckpt.get('epoch', 'N/A'),
        'train_loss': ckpt.get('loss', 'N/A'),
        'val_loss': ckpt.get('val_loss', 'N/A'),
        'best_val_loss': ckpt.get('best_val_loss', 'N/A'),
        'best_val_mse': ckpt.get('best_val_mse', 'N/A'),
        'best_val_drift': ckpt.get('best_val_drift', 'N/A'),
    }

    return info, ckpt['model_state_dict']


def evaluate_model(model_type: str, checkpoint_path: str, cfg: dict, device: torch.device):
    """Load and evaluate a model"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")

    # Load checkpoint info
    train_info, state_dict = load_checkpoint(checkpoint_path)

    print("📊 Training Summary:")
    print(f"  Final Epoch:      {train_info['epoch']}/80")
    print(f"  Final Loss:       {train_info['train_loss'] if isinstance(train_info['train_loss'], (int, float)) else 'N/A'}")
    print(f"  Best Val Loss:    {train_info['best_val_loss'] if isinstance(train_info['best_val_loss'], (int, float)) else 'N/A'}")
    print(f"  Best Val MSE:     {train_info['best_val_mse'] if isinstance(train_info['best_val_mse'], (int, float)) else 'N/A'}")
    print(f"  Best Val Drift:   {train_info['best_val_drift'] if isinstance(train_info['best_val_drift'], (int, float)) else 'N/A'}")

    # Model hyperparameters
    tin = cfg["data"]["tin"]
    tout = cfg["data"]["tout"]
    nx = cfg["data"]["nx"]
    hidden_dim = cfg["model"]["hidden_dim"]

    # Create model
    if model_type == "Exp2 (SRΨ Real-only)":
        model = SRPsiEngineReal(
            tin=tin,
            tout=tout,
            nx=nx,
            hidden_dim=hidden_dim,
            num_layers=cfg["model"]["num_layers"]
        )
    elif model_type == "Exp3 (SRΨ w/o R)":
        model = SRPsiEngineNoR(
            tin=tin,
            tout=tout,
            nx=nx,
            hidden_dim=hidden_dim,
            num_layers=cfg["model"]["num_layers"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🏗️  Model Architecture:")
    print(f"  Parameters:       {total_params:,}")
    print(f"  Checkpoint Size:  {Path(checkpoint_path).stat().st_size / (1024*1024):.2f} MB")

    # Get test dataloader
    test_loader = get_dataloader(
        cfg["data"]["train_path"],
        cfg["data"]["test_path"],
        batch_size=cfg["eval"]["batch_size"],
        tin=cfg["data"]["tin"],
        tout=cfg["data"]["tout"],
        split="test"
    )

    # Evaluate
    print(f"\n🔬 Running Evaluation...")
    metrics = compute_all_metrics(model, test_loader, device, cfg)

    return train_info, metrics, total_params


def main():
    """Evaluate Exp2 and Exp3"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Device: {device}")

    # Load config
    config_path = "config/burgers.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Checkpoint paths
    exp2_ckpt = "outputs/ablation_study/srpsi_real/srpsi_real/checkpoints/final.pt"
    exp3_ckpt = "outputs/ablation_study/srpsi_no_r/srpsi_no_r/checkpoints/final.pt"

    # Check if checkpoints exist
    if not Path(exp2_ckpt).exists():
        print(f"❌ Exp2 checkpoint not found: {exp2_ckpt}")
        return
    if not Path(exp3_ckpt).exists():
        print(f"❌ Exp3 checkpoint not found: {exp3_ckpt}")
        return

    # Evaluate both models
    exp2_train, exp2_metrics, exp2_params = evaluate_model(
        "Exp2 (SRΨ Real-only)", exp2_ckpt, cfg, device
    )

    exp3_train, exp3_metrics, exp3_params = evaluate_model(
        "Exp3 (SRΨ w/o R)", exp3_ckpt, cfg, device
    )

    # Comparison Report
    print("\n" + "="*70)
    print(" " * 15 + "COMPARISON REPORT: Exp2 vs Exp3")
    print("="*70)

    print("\n📊 Test Set Metrics:")
    print(f"\n{'Metric':<25} {'Exp2 (Real)':<20} {'Exp3 (w/o R)':<20} {'Difference':<20}")
    print("-" * 85)

    # Rollout MSE
    rollout_diff = ((exp3_metrics['rollout_mse'] - exp2_metrics['rollout_mse'])
                   / exp2_metrics['rollout_mse'] * 100)
    print(f"{'Rollout MSE':<25} {exp2_metrics['rollout_mse']:<20.6f} {exp3_metrics['rollout_mse']:<20.6f} {rollout_diff:+.2f}%")

    # Late Horizon MSE
    late_diff = ((exp3_metrics['late_horizon_mse'] - exp2_metrics['late_horizon_mse'])
                / exp2_metrics['late_horizon_mse'] * 100)
    print(f"{'Late Horizon MSE':<25} {exp2_metrics['late_horizon_mse']:<20.6f} {exp3_metrics['late_horizon_mse']:<20.6f} {late_diff:+.2f}%")

    # Energy Drift
    drift_diff = ((exp3_metrics['energy_drift'] - exp2_metrics['energy_drift'])
                 / exp2_metrics['energy_drift'] * 100)
    print(f"{'Energy Drift':<25} {exp2_metrics['energy_drift']:<20.6f} {exp3_metrics['energy_drift']:<20.6f} {drift_diff:+.2f}%")

    # Shift Robustness
    shift_diff = ((exp3_metrics['shift_robustness'] - exp2_metrics['shift_robustness'])
                 / exp2_metrics['shift_robustness'] * 100)
    print(f"{'Shift Robustness':<25} {exp2_metrics['shift_robustness']:<20.6f} {exp3_metrics['shift_robustness']:<20.6f} {shift_diff:+.2f}%")

    print("\n" + "="*70)
    print(" " * 20 + "KEY FINDINGS")
    print("="*70)

    print("\n🔍 Rhythm Operator (R) Impact:")
    print(f"  • Rollout MSE:        {rollout_diff:+.2f}% {'✓ BETTER' if rollout_diff > 0 else '✗ WORSE'} with R")
    print(f"  • Late Horizon MSE:   {late_diff:+.2f}% {'✓ BETTER' if late_diff > 0 else '✗ WORSE'} with R")
    print(f"  • Energy Drift:       {drift_diff:+.2f}% {'✓ BETTER' if drift_diff > 0 else '✗ WORSE'} with R")
    print(f"  • Shift Robustness:   {shift_diff:+.2f}% {'✓ BETTER' if shift_diff > 0 else '✗ WORSE'} with R")

    print("\n💡 Interpretation:")
    if rollout_diff > 0:
        print("  ✅ Exp2 (with R) achieves LOWER rollout MSE")
        print("     → R operator improves long-term prediction accuracy")
    else:
        print("  ⚠️  Exp3 (w/o R) achieves LOWER rollout MSE")
        print("     → Unexpected result, needs further investigation")

    if abs(shift_diff) > 10:
        print(f"\n  ⚠️  Large difference ({shift_diff:+.1f}%) in shift robustness")
        print("     → R operator significantly affects spatial generalization")

    print("\n" + "="*70)
    print("✅ Evaluation Complete")
    print("="*70)

    # Save results
    results = {
        "Exp2_SRΨ_Real": {
            "training": {
                "final_epoch": exp2_train['epoch'],
                "best_val_loss": float(exp2_train['best_val_loss']) if isinstance(exp2_train['best_val_loss'], (int, float)) else None,
                "best_val_mse": float(exp2_train['best_val_mse']) if isinstance(exp2_train['best_val_mse'], (int, float)) else None,
            },
            "test_metrics": {
                "rollout_mse": float(exp2_metrics['rollout_mse']),
                "late_horizon_mse": float(exp2_metrics['late_horizon_mse']),
                "energy_drift": float(exp2_metrics['energy_drift']),
                "shift_robustness": float(exp2_metrics['shift_robustness']),
            },
            "parameters": exp2_params,
        },
        "Exp3_SRΨ_w/o_R": {
            "training": {
                "final_epoch": exp3_train['epoch'],
                "best_val_loss": float(exp3_train['best_val_loss']) if isinstance(exp3_train['best_val_loss'], (int, float)) else None,
                "best_val_mse": float(exp3_train['best_val_mse']) if isinstance(exp3_train['best_val_mse'], (int, float)) else None,
            },
            "test_metrics": {
                "rollout_mse": float(exp3_metrics['rollout_mse']),
                "late_horizon_mse": float(exp3_metrics['late_horizon_mse']),
                "energy_drift": float(exp3_metrics['energy_drift']),
                "shift_robustness": float(exp3_metrics['shift_robustness']),
            },
            "parameters": exp3_params,
        },
        "comparison": {
            "rollout_mse_diff_pct": float(rollout_diff),
            "late_horizon_mse_diff_pct": float(late_diff),
            "energy_drift_diff_pct": float(drift_diff),
            "shift_robustness_diff_pct": float(shift_diff),
        }
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "exp2_vs_exp3.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {results_path}")


if __name__ == "__main__":
    main()
