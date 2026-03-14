"""
Ablation Study Evaluation Script
==================================

Evaluate all 5 models (SRΨ Full + 4 ablation models) on test set.

Author: SRΨ-Engine Tiny Experiment
"""

import torch
import yaml
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from models import (
    SRPsiEngine,
    SRPsiEngineReal,
    SRPsiEngineNoR,
    ConvBaseline,
    TransformerRelPE,
    BaselineTransformer
)
from datasets import get_dataloader
from metrics import compute_all_metrics


def load_model(model_type: str, checkpoint_path: str, cfg: dict, device: torch.device):
    """
    Load a trained model from checkpoint.

    Args:
        model_type: Model type identifier
        checkpoint_path: Path to model checkpoint
        cfg: Configuration dict
        device: Target device

    Returns:
        model: Loaded model in eval mode
    """
    # Model hyperparameters
    tin = cfg["data"]["tin"]
    tout = cfg["data"]["tout"]
    nx = cfg["data"]["nx"]
    hidden_dim = cfg["model"]["hidden_dim"]

    # Create model
    if model_type == "srpsi":
        model = SRPsiEngine(
            tin=tin,
            tout=tout,
            nx=nx,
            hidden_dim=hidden_dim,
            num_layers=cfg["model"]["num_layers"]
        )
    elif model_type == "srpsi_real":
        model = SRPsiEngineReal(
            tin=tin,
            tout=tout,
            nx=nx,
            hidden_dim=hidden_dim,
            num_layers=cfg["model"]["num_layers"]
        )
    elif model_type == "srpsi_no_r":
        model = SRPsiEngineNoR(
            tin=tin,
            tout=tout,
            nx=nx,
            hidden_dim=hidden_dim,
            num_layers=cfg["model"]["num_layers"]
        )
    elif model_type == "conv_baseline":
        model = ConvBaseline(
            tin=tin,
            tout=tout,
            nx=nx,
            hidden_dim=hidden_dim,
            depth=cfg["model"]["num_layers"],
            kernel_size=5
        )
    elif model_type == "transformer_rel_pe":
        model = TransformerRelPE(
            tin=tin,
            tout=tout,
            nx=nx,
            d_model=hidden_dim,
            nhead=4,
            num_layers=cfg["model"]["num_layers"]
        )
    elif model_type == "transformer":
        model = BaselineTransformer(
            tin=tin,
            tout=tout,
            nx=nx,
            d_model=hidden_dim,
            nhead=4,
            num_layers=cfg["model"]["num_layers"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def evaluate_all_models(
    cfg_path: str,
    results_dir: str = "results/ablation",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate all models and save comparison results.

    Args:
        cfg_path: Path to config file
        results_dir: Directory to save results
        device: Target device
    """
    device = torch.device(device)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Get test dataloader
    test_loader = get_dataloader(
        cfg["data"]["train_path"],
        cfg["data"]["test_path"],
        batch_size=cfg["eval"]["batch_size"],
        tin=cfg["data"]["tin"],
        tout=cfg["data"]["tout"],
        split="test"
    )

    # Model types and their checkpoint paths
    models_to_eval = {
        "SRΨ Full": ("srpsi", "outputs/srpsi_best.pth"),
        "SRΨ Real-only": ("srpsi_real", "outputs/ablation_srpsi_real_best.pth"),
        "SRΨ w/o R": ("srpsi_no_r", "outputs/ablation_srpsi_no_r_best.pth"),
        "Conv Baseline": ("conv_baseline", "outputs/ablation_conv_baseline_best.pth"),
        "Transformer Rel-PE": ("transformer_rel_pe", "outputs/ablation_transformer_rel_pe_best.pth"),
        "Transformer Baseline": ("transformer", "outputs/transformer_best.pth"),
    }

    # Results storage
    all_results = {}

    # Evaluate each model
    print("\n" + "="*60)
    print("ABLATION STUDY EVALUATION")
    print("="*60 + "\n")

    for model_name, (model_type, ckpt_path) in models_to_eval.items():
        print(f"\n▶ Evaluating: {model_name}")
        print(f"  Checkpoint: {ckpt_path}")

        # Check if checkpoint exists
        if not Path(ckpt_path).exists():
            print(f"  ⚠ Warning: Checkpoint not found, skipping...")
            continue

        try:
            # Load model
            model = load_model(model_type, ckpt_path, cfg, device)

            # Evaluate
            metrics = compute_all_metrics(model, test_loader, device, cfg)

            # Store results
            all_results[model_name] = metrics

            # Print results
            print(f"  ✓ Rollout MSE:         {metrics['rollout_mse']:.6f}")
            print(f"  ✓ Late Horizon MSE:    {metrics['late_horizon_mse']:.6f}")
            print(f"  ✓ Energy Drift:        {metrics['energy_drift']:.6f}")
            print(f"  ✓ Shift Robustness:    {metrics['shift_robustness']:.6f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Save results to JSON
    results_path = results_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    # Generate comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60 + "\n")

    # Create table header
    header = f"{'Model':<25} {'Rollout MSE':<15} {'Late MSE':<15} {'Energy Drift':<15} {'Shift Error':<15}"
    print(header)
    print("-" * len(header))

    # Sort by rollout MSE
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['rollout_mse'])

    for model_name, metrics in sorted_results:
        row = (
            f"{model_name:<25} "
            f"{metrics['rollout_mse']:<15.6f} "
            f"{metrics['late_horizon_mse']:<15.6f} "
            f"{metrics['energy_drift']:<15.6f} "
            f"{metrics['shift_robustness']:<15.6f}"
        )
        print(row)

    # Compute relative improvements vs SRΨ Full
    if "SRΨ Full" in all_results:
        print("\n" + "="*60)
        print("RELATIVE IMPROVEMENT (vs SRΨ Full)")
        print("="*60 + "\n")

        srpsi_metrics = all_results["SRΨ Full"]

        for model_name, metrics in sorted_results:
            if model_name == "SRΨ Full":
                continue

            rollout_diff = ((metrics['rollout_mse'] - srpsi_metrics['rollout_mse'])
                           / srpsi_metrics['rollout_mse'] * 100)
            shift_diff = ((metrics['shift_robustness'] - srpsi_metrics['shift_robustness'])
                         / srpsi_metrics['shift_robustness'] * 100)

            print(f"{model_name}:")
            print(f"  Rollout MSE:  {rollout_diff:+.2f}%")
            print(f"  Shift Error:  {shift_diff:+.2f}%")
            print()

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ablation study models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/burgers.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/ablation",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluate_all_models(
        cfg_path=args.config,
        results_dir=args.results_dir,
        device=args.device
    )
