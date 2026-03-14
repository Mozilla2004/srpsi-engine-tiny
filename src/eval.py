"""
Evaluation Script for SRΨ-Engine Experiments
=============================================

Loads trained models and generates comparison visualizations.

Outputs:
- Truth vs prediction plots for each model
- Temporal error growth curves
- Energy drift trajectories
- Model comparison bar charts

Usage:
    python src/eval.py --config config/burgers.yaml --output_dir outputs/burgers_1d

Author: SRΨ-Engine Tiny Experiment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils import load_config, set_seed, get_device, load_checkpoint, count_parameters
from src.datasets import create_dataloaders
from src.models import BaselineMLP, BaselineTransformer, SRPsiEngineTiny
from src.metrics import (
    compute_all_metrics, temporal_error_profile,
    energy_profile, rollout_mse
)
from src.plot import (
    plot_truth_vs_prediction, plot_temporal_error,
    plot_energy_drift, plot_shift_robustness, plot_model_comparison
)


def load_model_from_checkpoint(
    model_type: str,
    checkpoint_path: str,
    cfg: dict,
    device: torch.device
):
    """
    Load model from checkpoint.

    Args:
        model_type: Type of model
        checkpoint_path: Path to checkpoint file
        cfg: Configuration dict
        device: Target device

    Returns:
        model: Loaded model
    """
    # Create model
    if model_type == "baseline_mlp":
        model = BaselineMLP(
            cfg['task']['tin'],
            cfg['task']['tout'],
            cfg['task']['nx'],
            hidden_dim=cfg['model']['hidden_dim']
        )
    elif model_type == "baseline_transformer":
        model = BaselineTransformer(
            cfg['task']['tin'],
            cfg['task']['tout'],
            cfg['task']['nx'],
            d_model=cfg['model']['hidden_dim'],
            nhead=4,
            num_layers=cfg['model']['depth'],
            dropout=cfg['model']['dropout']
        )
    elif model_type == "srpsi_engine":
        model = SRPsiEngineTiny(
            tin=cfg['task']['tin'],
            nx=cfg['task']['nx'],
            hidden_dim=cfg['model']['hidden_dim'],
            depth=cfg['model']['depth'],
            kernel_size=cfg['model']['kernel_size'],
            dt=0.1,
            tout=cfg['task']['tout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)

    return model


def evaluate_single_model(
    model: torch.nn.Module,
    test_loader,
    cfg: dict,
    device: torch.device,
    model_name: str
) -> dict:
    """
    Evaluate a single model and collect metrics and samples.

    Args:
        model: Model instance
        test_loader: Test dataloader
        cfg: Configuration dict
        device: Target device
        model_name: Name of model

    Returns:
        results: Dict containing metrics and sample data
    """
    print(f"\nEvaluating {model_name}...")

    model.eval()

    # Collect metrics
    metrics = compute_all_metrics(model, test_loader, device, cfg)

    # Collect sample predictions and errors
    sample_batch = next(iter(test_loader))
    x = sample_batch["x"].to(device)
    y = sample_batch["y"].to(device)

    with torch.no_grad():
        pred = model(x)

    # Convert to numpy for plotting
    pred_np = pred.cpu().numpy()
    target_np = y.cpu().numpy()

    # Get first sample
    pred_sample = pred_np[0]
    target_sample = target_np[0]

    # Temporal error profile
    temporal_errors = temporal_error_profile(pred, y)

    # Energy profiles
    e_pred, e_true = energy_profile(pred, y)

    results = {
        "metrics": metrics,
        "pred_sample": pred_sample,
        "target_sample": target_sample,
        "temporal_errors": temporal_errors,
        "energy_pred": e_pred,
        "energy_true": e_true,
    }

    # Print metrics
    print(f"  Rollout MSE:       {metrics['rollout_mse']:.6f}")
    print(f"  Late Horizon MSE:  {metrics['late_horizon_mse']:.6f}")
    print(f"  Energy Drift:      {metrics['energy_drift']:.6f}")
    print(f"  Shift Robustness:  {metrics['shift_robustness']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SRΨ-Engine models")
    parser.add_argument("--config", type=str, default="config/burgers.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs/burgers_1d",
                        help="Base output directory")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data file (will search if not provided)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename to use (default: final.pt)")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Setup device
    device = get_device(cfg.get("device", "cuda"))

    # Find data file
    if args.data is None:
        # Search in model directories
        for model_name in ["baseline_mlp", "baseline_transformer", "srpsi_engine"]:
            model_dir = Path(args.output_dir) / model_name
            data_path = model_dir / "data.npy"
            if data_path.exists():
                args.data = str(data_path)
                break

    if args.data is None:
        raise FileNotFoundError("Could not find data file. Please specify --data")

    print(f"Using data: {args.data}")

    # Create dataloader
    print("Creating test dataloader...")
    _, _, test_loader = create_dataloaders(
        data_path=args.data,
        tin=cfg['task']['tin'],
        tout=cfg['task']['tout'],
        batch_size=cfg['training']['batch_size'],
        num_train=cfg['task']['samples_train'],
        num_val=cfg['task']['samples_val'],
        num_test=cfg['task']['samples_test']
    )

    # Custom paths for our experiments
    experiment_models = {
        "SRΨ-Engine": {
            "type": "srpsi_engine",
            "path": Path(args.output_dir) / "srpsi_engine_v0.1.3" / "srpsi_engine" / "checkpoints" / "final.pt"
        },
        "Baseline Transformer": {
            "type": "baseline_transformer",
            "path": Path(args.output_dir) / "baseline_transformer" / "baseline_transformer" / "checkpoints" / "final.pt"
        }
    }

    # Evaluate all models
    all_results = {}

    for model_name, model_info in experiment_models.items():
        checkpoint_path = model_info["path"]
        model_type = model_info["type"]

        if not checkpoint_path.exists():
            print(f"⚠ Warning: Checkpoint not found for {model_name}: {checkpoint_path}")
            continue

        # Load model
        model = load_model_from_checkpoint(model_type, str(checkpoint_path), cfg, device)
        model.eval()

        # Evaluate
        results = evaluate_single_model(model, test_loader, cfg, device, model_name)
        all_results[model_name] = results

    if len(all_results) == 0:
        print("Error: No models evaluated successfully")
        return

    # Generate comparison plots
    print("\n" + "=" * 50)
    print("Generating comparison plots...")
    print("=" * 50)

    from src.plot import generate_comparison_plots

    comparison_dir = Path(args.output_dir) / "comparison"
    generate_comparison_plots(all_results, str(comparison_dir))

    # Print summary table
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print(f"\n{'Model':<25} {'Rollout MSE':<15} {'Late MSE':<15} {'Energy Drift':<15} {'Shift Error':<15}")
    print("-" * 85)

    for model_name in all_results.keys():
        metrics = all_results[model_name]["metrics"]
        print(f"{model_name:<25} "
              f"{metrics['rollout_mse']:<15.6f} "
              f"{metrics['late_horizon_mse']:<15.6f} "
              f"{metrics['energy_drift']:<15.6f} "
              f"{metrics['shift_robustness']:<15.6f}")

    print("\n✓ Evaluation complete!")
    print(f"Comparison plots saved to: {comparison_dir}")


if __name__ == "__main__":
    main()
