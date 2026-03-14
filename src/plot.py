"""
Plotting Utilities for SRΨ-Engine Experiments
==============================================

Generates key visualizations:
1. Truth vs prediction comparison
2. Temporal error growth
3. Energy drift trajectories
4. Model comparison bar charts

These plots directly test the four hypotheses:
- Long-term stability (error growth)
- Conservation control (energy drift)
- Shift robustness (bar chart)
- Recovery capability (implicit in trajectories)

Author: SRΨ-Engine Tiny Experiment
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional


def plot_truth_vs_prediction(
    pred: np.ndarray,
    target: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Truth vs Prediction"
):
    """
    Plot ground truth vs prediction at multiple time points.

    Args:
        pred: Prediction [Tout, X] or [B, Tout, X]
        target: Ground truth [Tout, X] or [B, Tout, X]
        save_path: Path to save figure
        title: Plot title
    """
    # Remove batch dimension if present
    if pred.ndim == 3:
        pred = pred[0]
        target = target[0]

    tout, nx = pred.shape

    # Select time points to visualize
    time_points = [0, tout // 4, tout // 2, 3 * tout // 4, tout - 1]

    fig, axes = plt.subplots(1, len(time_points), figsize=(4 * len(time_points), 3))

    x = np.arange(nx)

    for ax, t in zip(axes, time_points):
        ax.plot(x, target[t], 'k-', label='Truth', linewidth=2)
        ax.plot(x, pred[t], 'r--', label='Prediction', linewidth=1.5, alpha=0.8)
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_temporal_error(
    errors_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Temporal Error Growth"
):
    """
    Plot error growth over time horizon.

    Args:
        errors_dict: Dict of {model_name: errors [T]}
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, errors in errors_dict.items():
        ax.plot(errors, label=model_name, linewidth=2)

    ax.set_xlabel("Time step")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_energy_drift(
    energy_dict: Dict[str, tuple],
    save_path: Optional[str] = None,
    title: str = "Energy Trajectories"
):
    """
    Plot energy trajectories over time.

    Args:
        energy_dict: Dict of {model_name: (e_pred, e_true)}
                    where each is [T] or averaged over batch
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, (e_pred, e_true) in energy_dict.items():
        # Average over batch if needed
        if e_pred.ndim == 2:
            e_pred = e_pred.mean(0)
            e_true = e_true.mean(0)

        ax.plot(e_true, 'k-', label='Truth', linewidth=2, alpha=0.5)
        ax.plot(e_pred, label=model_name, linewidth=1.5)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_model_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Model Comparison"
):
    """
    Bar chart comparing metrics across models.

    Args:
        metrics_dict: Dict of {model_name: {metric_name: value}}
        save_path: Path to save figure
        title: Plot title
    """
    # Extract metric names and model names
    metric_names = list(next(iter(metrics_dict.values())).keys())
    model_names = list(metrics_dict.keys())

    # Setup plot
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4))

    if num_metrics == 1:
        axes = [axes]

    # Plot each metric
    for ax, metric_name in zip(axes, metric_names):
        values = [metrics_dict[model][metric_name] for model in model_names]

        bars = ax.bar(model_names, values, alpha=0.7)

        # Color bars: lower is better for all our metrics
        for bar in bars:
            bar.set_color('steelblue')

        ax.set_ylabel(metric_name)
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_shift_robustness(
    shift_errors_dict: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Shift Robustness (lower is better)"
):
    """
    Bar chart for shift robustness comparison.

    Args:
        shift_errors_dict: Dict of {model_name: shift_error}
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    model_names = list(shift_errors_dict.keys())
    errors = list(shift_errors_dict.values())

    bars = ax.bar(model_names, errors, alpha=0.7, color='coral')

    ax.set_ylabel("Shift Consistency Error")
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def generate_comparison_plots(
    results: Dict[str, Dict],
    output_dir: str
):
    """
    Generate all comparison plots from experiment results.

    Args:
        results: Dict of {model_name: {metrics, predictions, etc.}}
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Truth vs prediction for each model
    for model_name, model_results in results.items():
        if "pred_sample" in model_results and "target_sample" in model_results:
            plot_truth_vs_prediction(
                model_results["pred_sample"],
                model_results["target_sample"],
                save_path=output_path / f"{model_name}_truth_vs_pred.png",
                title=f"{model_name}: Truth vs Prediction"
            )

    # 2. Temporal error growth
    errors_dict = {}
    for model_name, model_results in results.items():
        if "temporal_errors" in model_results:
            errors_dict[model_name] = model_results["temporal_errors"]

    if errors_dict:
        plot_temporal_error(
            errors_dict,
            save_path=output_path / "temporal_error_comparison.png",
            title="Temporal Error Growth (lower is better)"
        )

    # 3. Energy drift
    energy_dict = {}
    for model_name, model_results in results.items():
        if "energy_pred" in model_results and "energy_true" in model_results:
            energy_dict[model_name] = (
                model_results["energy_pred"],
                model_results["energy_true"]
            )

    if energy_dict:
        plot_energy_drift(
            energy_dict,
            save_path=output_path / "energy_drift_comparison.png",
            title="Energy Trajectories"
        )

    # 4. Model comparison (metrics)
    metrics_dict = {}
    for model_name, model_results in results.items():
        if "metrics" in model_results:
            metrics_dict[model_name] = model_results["metrics"]

    if metrics_dict:
        # Filter key metrics for comparison
        key_metrics = ["rollout_mse", "late_horizon_mse", "energy_drift", "shift_robustness"]
        comparison_dict = {}

        for model_name, model_metrics in metrics_dict.items():
            comparison_dict[model_name] = {
                k: model_metrics[k] for k in key_metrics if k in model_metrics
            }

        plot_model_comparison(
            comparison_dict,
            save_path=output_path / "model_comparison.png",
            title="Model Performance Comparison"
        )

    # 5. Shift robustness
    shift_errors = {}
    for model_name, model_results in results.items():
        if "metrics" in model_results and "shift_robustness" in model_results["metrics"]:
            shift_errors[model_name] = model_results["metrics"]["shift_robustness"]

    if shift_errors:
        plot_shift_robustness(
            shift_errors,
            save_path=output_path / "shift_robustness.png",
            title="Shift Robustness (lower is better)"
        )

    print(f"\n✓ Plots saved to: {output_path}")


if __name__ == "__main__":
    # Test plotting functions
    print("Testing plotting utilities...")

    # Create dummy data
    tout, nx = 32, 128

    # Test truth vs prediction
    pred = np.random.randn(tout, nx)
    target = np.random.randn(tout, nx)

    plot_truth_vs_prediction(pred, target)
    print("✓ Truth vs prediction plot created")

    # Test temporal error
    errors_dict = {
        "MLP": np.exp(0.05 * np.arange(tout)),
        "Transformer": np.exp(0.03 * np.arange(tout)),
        "SRPsi": np.exp(0.02 * np.arange(tout)),
    }
    plot_temporal_error(errors_dict)
    print("✓ Temporal error plot created")

    # Test energy drift
    energy_dict = {
        "MLP": (np.random.randn(tout) + np.arange(tout) * 0.1, np.arange(tout)),
        "Transformer": (np.random.randn(tout) + np.arange(tout) * 0.05, np.arange(tout)),
    }
    plot_energy_drift(energy_dict)
    print("✓ Energy drift plot created")

    # Test model comparison
    metrics_dict = {
        "MLP": {"rollout_mse": 0.5, "energy_drift": 0.3},
        "Transformer": {"rollout_mse": 0.3, "energy_drift": 0.2},
        "SRPsi": {"rollout_mse": 0.2, "energy_drift": 0.1},
    }
    plot_model_comparison(metrics_dict)
    print("✓ Model comparison plot created")

    # Test shift robustness
    shift_errors = {"MLP": 0.8, "Transformer": 0.5, "SRPsi": 0.2}
    plot_shift_robustness(shift_errors)
    print("✓ Shift robustness plot created")

    print("\n✓ All plotting tests passed")
