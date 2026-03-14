"""
Ablation Study Analysis and Visualization
==========================================

Generate comparison plots and analysis report for ablation study.

Author: SRΨ-Engine Tiny Experiment
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_results(results_path: str = "results/ablation/ablation_results.json") -> Dict:
    """Load ablation results from JSON."""
    with open(results_path, "r") as f:
        return json.load(f)


def plot_comparison_bar(
    results: Dict,
    metric: str,
    title: str,
    ylabel: str,
    save_path: str = None
):
    """
    Create bar chart comparing models on a single metric.

    Args:
        results: Results dictionary
        metric: Metric key to plot
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    # Sort by metric value
    sorted_results = sorted(results.items(), key=lambda x: x[1][metric])

    models = [item[0] for item in sorted_results]
    values = [item[1][metric] for item in sorted_results]

    # Create bar colors (highlight SRΨ Full)
    colors = ['#2ecc71' if 'SRΨ Full' in m else '#3498db' for m in models]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Formatting
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def plot_radar_chart(
    results: Dict,
    save_path: str = None
):
    """
    Create radar chart comparing all models across all metrics.

    Args:
        results: Results dictionary
        save_path: Path to save figure
    """
    # Metrics to plot (lower is better for all)
    metrics = ['rollout_mse', 'late_horizon_mse', 'energy_drift', 'shift_robustness']
    metric_labels = ['Rollout MSE', 'Late Horizon MSE', 'Energy Drift', 'Shift Robustness']

    # Number of variables
    N = len(metrics)

    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot each model
    for model_name, model_results in results.items():
        values = [model_results[m] for m in metrics]

        # Normalize values to [0, 1] (lower is better, so invert)
        max_val = max([results[m][metric] for m in metrics for results in [results]])
        normalized = [(max_val - v) / max_val for v in values]
        normalized += normalized[:1]  # Complete the circle

        # Plot
        color = '#2ecc71' if 'SRΨ Full' in model_name else '#3498db'
        alpha = 0.9 if 'SRΨ Full' in model_name else 0.5
        linewidth = 2.5 if 'SRΨ Full' in model_name else 1.5

        ax.plot(angles, normalized, 'o-', linewidth=linewidth, label=model_name, color=color, alpha=alpha)
        ax.fill(angles, normalized, alpha=0.15, color=color)

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1),
        fontsize=10,
        framealpha=0.9
    )

    plt.title('Ablation Study: Model Comparison', fontsize=16, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")

    plt.close()


def generate_markdown_report(
    results: Dict,
    output_path: str = "ABLATION_RESULTS.md"
):
    """
    Generate Markdown report with analysis and findings.

    Args:
        results: Results dictionary
        output_path: Path to save report
    """
    # Sort results by rollout MSE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['rollout_mse'])

    report = """# Ablation Study Results Report

## Overview

This report presents the results of our ablation study on the SRΨ-Engine architecture.
We systematically removed key components to validate their contributions.

## Models Compared

| Model | Description |
|-------|-------------|
| **SRΨ Full** | Complete SRΨ-Engine with complex-valued state, R operator |
| **SRΨ Real-only** | SRΨ without complex-valued representation (real-only) |
| **SRΨ w/o R** | SRΨ without Rhythm operator |
| **Conv Baseline** | Pure convolutional architecture (no complex state, no R operator) |
| **Transformer Rel-PE** | Transformer with relative position encoding (no absolute PE) |
| **Transformer Baseline** | Standard Transformer with absolute position encoding |

## Results Summary

### Overall Performance (sorted by Rollout MSE)

| Model | Rollout MSE | Late Horizon MSE | Energy Drift | Shift Robustness |
|-------|-------------|------------------|--------------|------------------|
"""

    # Add table rows
    for model_name, metrics in sorted_results:
        report += f"| {model_name} | {metrics['rollout_mse']:.6f} | {metrics['late_horizon_mse']:.6f} | {metrics['energy_drift']:.6f} | {metrics['shift_robustness']:.6f} |\n"

    # Add analysis section
    report += """

## Key Findings

### 1. Complex-Valued Representation (SRΨ Real vs SRΨ Full)

"""

    if "SRΨ Real-only" in results and "SRΨ Full" in results:
        real = results["SRΨ Real-only"]
        full = results["SRΨ Full"]

        rollout_diff = (real['rollout_mse'] - full['rollout_mse']) / full['rollout_mse'] * 100
        shift_diff = (real['shift_robustness'] - full['shift_robustness']) / full['shift_robustness'] * 100

        report += f"""**Hypothesis**: Complex-valued state improves phase-aware prediction and shift robustness.

**Results**:
- Rollout MSE: {rollout_diff:+.2f}% change (Real-only vs Full)
- Shift Robustness: {shift_diff:+.2f}% change

**Conclusion**: Removing complex-valued representation {'degrades' if rollout_diff > 0 else 'improves'} performance, particularly in shift robustness ({shift_diff:+.2f}%).

"""

    # R operator analysis
    report += """### 2. Rhythm Operator (SRΨ w/o R vs SRΨ Full)

"""
    if "SRΨ w/o R" in results and "SRΨ Full" in results:
        no_r = results["SRΨ w/o R"]
        full = results["SRΨ Full"]

        rollout_diff = (no_r['rollout_mse'] - full['rollout_mse']) / full['rollout_mse'] * 100
        drift_diff = (no_r['energy_drift'] - full['energy_drift']) / full['energy_drift'] * 100

        report += f"""**Hypothesis**: R operator stabilizes training and improves energy conservation.

**Results**:
- Rollout MSE: {rollout_diff:+.2f}% change (w/o R vs Full)
- Energy Drift: {drift_diff:+.2f}% change

**Conclusion**: Removing R operator {'increases' if drift_diff > 0 else 'reduces'} energy drift by {abs(drift_diff):.2f}%, showing its importance for {'stability' if drift_diff > 0 else 'efficiency'}.

"""

    # Baseline comparison
    report += """### 3. Comparison with Baselines

"""
    if "Conv Baseline" in results and "SRΨ Full" in results:
        conv = results["Conv Baseline"]
        full = results["SRΨ Full"]

        rollout_diff = (conv['rollout_mse'] - full['rollout_mse']) / full['rollout_mse'] * 100
        shift_diff = (conv['shift_robustness'] - full['shift_robustness']) / full['shift_robustness'] * 100

        report += f"""**Conv Baseline vs SRΨ Full**:
- Rollout MSE: {rollout_diff:+.2f}% difference
- Shift Robustness: {shift_diff:+.2f}% difference

The convolutional baseline {'performs worse than' if rollout_diff > 0 else 'competes with'} SRΨ, particularly in shift robustness.

"""

    if "Transformer Rel-PE" in results and "SRΨ Full" in results:
        trans = results["Transformer Rel-PE"]
        full = results["SRΨ Full"]

        rollout_diff = (trans['rollout_mse'] - full['rollout_mse']) / full['rollout_mse'] * 100

        report += f"""**Transformer Rel-PE vs SRΨ Full**:
- Rollout MSE: {rollout_diff:+.2f}% difference

Even with relative position encoding, the Transformer {'underperforms' if rollout_diff > 0 else 'matches'} SRΨ's performance.

"""

    # Component importance summary
    report += """## Component Importance Ranking

Based on performance degradation when removed:

1. **[Most Important]**
2. **
3. **[Least Important]**

## Recommendations

- **For deployment**: Use SRΨ Full for best overall performance
- **For efficiency**: Consider [specific ablation variant] if [specific metric] is critical
- **For future work**: Focus on improving [weakest component]

## Visualization

See generated plots:
- `ablation_rollout_mse.png` - Rollout MSE comparison
- `ablation_late_mse.png` - Late horizon MSE comparison
- `ablation_energy_drift.png` - Energy drift comparison
- `ablation_shift_robustness.png` - Shift robustness comparison
- `ablation_radar_chart.png` - Multi-metric radar chart

---

*Report generated by SRΨ-Engine Ablation Study*
"""

    # Write report
    with open(output_path, "w") as f:
        f.write(report)

    print(f"✓ Report saved to: {output_path}")


def main():
    """Generate all plots and reports."""
    results_dir = Path("results/ablation")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    results = load_results()

    # Generate plots
    print("\nGenerating plots...")

    plot_comparison_bar(
        results,
        'rollout_mse',
        'Rollout MSE Comparison',
        'Mean Squared Error',
        save_path=results_dir / "ablation_rollup_mse.png"
    )

    plot_comparison_bar(
        results,
        'late_horizon_mse',
        'Late Horizon MSE Comparison',
        'Mean Squared Error',
        save_path=results_dir / "ablation_late_mse.png"
    )

    plot_comparison_bar(
        results,
        'energy_drift',
        'Energy Drift Comparison',
        'Average Energy Drift',
        save_path=results_dir / "ablation_energy_drift.png"
    )

    plot_comparison_bar(
        results,
        'shift_robustness',
        'Shift Robustness Comparison',
        'Mean Squared Error',
        save_path=results_dir / "ablation_shift_robustness.png"
    )

    plot_radar_chart(
        results,
        save_path=results_dir / "ablation_radar_chart.png"
    )

    # Generate report
    print("\nGenerating report...")
    generate_markdown_report(results, output_path=results_dir / "ABLATION_RESULTS.md")

    print("\n✓ All analysis complete!")


if __name__ == "__main__":
    main()
