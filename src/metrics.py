"""
Evaluation Metrics for SRΨ-Engine Experiments
==============================================

Metrics for comparing model performance:
1. Rollout MSE: Prediction error across time horizon
2. Late-horizon MSE: Error in later time steps (stability test)
3. Energy drift: Conservation quality
4. Shift robustness: Translation equivariance

These metrics test the four key hypotheses:
- Long-term rollout stability
- Conservation control
- Shift/phase robustness
- Recovery from perturbations

Author: SRΨ-Engine Tiny Experiment
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


def rollout_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean squared error across entire rollout.

    Args:
        pred: Prediction [B, Tout, X]
        target: Ground truth [B, Tout, X]

    Returns:
        mse: Average MSE
    """
    return ((pred - target) ** 2).mean().item()


def late_horizon_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    frac: float = 0.5
) -> float:
    """
    MSE in later half of prediction horizon.

    Tests long-term stability: models that drift will fail here.

    Args:
        pred: Prediction [B, Tout, X]
        target: Ground truth [B, Tout, X]
        frac: Fraction of horizon to consider as "late" (default: 0.5)

    Returns:
        mse: Late-horizon MSE
    """
    t0 = int(pred.shape[1] * frac)
    return ((pred[:, t0:] - target[:, t0:]) ** 2).mean().item()


def energy_drift(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Average absolute energy drift.

    Energy(t) = Σ_x u(x, t)²
    Drift = |Energy_pred(t) - Energy_true(t)|

    Args:
        pred: Prediction [B, Tout, X]
        target: Ground truth [B, Tout, X]

    Returns:
        drift: Average energy drift
    """
    e_pred = (pred ** 2).sum(dim=-1)       # [B, T]
    e_true = (target ** 2).sum(dim=-1)     # [B, T]
    drift = (e_pred - e_true).abs().mean().item()
    return drift


@torch.no_grad()
def shift_robustness(
    model: nn.Module,
    x: torch.Tensor,
    shift: int = 4
) -> float:
    """
    Test translation equivariance: model(shift(x)) ≈ shift(model(x)).

    Lower is better (more robust to shifts).

    Args:
        model: Model to test
        x: Input [B, Tin, X]
        shift: Number of grid points to shift

    Returns:
        mse: Shift consistency error
    """
    y1 = model(x)
    y2 = model(torch.roll(x, shifts=shift, dims=-1))
    y1_shift = torch.roll(y1, shifts=shift, dims=-1)

    return ((y2 - y1_shift) ** 2).mean().item()


def compute_all_metrics(
    model: nn.Module,
    dataloader,
    device: torch.device,
    cfg: dict
) -> Dict[str, float]:
    """
    Compute all evaluation metrics on a dataset.

    Args:
        model: Model instance
        dataloader: Data loader (test or val)
        device: Target device
        cfg: Configuration dict

    Returns:
        metrics: Dict of all metric values
    """
    model.eval()

    rollout_mses = []
    late_mses = []
    drifts = []
    shift_errors = []

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            pred = model(x)

        # Compute metrics
        rollout_mses.append(rollout_mse(pred, y))
        late_mses.append(late_horizon_mse(pred, y))
        drifts.append(energy_drift(pred, y))

        # Shift robustness (compute on fewer batches for speed)
        if len(shift_errors) < 10:
            shift_errors.append(shift_robustness(model, x, cfg["eval"]["perturb_shift"]))

    return {
        "rollout_mse": np.mean(rollout_mses),
        "late_horizon_mse": np.mean(late_mses),
        "energy_drift": np.mean(drifts),
        "shift_robustness": np.mean(shift_errors),
    }


def temporal_error_profile(
    pred: torch.Tensor,
    target: torch.Tensor
) -> np.ndarray:
    """
    Compute MSE at each time step.

    Useful for plotting error growth over time.

    Args:
        pred: Prediction [B, Tout, X]
        target: Ground truth [B, Tout, X]

    Returns:
        errors: MSE at each time step [Tout]
    """
    errors = ((pred - target) ** 2).mean(dim=(0, 2))  # [T]
    return errors.cpu().numpy()


def energy_profile(pred: torch.Tensor, target: torch.Tensor) -> tuple:
    """
    Compute energy trajectories.

    Args:
        pred: Prediction [B, Tout, X]
        target: Ground truth [B, Tout, X]

    Returns:
        e_pred: Predicted energy [B, Tout]
        e_true: True energy [B, Tout]
    """
    e_pred = (pred ** 2).sum(dim=-1).cpu().numpy()   # [B, T]
    e_true = (target ** 2).sum(dim=-1).cpu().numpy() # [B, T]
    return e_pred, e_true


if __name__ == "__main__":
    # Test metrics
    from models.baseline_mlp import BaselineMLP

    tin, tout, nx = 16, 32, 128
    batch_size = 4

    model = BaselineMLP(tin, tout, nx)
    device = torch.device("cpu")

    # Create dummy data
    x = torch.randn(batch_size, tin, nx)
    y = torch.randn(batch_size, tout, nx)

    print("Testing metrics...")

    # Test rollout MSE
    mse = rollout_mse(y, y + 0.1 * torch.randn_like(y))
    print(f"✓ Rollout MSE: {mse:.6f}")

    # Test late horizon MSE
    late_mse = late_horizon_mse(y, y + 0.1 * torch.randn_like(y))
    print(f"✓ Late horizon MSE: {late_mse:.6f}")

    # Test energy drift
    drift = energy_drift(y, y * 1.1)
    print(f"✓ Energy drift: {drift:.6f}")

    # Test shift robustness
    shift_err = shift_robustness(model, x, shift=4)
    print(f"✓ Shift robustness: {shift_err:.6f}")

    # Test temporal error profile
    errors = temporal_error_profile(y, y + 0.1 * torch.randn_like(y))
    print(f"✓ Temporal error profile: {errors.shape}")

    # Test energy profile
    e_pred, e_true = energy_profile(y, y)
    print(f"✓ Energy profile: pred={e_pred.shape}, true={e_true.shape}")

    print("\n✓ All metrics tests passed")
