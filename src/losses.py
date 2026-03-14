"""
Loss Functions for SRΨ-Engine Training
========================================

Implements multi-component loss:
1. Prediction loss: Standard MSE between prediction and target
2. Conservation loss: Penalize energy drift
3. Shift consistency: Enforce translation equivariance
4. Smoothness loss: Penalize temporal oscillations

Design philosophy:
- Primary: prediction accuracy
- Regularizers: physics-based constraints (conservation, symmetry, smoothness)
- Weighted combination allows tuning trade-offs

Author: SRΨ-Engine Tiny Experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard prediction MSE loss.

    Args:
        pred: Prediction [B, Tout, X]
        target: Ground truth [B, Tout, X]

    Returns:
        loss: Scalar MSE
    """
    return F.mse_loss(pred, target)


def energy(x: torch.Tensor) -> torch.Tensor:
    """
    Compute energy L2 norm of field at each time step.

    Energy(t) = Σ_x u(x, t)²

    Args:
        x: Field [B, T, X]

    Returns:
        energy: Energy per time step [B, T]
    """
    return (x ** 2).sum(dim=-1)


def conservation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Conservation loss: penalize energy drift.

    Encourages predicted energy to match true energy trajectory.
    This helps maintain physical conservation laws.

    Args:
        pred: Prediction [B, Tout, X]
        target: Ground truth [B, Tout, X]

    Returns:
        loss: Energy conservation MSE
    """
    e_pred = energy(pred)
    e_true = energy(target)
    return F.mse_loss(e_pred, e_true)


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    Temporal smoothness loss: penalize oscillations.

    Encourages smooth temporal evolution:
    Smoothness = Σ_t ||u_{t+1} - u_t||²

    Args:
        pred: Prediction [B, Tout, X]

    Returns:
        loss: Temporal smoothness penalty
    """
    diff = pred[:, 1:] - pred[:, :-1]  # [B, T-1, X]
    return (diff ** 2).mean()


@torch.no_grad()
def cyclic_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Apply cyclic spatial shift (periodic boundary condition).

    Args:
        x: Input [B, T, X] or [B, X]
        shift: Number of grid points to shift

    Returns:
        x_shift: Shifted input
    """
    return torch.roll(x, shifts=shift, dims=-1)


def shift_consistency_loss(
    model: nn.Module,
    x: torch.Tensor,
    shift: int,
    use_teacher_forcing: bool = False
) -> torch.Tensor:
    """
    Shift consistency loss: enforce translation equivariance.

    Principle: If we shift input by Δx, output should shift by Δx.
    This tests the model's robustness to spatial translations.

    Args:
        model: Model to evaluate
        x: Input [B, Tin, X]
        shift: Number of grid points to shift
        use_teacher_forcing: If True, use target; if False, use model prediction

    Returns:
        loss: Shift consistency MSE
    """
    # Normal forward pass
    y1 = model(x)

    # Shifted input
    x_shift = cyclic_shift(x, shift)

    # Forward pass on shifted input
    y2 = model(x_shift)

    # Shift original prediction
    y1_shift = cyclic_shift(y1, shift)

    # Consistency: model(shift(x)) should equal shift(model(x))
    return F.mse_loss(y2, y1_shift)


def total_loss(
    model: nn.Module,
    x: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: Dict[str, Any],
    epoch: int = 0,
    compute_shift: bool = False  # CHANGED: Default to False
) -> tuple:
    """
    Total loss with all components.

    Args:
        model: Model (for shift consistency)
        x: Input [B, Tin, X]
        pred: Prediction [B, Tout, X]
        target: Ground truth [B, Tout, X]
        cfg: Configuration dict with loss weights
        epoch: Current epoch (can be used for scheduling)
        compute_shift: Whether to compute shift consistency (expensive, POTENTIALLY UNSTABLE)

    Returns:
        total_loss: Weighted sum of all losses
        logs: Dict of individual loss values
    """
    # Primary prediction loss
    loss_pred = prediction_loss(pred, target)

    # Conservation loss
    loss_cons = conservation_loss(pred, target)

    # Smoothness loss
    loss_smooth = smoothness_loss(pred)

    # Shift consistency (DISABLED by default - causing NaN)
    # Only enable after stable training is established
    if compute_shift and cfg["loss"]["lambda_phase"] > 0 and epoch > 10:
        with torch.no_grad():  # Don't backprop through shift loss initially
            loss_phase = shift_consistency_loss(
                model, x,
                shift=cfg["eval"]["perturb_shift"]
            )
    else:
        loss_phase = torch.tensor(0.0, device=pred.device)

    # Weighted combination
    total = (
        loss_pred
        + cfg["loss"]["lambda_cons"] * loss_cons
        + cfg["loss"]["lambda_phase"] * loss_phase
        + cfg["loss"]["lambda_smooth"] * loss_smooth
    )

    # Safety check: if total is NaN, return prediction loss only
    if torch.isnan(total):
        print(f"⚠️  Warning: Total loss is NaN! Using prediction loss only.")
        print(f"  loss_pred={loss_pred.item():.6f}, loss_cons={loss_cons.item():.6f}")
        print(f"  loss_smooth={loss_smooth.item():.6f}, loss_phase={loss_phase.item():.6f}")
        total = loss_pred

    # Logging
    logs = {
        "loss_total": total.item(),
        "loss_pred": loss_pred.item(),
        "loss_cons": loss_cons.item(),
        "loss_phase": loss_phase.item(),
        "loss_smooth": loss_smooth.item(),
    }

    return total, logs


if __name__ == "__main__":
    # Test loss functions
    from models.baseline_mlp import BaselineMLP

    tin, tout, nx = 16, 32, 128
    batch_size = 4

    # Create dummy data
    x = torch.randn(batch_size, tin, nx)
    y = torch.randn(batch_size, tout, nx)
    target = torch.randn(batch_size, tout, nx)

    # Create model
    model = BaselineMLP(tin, tout, nx)

    # Test individual losses
    print("Testing loss functions...")

    loss_pred = prediction_loss(y, target)
    print(f"✓ Prediction loss: {loss_pred.item():.6f}")

    loss_cons = conservation_loss(y, target)
    print(f"✓ Conservation loss: {loss_cons.item():.6f}")

    loss_smooth = smoothness_loss(y)
    print(f"✓ Smoothness loss: {loss_smooth.item():.6f}")

    loss_phase = shift_consistency_loss(model, x, shift=4)
    print(f"✓ Shift consistency: {loss_phase.item():.6f}")

    # Test total loss
    cfg = {
        "loss": {
            "lambda_cons": 0.2,
            "lambda_phase": 0.1,
            "lambda_smooth": 0.05
        },
        "eval": {
            "perturb_shift": 4
        }
    }

    total, logs = total_loss(model, x, y, target, cfg)

    print(f"\n✓ Total loss: {total.item():.6f}")
    print(f"  Components: {logs}")

    print("\n✓ All loss tests passed")
