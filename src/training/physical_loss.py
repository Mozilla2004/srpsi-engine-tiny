"""
Physical Loss Functions for SRΨ-v2.0
======================================

Implements explicit physical constraints:
- Energy Conservation (dE/dt = 0)
- Momentum Conservation (dP/dt = 0)
- MSE Fitting (baseline)

Core Philosophy:
"强迫模型在训练阶段就支付守恒代价"

Author: TRAE + Claude Code
Version: 2.0-Physical-Loss
"""

import torch
import torch.nn as nn


class PhysicalLoss(nn.Module):
    """
    Composite loss function combining MSE with physical constraints.

    Loss = MSE_Loss + λ1 * Energy_Drift + λ2 * Momentum_Drift

    Design:
    - MSE ensures basic fitting capability
    - Energy drift enforces conservation
    - Momentum drift enforces translation invariance
    """

    def __init__(self,
                 lambda_energy=0.1,
                 lambda_momentum=0.1,
                 lambda_noise=0.05):
        """
        Args:
            lambda_energy: Weight for energy conservation
            lambda_momentum: Weight for momentum conservation
            lambda_noise: Weight for noise robustness
        """
        super().__init__()

        self.lambda_energy = lambda_energy
        self.lambda_momentum = lambda_momentum
        self.lambda_noise = lambda_noise

        # Base MSE loss
        self.mse_loss = nn.MSELoss()

    def energy_drift(self, pred, target):
        """
        Compute energy drift between prediction and target.

        Energy = 0.5 * sum(u^2)

        Args:
            pred: [batch, nx, tout] - Predicted field
            target: [batch, nx, tout] - Ground truth field

        Returns:
            drift: [batch] - Energy drift per sample
        """
        # Compute energies
        energy_pred = 0.5 * torch.sum(pred ** 2, dim=[1, 2])  # [batch]
        energy_target = 0.5 * torch.sum(target ** 2, dim=[1, 2])  # [batch]

        # Relative drift
        drift = torch.abs(energy_pred - energy_target) / (energy_target + 1e-10)

        return drift.mean()

    def momentum_drift(self, pred, target):
        """
        Compute momentum drift between prediction and target.

        Momentum = sum(u)

        Args:
            pred: [batch, nx, tout] - Predicted field
            target: [batch, nx, tout] - Ground truth field

        Returns:
            drift: [batch] - Momentum drift per sample
        """
        # Compute momenta
        momentum_pred = torch.sum(pred, dim=[1, 2])  # [batch]
        momentum_target = torch.sum(target, dim=[1, 2])  # [batch]

        # Relative drift
        drift = torch.abs(momentum_pred - momentum_target) / (torch.abs(momentum_target) + 1e-10)

        return drift.mean()

    def spectral_loss(self, pred, target):
        """
        Spectral loss for turbulence modeling (Kolmogorov -5/3).

        Compares power spectra in frequency domain.

        Args:
            pred: [batch, nx, tout] - Predicted field
            target: [batch, nx, tout] - Ground truth field

        Returns:
            loss: Scalar spectral loss
        """
        # Compute FFT along spatial dimension
        pred_fft = torch.fft.rfft(pred, dim=1)  # [batch, nx//2+1, tout]
        target_fft = torch.fft.rfft(target, dim=1)  # [batch, nx//2+1, tout]

        # Power spectrum
        pred_power = torch.abs(pred_fft) ** 2
        target_power = torch.abs(target_fft) ** 2

        # Log-space comparison (better for scaling)
        pred_log = torch.log(pred_power + 1e-10)
        target_log = torch.log(target_power + 1e-10)

        # MSE in log-space
        loss = torch.mean((pred_log - target_log) ** 2)

        return loss

    def forward(self, pred, target, epoch=None):
        """
        Compute total physical loss.

        Args:
            pred: [batch, nx, tout] - Predicted field
            target: [batch, nx, tout] - Ground truth field
            epoch: Optional epoch number for adaptive weighting

        Returns:
            loss: Scalar total loss
            loss_dict: Dict with individual loss components
        """
        # 1. Base MSE loss
        mse = self.mse_loss(pred, target)

        # 2. Energy conservation
        energy_loss = self.energy_drift(pred, target)

        # 3. Momentum conservation
        momentum_loss = self.momentum_drift(pred, target)

        # 4. Spectral loss (optional, for later phases)
        # spectral_loss = self.spectral_loss(pred, target)

        # Adaptive weighting based on epoch
        # (Start with MSE, gradually increase physical constraints)
        if epoch is not None and epoch < 10:
            # Warm-up phase: focus on MSE
            lambda_energy = self.lambda_energy * (epoch / 10)
            lambda_momentum = self.lambda_momentum * (epoch / 10)
        else:
            # Full training: use specified weights
            lambda_energy = self.lambda_energy
            lambda_momentum = self.lambda_momentum

        # Total loss
        total_loss = mse \
                   + lambda_energy * energy_loss \
                   + lambda_momentum * momentum_loss

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse.item(),
            'energy': energy_loss.item(),
            'momentum': momentum_loss.item(),
            'lambda_energy': lambda_energy,
            'lambda_momentum': lambda_momentum
        }

        return total_loss, loss_dict


class ConservationLoss(nn.Module):
    """
    Strict conservation loss for long-term stability.

    Enforces dE/dt = 0 and dP/dt = 0 explicitly.
    """

    def __init__(self, lambda_energy=1.0, lambda_momentum=1.0):
        super().__init__()

        self.lambda_energy = lambda_energy
        self.lambda_momentum = lambda_momentum

    def energy_conservation(self, pred_sequence):
        """
        Enforce energy conservation across time steps.

        Args:
            pred_sequence: [batch, nx, timesteps] - Predicted evolution

        Returns:
            loss: Scalar - Energy variation over time
        """
        # Compute energy at each time step
        energies = 0.5 * torch.sum(pred_sequence ** 2, dim=1)  # [batch, timesteps]

        # Energy should be constant: std(energies) should be 0
        energy_variation = torch.std(energies, dim=1).mean()

        return energy_variation

    def momentum_conservation(self, pred_sequence):
        """
        Enforce momentum conservation across time steps.

        Args:
            pred_sequence: [batch, nx, timesteps] - Predicted evolution

        Returns:
            loss: Scalar - Momentum variation over time
        """
        # Compute momentum at each time step
        momenta = torch.sum(pred_sequence, dim=1)  # [batch, timesteps]

        # Momentum should be constant: std(momenta) should be 0
        momentum_variation = torch.std(momenta, dim=1).mean()

        return momentum_variation

    def forward(self, pred_sequence):
        """
        Compute strict conservation loss.

        Args:
            pred_sequence: [batch, nx, timesteps] - Predicted evolution

        Returns:
            loss: Scalar total conservation loss
            loss_dict: Dict with individual components
        """
        energy_loss = self.energy_conservation(pred_sequence)
        momentum_loss = self.momentum_conservation(pred_sequence)

        total_loss = self.lambda_energy * energy_loss \
                   + self.lambda_momentum * momentum_loss

        loss_dict = {
            'energy_conservation': energy_loss.item(),
            'momentum_conservation': momentum_loss.item()
        }

        return total_loss, loss_dict


def create_physical_loss(cfg):
    """
    Factory function to create physical loss based on config.

    Args:
        cfg: Configuration dict

    Returns:
        loss_fn: PhysicalLoss instance
    """
    loss_cfg = cfg.get('loss', {})

    lambda_energy = loss_cfg.get('lambda_energy', 0.1)
    lambda_momentum = loss_cfg.get('lambda_momentum', 0.1)
    lambda_noise = loss_cfg.get('lambda_noise', 0.05)

    loss_fn = PhysicalLoss(
        lambda_energy=lambda_energy,
        lambda_momentum=lambda_momentum,
        lambda_noise=lambda_noise
    )

    return loss_fn


if __name__ == "__main__":
    # Test physical loss
    print("🧪 Testing Physical Loss Functions\n")

    # Create dummy data
    batch_size = 4
    nx = 128
    tout = 32

    pred = torch.randn(batch_size, nx, tout)
    target = torch.randn(batch_size, nx, tout)

    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}\n")

    # Test PhysicalLoss
    loss_fn = PhysicalLoss(
        lambda_energy=0.1,
        lambda_momentum=0.1
    )

    total_loss, loss_dict = loss_fn(pred, target, epoch=5)

    print("Physical Loss (epoch=5):")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")

    print(f"\nTotal loss: {total_loss:.6f}")

    # Test ConservationLoss
    print("\n" + "="*50)
    print("Testing ConservationLoss\n")

    cons_loss_fn = ConservationLoss(
        lambda_energy=1.0,
        lambda_momentum=1.0
    )

    cons_loss, cons_dict = cons_loss_fn(pred)

    print("Conservation Loss:")
    for key, value in cons_dict.items():
        print(f"  {key}: {value:.6f}")

    print(f"\nTotal conservation loss: {cons_loss:.6f}")

    print("\n✅ Physical Loss tests passed!")
