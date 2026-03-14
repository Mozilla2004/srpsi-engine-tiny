"""
SRΨ-Engine Real-only: Complex-free Architecture for Ablation Study
================================================================

Ablation Study Experiment 2: SRΨ without Complex-valued State

Key differences from full SRΨ:
- Uses single-channel real-valued state [B, X, D] instead of [B, X, D, 2]
- Removes imaginary component from all operators
- Tests whether complex representation is necessary for performance

Purpose: Isolate the contribution of phase representation

Author: SRΨ-Engine Ablation Study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _init_weights(module):
    """Apply Xavier initialization with small gain for stability"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight, gain=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class InputEncoderReal(nn.Module):
    """
    Encode historical field [B, Tin, X] → initial real state ψ₀ [B, X, D].

    Design:
    - Each spatial position independently processes temporal history
    - Output is single-channel (no complex representation)
    """

    def __init__(self, tin: int, nx: int, hidden_dim: int):
        super().__init__()
        self.tin = tin
        self.nx = nx
        self.hidden_dim = hidden_dim

        # Process temporal history at each spatial location
        self.net = nn.Sequential(
            nn.Linear(tin, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),  # Single channel (not dual)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input field [B, Tin, X]

        Returns:
            psi: Initial real state [B, X, D]
        """
        # Normalize input (critical for stability)
        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, X]
        x_std = x.std(dim=1, keepdim=True) + 1e-6  # [B, 1, X]
        x = (x - x_mean) / x_std

        # Transpose: [B, Tin, X] -> [B, X, Tin]
        x = x.transpose(1, 2)

        # Process: [B, X, Tin] -> [B, X, D]
        z = self.net(x)

        # Clip to prevent explosion
        z = torch.clamp(z, min=-10.0, max=10.0)

        return z  # [B, X, D] (single channel)


class StructureOperatorS(nn.Module):
    """
    Structure Operator S: Local spatial coupling (real-valued).

    Design:
    - Depthwise + pointwise convolution for local structure
    - Operates on single channel
    - Represents spatial neighborhood interactions
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2

        # Single convolution for real-valued state
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)

        # Channel mixing
        self.mix = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Real state [B, X, D]

        Returns:
            dpsi: Structure update [B, X, D]
        """
        # Transpose for conv: [B, X, D] -> [B, D, X]
        psi_conv = psi.transpose(1, 2)

        # Apply convolution
        out = self.conv(psi_conv).transpose(1, 2)  # [B, X, D]

        # Channel mixing
        out = self.mix(out)

        return out


class RhythmOperatorR(nn.Module):
    """
    Rhythm Operator R: Local oscillation (real-valued approximation).

    Design:
    - Predicts local modulation coefficient m(x, d)
    - Applies modulation: ψ → m ⊙ ψ
    - Represents oscillatory dynamics (real approximation)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Modulation gate: predicts local coefficient
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Bounded modulation
        )

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Real state [B, X, D]

        Returns:
            dpsi: Rhythm update [B, X, D]
        """
        # Compute modulation gate
        m = self.gate(psi)  # [B, X, D]

        # Apply modulation
        return m * psi


class NonlinearOperatorN(nn.Module):
    """
    Nonlinear Operator N: Nonlinear modulation (real-valued).

    Design:
    - Standard MLP with gating
    - Operates on real-valued features
    - Represents nonlinear interactions
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Real state [B, X, D]

        Returns:
            dpsi: Nonlinear update [B, X, D]
        """
        # Nonlinear transformation
        h = F.gelu(self.fc1(psi))
        h = self.fc2(h)

        # Gate
        g = self.gate(psi)  # [B, X, D]

        return g * h


class StableProjector(nn.Module):
    """
    Stable Projector Φ: Energy/normalization control (real-valued).

    Design:
    - LayerNorm-style stabilization per spatial location
    - Prevents numerical explosion while preserving relative structure
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_dim = hidden_dim

        # LayerNorm over the channel dimension
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Real state [B, X, D]

        Returns:
            psi: Normalized state [B, X, D]
        """
        # Apply LayerNorm
        psi_normed = self.norm(psi)

        return psi_normed


class SRPsiBlockReal(nn.Module):
    """
    Single SRΨ block (real-valued): Composition of S, R, N, Φ.

    Evolution step:
        Δψ = S(ψ) + R(ψ) + N(ψ)
        ψ_{next} = ψ + dt · Δψ
        ψ_{next} = Φ(ψ_{next})
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 5, dt: float = 0.01):
        super().__init__()

        self.S = StructureOperatorS(hidden_dim, kernel_size)
        self.R = RhythmOperatorR(hidden_dim)
        self.N = NonlinearOperatorN(hidden_dim)
        self.P = StableProjector(hidden_dim)

        self.dt = dt

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Current state [B, X, D]

        Returns:
            psi_next: Next state [B, X, D]
        """
        # Compute update from all operators
        delta = self.S(psi) + self.R(psi) + self.N(psi)

        # Clip delta to prevent explosion
        delta = torch.clamp(delta, min=-5.0, max=5.0)

        # Euler integration
        psi_next = psi + self.dt * delta

        # Clip intermediate state
        psi_next = torch.clamp(psi_next, min=-10.0, max=10.0)

        # Stable projection
        psi_next = self.P(psi_next)

        return psi_next


class OutputDecoderReal(nn.Module):
    """
    Decode real state ψ [B, X, D] → real field [B, X].

    Design:
    - Project to single value per spatial location
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Real state [B, X, D]

        Returns:
            out: Real field [B, X]
        """
        # Project to scalar
        out = self.head(psi).squeeze(-1)  # [B, X]

        return out


class SRPsiEngineReal(nn.Module):
    """
    SRΨ-Engine Real-only: Complete model for field evolution prediction.

    Architecture:
        1. InputEncoder: history → ψ₀ (real)
        2. Autoregressive rollout: repeat K steps
           - Each step: apply D SRΨ blocks
           - Decode each intermediate state
        3. Collect predictions into [B, Tout, X]

    Ablation Study: Experiment 2
    Tests whether complex-valued representation is necessary
    """

    def __init__(
        self,
        tin: int,
        nx: int,
        hidden_dim: int = 64,
        depth: int = 3,
        kernel_size: int = 5,
        dt: float = 0.01,
        tout: int = 32
    ):
        """
        Args:
            tin: Input time steps
            nx: Spatial resolution
            hidden_dim: Hidden dimension
            depth: Number of SRΨ blocks per step
            kernel_size: Convolution kernel size
            dt: Integration time step
            tout: Output time steps
        """
        super().__init__()

        self.tin = tin
        self.nx = nx
        self.tout = tout
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Encoder: [B, Tin, X] → [B, X, D]
        self.encoder = InputEncoderReal(tin, nx, hidden_dim)

        # SRΨ blocks
        self.blocks = nn.ModuleList([
            SRPsiBlockReal(hidden_dim, kernel_size, dt) for _ in range(depth)
        ])

        # Decoder: [B, X, D] → [B, X]
        self.decoder = OutputDecoderReal(hidden_dim)

        # Apply weight initialization
        self.apply(_init_weights)

    def step(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Apply one full evolution step (all blocks).

        Args:
            psi: Current state [B, X, D]

        Returns:
            psi_next: Next state [B, X, D]
        """
        for blk in self.blocks:
            psi = blk(psi)
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout prediction.

        Args:
            x: Input history [B, Tin, X]

        Returns:
            y: Predicted future [B, Tout, X]
        """
        # Encode initial state
        psi = self.encoder(x)  # [B, X, D]

        # Autoregressive rollout
        preds = []
        for _ in range(self.tout):
            psi = self.step(psi)
            y = self.decoder(psi)
            preds.append(y)

        # Stack predictions
        return torch.stack(preds, dim=1)  # [B, Tout, X]


if __name__ == "__main__":
    # Test model
    tin, tout, nx = 16, 32, 128
    batch_size = 4

    model = SRPsiEngineReal(
        tin=tin,
        nx=nx,
        hidden_dim=64,
        depth=3,
        kernel_size=5,
        dt=0.01,
        tout=tout
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"SRPsiEngineReal parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(batch_size, tin, nx)
    y = model(x)

    print(f"Input shape:  {x.shape}")   # [4, 16, 128]
    print(f"Output shape: {y.shape}")   # [4, 32, 128]

    # Test backward pass
    loss = y.mean()
    loss.backward()

    # Test intermediate state inspection
    psi = model.encoder(x)
    print(f"Psi shape:    {psi.shape}")  # [4, 128, 64]
    print(f"Psi norm:     {psi.norm():.4f}")

    print("\n✓ SRPsiEngineReal test passed")
