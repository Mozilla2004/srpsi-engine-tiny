"""
SRΨ-Engine w/o R: Architecture without Rhythm Operator
========================================================

Ablation Study Experiment 3: SRΨ without Rhythm Operator

Key differences from full SRΨ:
- Removes Rhythm Operator (R) from SRΨ blocks
- Keeps Complex-valued state
- Tests independent contribution of phase rotation

Purpose: Isolate the contribution of Rhythm Operator

Author: SRΨ-Engine Ablation Study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import shared components from full SRΨ
import sys
sys.path.append('src/models')
from srpsi_engine_tiny import (
    _init_weights,
    InputEncoder,
    StructureOperatorS,
    NonlinearOperatorN,
    StableProjector,
    OutputDecoder
)


class SRPsiBlockNoR(nn.Module):
    """
    Single SRΨ block without Rhythm Operator: S + N + Φ only.

    Evolution step:
        Δψ = S(ψ) + N(ψ)  # No Rhythm Operator
        ψ_{next} = ψ + dt · Δψ
        ψ_{next} = Φ(ψ_{next})

    Purpose: Test independent contribution of R operator
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 5, dt: float = 0.01):
        super().__init__()

        self.S = StructureOperatorS(hidden_dim, kernel_size)
        # No Rhythm Operator (R)
        self.N = NonlinearOperatorN(hidden_dim)
        self.P = StableProjector(hidden_dim)

        self.dt = dt

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Current state [B, X, D, 2]

        Returns:
            psi_next: Next state [B, X, D, 2]
        """
        # Compute update from S and N only (no R)
        delta = self.S(psi) + self.N(psi)

        # Clip delta to prevent explosion
        delta = torch.clamp(delta, min=-5.0, max=5.0)

        # Euler integration
        psi_next = psi + self.dt * delta

        # Clip intermediate state
        psi_next = torch.clamp(psi_next, min=-10.0, max=10.0)

        # Stable projection
        psi_next = self.P(psi_next)

        return psi_next


class SRPsiEngineNoR(nn.Module):
    """
    SRΨ-Engine without Rhythm Operator: Complete model.

    Architecture:
        1. InputEncoder: history → ψ₀
        2. Autoregressive rollout: repeat K steps
           - Each step: apply D SRΨ blocks (no R)
           - Decode each intermediate state
        3. Collect predictions into [B, Tout, X]

    Ablation Study: Experiment 3
    Tests independent contribution of Rhythm Operator
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

        # Encoder: [B, Tin, X] → [B, X, D, 2]
        self.encoder = InputEncoder(tin, nx, hidden_dim)

        # SRΨ blocks without R
        self.blocks = nn.ModuleList([
            SRPsiBlockNoR(hidden_dim, kernel_size, dt) for _ in range(depth)
        ])

        # Decoder: [B, X, D, 2] → [B, X]
        self.decoder = OutputDecoder(hidden_dim)

        # Apply weight initialization
        self.apply(_init_weights)

    def step(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Apply one full evolution step (all blocks).

        Args:
            psi: Current state [B, X, D, 2]

        Returns:
            psi_next: Next state [B, X, D, 2]
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
        psi = self.encoder(x)  # [B, X, D, 2]

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

    model = SRPsiEngineNoR(
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
    print(f"SRPsiEngineNoR parameters: {num_params:,}")

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
    print(f"Psi shape:    {psi.shape}")  # [4, 128, 64, 2]
    print(f"Psi norm:     {psi.norm():.4f}")

    print("\n✓ SRPsiEngineNoR test passed")
