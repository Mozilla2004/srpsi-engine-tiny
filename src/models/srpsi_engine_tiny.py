"""
SRΨ-Engine Tiny: Dynamics-Oriented Field Evolution Model
==========================================================

Core innovation: Encode evolution operator as composition of structure, rhythm, and stability.
Represents state as complex-valued field using dual real channels.

Architecture:
    Input field [B, Tin, X]
        ↓ InputEncoder
    ψ₀ [B, X, D, 2]  (complex-valued: real + imag channels)
        ↓ SRΨ block repeated K times
    ψₖ [B, X, D, 2]
        ↓ OutputDecoder
    Predicted field [B, Tout, X]

Single SRΨ block:
    Δψ = S(ψ) + R(ψ) + N(ψ)  # structure + rhythm + nonlinear
    ψ_{t+1} = ψ_t + dt · Δψ
    ψ_{t+1} = Φ(ψ_{t+1})     # stable projection

Key design principles:
- S: Local spatial coupling (structure)
- R: Local phase rotation (rhythm)
- N: Nonlinear modulation
- Φ: Stable projection (energy control)

Expected behavior:
- Better long-term rollout stability
- Controlled energy drift
- Shift/phase robustness
- Recovery from small perturbations

Author: SRΨ-Engine Tiny Experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _init_weights(module):
    """
    Apply Xavier initialization to linear and conv layers
    for better numerical stability.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.01)  # Very small gain for stability
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight, gain=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class InputEncoder(nn.Module):
    """
    Encode historical field [B, Tin, X] → initial complex state ψ₀ [B, X, D, 2].

    Design:
    - Each spatial position independently processes temporal history
    - Output is dual-channel (real, imag) representing complex values
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
            nn.Linear(hidden_dim, hidden_dim * 2),  # Split into real + imag
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input field [B, Tin, X]

        Returns:
            psi: Initial complex state [B, X, D, 2]
        """
        # Normalize input (critical for stability)
        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, X]
        x_std = x.std(dim=1, keepdim=True) + 1e-6  # [B, 1, X]
        x = (x - x_mean) / x_std

        # Transpose: [B, Tin, X] -> [B, X, Tin]
        x = x.transpose(1, 2)

        # Process: [B, X, Tin] -> [B, X, 2*D]
        z = self.net(x)

        # Clip to prevent explosion
        z = torch.clamp(z, min=-10.0, max=10.0)

        # Split into real and imag channels
        real, imag = torch.chunk(z, 2, dim=-1)

        # Stack: [B, X, D, 2]
        psi = torch.stack([real, imag], dim=-1)

        return psi


class StructureOperatorS(nn.Module):
    """
    Structure Operator S: Local spatial coupling.

    Design:
    - Depthwise + pointwise convolution for local structure
    - Operates independently on real and imag channels
    - Represents spatial neighborhood interactions
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2

        # Separate convolutions for real and imag
        self.real_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.imag_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)

        # Channel mixing
        self.mix = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Complex state [B, X, D, 2]

        Returns:
            dpsi: Structure update [B, X, D, 2]
        """
        # Extract channels
        real = psi[..., 0].transpose(1, 2)  # [B, D, X]
        imag = psi[..., 1].transpose(1, 2)

        # Apply convolutions
        real_out = self.real_conv(real).transpose(1, 2)  # [B, X, D]
        imag_out = self.imag_conv(imag).transpose(1, 2)

        # Channel mixing
        real_out = self.mix(real_out)
        imag_out = self.mix(imag_out)

        # Stack back
        return torch.stack([real_out, imag_out], dim=-1)


class RhythmOperatorR(nn.Module):
    """
    Rhythm Operator R: Local phase rotation.

    Design:
    - Predicts local phase shift θ(x, d)
    - Applies approximate rotation: ψ → θ · (-imag, real)
    - Represents oscillatory/rhythmic dynamics
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Phase gate: predicts local rotation angle
        self.phase_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Bounded phase shift
        )

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Complex state [B, X, D, 2]

        Returns:
            dpsi: Rhythm update [B, X, D, 2]
        """
        # Extract real and imag
        real = psi[..., 0]  # [B, X, D]
        imag = psi[..., 1]

        # Compute phase gate
        feat = torch.cat([real, imag], dim=-1)  # [B, X, 2D]
        theta = self.phase_gate(feat)           # [B, X, D]

        # Apply approximate rotation: J(ψ) = (-imag, real)
        r_real = -theta * imag
        r_imag =  theta * real

        return torch.stack([r_real, r_imag], dim=-1)


class NonlinearOperatorN(nn.Module):
    """
    Nonlinear Operator N: Nonlinear modulation.

    Design:
    - Standard MLP with gating
    - Operates on concatenated real+imag features
    - Represents nonlinear interactions
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Complex state [B, X, D, 2]

        Returns:
            dpsi: Nonlinear update [B, X, D, 2]
        """
        real = psi[..., 0]
        imag = psi[..., 1]
        feat = torch.cat([real, imag], dim=-1)  # [B, X, 2D]

        # Nonlinear transformation
        h = F.gelu(self.fc1(feat))
        h = self.fc2(h)

        # Split and gate
        h_real, h_imag = torch.chunk(h, 2, dim=-1)
        g = self.gate(feat)  # [B, X, D]

        h_real = g * h_real
        h_imag = g * h_imag

        return torch.stack([h_real, h_imag], dim=-1)


class StableProjector(nn.Module):
    """
    Stable Projector Φ: Energy/normalization control (FIXED VERSION).

    Design:
    - LayerNorm-style stabilization per spatial location
    - Prevents numerical explosion while preserving relative structure
    - Ensures numerical stability during rollout

    FIX: Changed from global norm clipping to per-location LayerNorm
    Reason: Global norm was causing NaN due to division issues
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_dim = hidden_dim

        # LayerNorm over the complex channel dimension
        # Normalizes [B, X, D*2] to have zero mean and unit variance
        self.norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Complex state [B, X, D, 2]

        Returns:
            psi: Normalized state [B, X, D, 2]
        """
        # Reshape: [B, X, D, 2] -> [B, X, D*2]
        b, x, d, c = psi.shape
        psi_flat = psi.reshape(b, x, d * c)

        # Apply LayerNorm (stable, prevents explosion)
        psi_flat = self.norm(psi_flat)

        # Reshape back: [B, X, D*2] -> [B, X, D, 2]
        return psi_flat.reshape(b, x, d, c)


class SRPsiBlock(nn.Module):
    """
    Single SRΨ block: Composition of S, R, N, Φ.

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
        self.P = StableProjector(hidden_dim)  # Pass hidden_dim

        self.dt = dt

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Current state [B, X, D, 2]

        Returns:
            psi_next: Next state [B, X, D, 2]
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


class OutputDecoder(nn.Module):
    """
    Decode complex state ψ [B, X, D, 2] → real field [B, X].

    Design:
    - Concatenate real + imag channels
    - Project to single value per spatial location
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Complex state [B, X, D, 2]

        Returns:
            out: Real field [B, X]
        """
        # Concatenate real and imag
        real = psi[..., 0]
        imag = psi[..., 1]
        feat = torch.cat([real, imag], dim=-1)  # [B, X, 2D]

        # Project to scalar
        out = self.head(feat).squeeze(-1)  # [B, X]

        return out


class SRPsiEngineTiny(nn.Module):
    """
    SRΨ-Engine Tiny: Complete model for field evolution prediction.

    Architecture:
        1. InputEncoder: history → ψ₀
        2. Autoregressive rollout: repeat K steps
           - Each step: apply D SRΨ blocks
           - Decode each intermediate state
        3. Collect predictions into [B, Tout, X]
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
            dt: Integration time step (FIXED: 0.01 for stability)
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

        # SRΨ blocks
        self.blocks = nn.ModuleList([
            SRPsiBlock(hidden_dim, kernel_size, dt) for _ in range(depth)
        ])

        # Decoder: [B, X, D, 2] → [B, X]
        self.decoder = OutputDecoder(hidden_dim)

        # Apply weight initialization for stability
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

    model = SRPsiEngineTiny(
        tin=tin,
        nx=nx,
        hidden_dim=64,
        depth=3,
        kernel_size=5,
        dt=0.1,
        tout=tout
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"SRPsiEngineTiny parameters: {num_params:,}")

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

    print("\n✓ SRPsiEngineTiny test passed")
