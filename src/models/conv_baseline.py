"""
Convolution Baseline: Pure Conv Architecture for Ablation Study
==============================================================

Ablation Study Experiment 4: Pure Convolution Baseline

Key differences:
- No complex-valued state
- No rhythm operator
- Simple depthwise conv + MLP
- Tests whether convolution alone provides benefits

Purpose: Establish baseline for convolution bias

Author: SRΨ-Engine Ablation Study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module):
    """Apply Xavier initialization"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight, gain=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ConvBaseline(nn.Module):
    """
    Pure Convolution Baseline: Simple conv architecture

    Design:
    - Input encoder: Linear projection
    - Conv layers: Depthwise conv + MLP
    - Autoregressive rollout
    - Output decoder: Linear projection

    Purpose: Establish if convolution bias alone provides benefits
    """

    def __init__(
        self,
        tin: int,
        nx: int,
        hidden_dim: int = 64,
        depth: int = 3,
        kernel_size: int = 5,
        tout: int = 32
    ):
        """
        Args:
            tin: Input time steps
            nx: Spatial resolution
            hidden_dim: Hidden dimension
            depth: Number of conv layers
            kernel_size: Convolution kernel size
            tout: Output time steps
        """
        super().__init__()

        self.tin = tin
        self.nx = nx
        self.tout = tout
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Encoder: [B, Tin, X] → [B, X, D]
        self.encoder = nn.Linear(tin, hidden_dim)

        # Conv layers
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
            for _ in range(depth)
        ])

        # MLP after conv
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(depth)
        ])

        # Layer normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(depth)
        ])

        # Decoder: [B, X, D] → [B, X]
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # Apply initialization
        self.apply(_init_weights)

    def step(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Apply one evolution step (all conv layers).

        Args:
            psi: Current state [B, X, D]

        Returns:
            psi_next: Next state [B, X, D]
        """
        for conv, mlp, norm in zip(self.convs, self.mlps, self.norms):
            # Conv: [B, X, D] → [B, D, X] → [B, D, X] → [B, X, D]
            psi_conv = conv(psi.transpose(1, 2)).transpose(1, 2)

            # MLP + Residual
            psi_mlp = mlp(psi_conv)

            # Residual connection + normalization
            psi = norm(psi + psi_mlp)

            # Clip for stability
            psi = torch.clamp(psi, min=-10.0, max=10.0)

        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout prediction.

        Args:
            x: Input history [B, Tin, X]

        Returns:
            y: Predicted future [B, Tout, X]
        """
        # Encode: [B, Tin, X] → [B, X, Tin] → [B, X, D]
        psi = self.encoder(x.transpose(1, 2))

        # Normalize
        psi_mean = psi.mean(dim=-1, keepdim=True)
        psi_std = psi.std(dim=-1, keepdim=True) + 1e-6
        psi = (psi - psi_mean) / psi_std

        # Autoregressive rollout
        preds = []
        for _ in range(self.tout):
            psi = self.step(psi)
            y = self.decoder(psi).squeeze(-1)  # [B, X]
            preds.append(y)

        # Stack predictions
        return torch.stack(preds, dim=1)  # [B, Tout, X]


if __name__ == "__main__":
    # Test model
    tin, tout, nx = 16, 32, 128
    batch_size = 4

    model = ConvBaseline(
        tin=tin,
        nx=nx,
        hidden_dim=64,
        depth=3,
        kernel_size=5,
        tout=tout
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"ConvBaseline parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(batch_size, tin, nx)
    y = model(x)

    print(f"Input shape:  {x.shape}")   # [4, 16, 128]
    print(f"Output shape: {y.shape}")   # [4, 32, 128]

    # Test backward pass
    loss = y.mean()
    loss.backward()

    print("\n✓ ConvBaseline test passed")
