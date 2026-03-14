"""
Baseline A: Simple MLP Predictor
==================================

A minimal baseline that flattens spatiotemporal input and predicts future field.
This represents a "dumb" approach with no inductive bias for spatiotemporal structure.

Design choice:
- Flatten [B, Tin, X] → [B, Tin*X]
- Process through FC layers
- Reshape to [B, Tout, X]

Expected behavior:
- Should fit short-term patterns
- Likely to struggle with long-term rollout stability
- No explicit conservation or spatial inductive bias

Author: SRΨ-Engine Tiny Experiment
"""

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Simple MLP baseline for field evolution prediction.

    Flattens spatiotemporal input and predicts flattened output,
    then reshapes back to [B, Tout, X].
    """

    def __init__(
        self,
        tin: int,
        tout: int,
        nx: int,
        hidden_dim: int = 256
    ):
        """
        Args:
            tin: Input time steps
            tout: Output time steps
            nx: Spatial resolution
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.tin = tin
        self.tout = tout
        self.nx = nx

        in_dim = tin * nx
        out_dim = tout * nx

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input field [B, Tin, X]

        Returns:
            y: Predicted future field [B, Tout, X]
        """
        b = x.shape[0]

        # Flatten spatiotemporal input
        z = x.reshape(b, -1)  # [B, Tin*X]

        # Process through MLP
        y = self.net(z)  # [B, Tout*X]

        # Reshape to spatiotemporal output
        y = y.reshape(b, self.tout, self.nx)  # [B, Tout, X]

        return y


if __name__ == "__main__":
    # Test model
    tin, tout, nx = 16, 32, 128
    batch_size = 4

    model = BaselineMLP(tin, tout, nx, hidden_dim=256)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"BaselineMLP parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(batch_size, tin, nx)
    y = model(x)

    print(f"Input shape:  {x.shape}")   # [4, 16, 128]
    print(f"Output shape: {y.shape}")   # [4, 32, 128]

    # Test backward pass
    loss = y.mean()
    loss.backward()

    print("\n✓ BaselineMLP test passed")
