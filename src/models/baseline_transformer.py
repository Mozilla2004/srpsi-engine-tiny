"""
Baseline B: Transformer-Style Predictor
========================================

A minimal Transformer baseline that treats each time frame as a token.
Represents mainstream attention-based approach to sequence modeling.

Design choice:
- Each time frame [X] is embedded to [D]
- TransformerEncoder processes temporal sequence
- Learnable query tokens generate future predictions

Expected behavior:
- Should capture temporal dependencies via self-attention
- Strong short-term performance
- May suffer from drift in long rollouts
- No explicit conservation or physics inductive bias

Author: SRΨ-Engine Tiny Experiment
"""

import torch
import torch.nn as nn


class BaselineTransformer(nn.Module):
    """
    Minimal Transformer baseline for field evolution prediction.

    Treats each time frame as a token and applies self-attention.
    Uses learnable query tokens to generate future predictions.
    """

    def __init__(
        self,
        tin: int,
        tout: int,
        nx: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = None,
        dropout: float = 0.0
    ):
        """
        Args:
            tin: Input time steps
            tout: Output time steps
            nx: Spatial resolution
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Feedforward dimension (default: 4*d_model)
            dropout: Dropout rate
        """
        super().__init__()

        self.tin = tin
        self.tout = tout
        self.nx = nx
        self.d_model = d_model

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Input projection: [X] -> [D]
        self.input_proj = nn.Linear(nx, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable query tokens for future prediction
        self.query_tokens = nn.Parameter(torch.randn(1, tout, d_model))

        # Output projection: [D] -> [X]
        self.output_proj = nn.Linear(d_model, nx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input field [B, Tin, X]

        Returns:
            y: Predicted future field [B, Tout, X]
        """
        b = x.shape[0]

        # Encode input frames to tokens
        h = self.input_proj(x)  # [B, Tin, D]

        # Apply transformer encoder
        h = self.encoder(h)     # [B, Tin, D]

        # Expand query tokens for batch
        q = self.query_tokens.expand(b, -1, -1)  # [B, Tout, D]

        # Simple strategy: use last hidden state to condition queries
        # (More sophisticated variants could use cross-attention)
        last_hidden = h[:, -1:, :].expand(-1, self.tout, -1)  # [B, Tout, D]

        # Combine queries with last hidden state
        h_out = q + last_hidden  # [B, Tout, D]

        # Decode to field values
        y = self.output_proj(h_out)  # [B, Tout, X]

        return y


if __name__ == "__main__":
    # Test model
    tin, tout, nx = 16, 32, 128
    batch_size = 4

    model = BaselineTransformer(
        tin, tout, nx,
        d_model=128,
        nhead=4,
        num_layers=3
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"BaselineTransformer parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(batch_size, tin, nx)
    y = model(x)

    print(f"Input shape:  {x.shape}")   # [4, 16, 128]
    print(f"Output shape: {y.shape}")   # [4, 32, 128]

    # Test backward pass
    loss = y.mean()
    loss.backward()

    print("\n✓ BaselineTransformer test passed")
