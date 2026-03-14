"""
Transformer with Relative Position Encoding for Ablation Study
=============================================================

Ablation Study Experiment 5: Transformer without Absolute PE

Key differences from baseline Transformer:
- Removes absolute positional encoding
- Uses relative position bias in attention
- Fairer comparison to SRΨ (which has translation equivariance)

Purpose: Test if relative position attention improves shift robustness

Author: SRΨ-Engine Ablation Study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _init_weights(module):
    """Apply Xavier initialization"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class RelativePositionBias(nn.Module):
    """
    Learnable relative position bias.

    Generates bias for attention based on relative distance between tokens.
    """

    def __init__(self, num_heads: int, max_relative_position: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position

        # Learnable bias for each relative distance
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * max_relative_position - 1, num_heads)
        )

        # Initialize with small values
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Generate relative position bias matrix.

        Args:
            seq_len: Sequence length

        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # Generate relative position indices
        range_vec = torch.arange(seq_len)
        distance_mat = range_vec[None, :] - range_vec[:, None]  # [seq_len, seq_len]

        # Clip to max distance
        distance_mat = torch.clamp(
            distance_mat,
            -self.max_relative_position + 1,
            self.max_relative_position - 1
        )

        # Shift to [0, 2*max-2]
        distance_mat = distance_mat + self.max_relative_position - 1

        # Convert to long for indexing
        distance_mat = distance_mat.long()

        # Gather biases
        bias = self.relative_position_bias[distance_mat]  # [seq_len, seq_len, num_heads]
        bias = bias.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]

        return bias


class RelativePositionMultiheadAttention(nn.Module):
    """
    Multi-head attention with relative position bias.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Relative position bias
        self.relative_bias = RelativePositionBias(nhead)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, D]

        Returns:
            out: Output [B, T, D]
        """
        B, T, D = x.shape

        # Project Q, K, V
        Q = self.q_proj(x)  # [B, T, D]
        K = self.k_proj(x)  # [B, T, D]
        V = self.v_proj(x)  # [B, T, D]

        # Reshape for multi-head
        Q = Q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, nhead, T, head_dim]
        K = K.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nhead, T, T]

        # Add relative position bias
        bias = self.relative_bias(T).to(scores.device)  # [nhead, T, T]
        scores = scores + bias.unsqueeze(0)  # [B, nhead, T, T]

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to V
        attn_output = torch.matmul(attn_weights, V)  # [B, nhead, T, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        output = self.out_proj(attn_output)

        return output


class TransformerRelPE(nn.Module):
    """
    Transformer with Relative Position Encoding.

    Removes absolute positional encoding, uses relative position bias instead.
    """

    def __init__(
        self,
        tin: int,
        nx: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
        tout: int = 32
    ):
        super().__init__()

        self.tin = tin
        self.tout = tout
        self.nx = nx

        # Input projection
        self.input_projection = nn.Linear(tin, d_model)

        # Transformer encoder layers with relative position attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        # Replace self-attention with relative position version
        self.transformer_layers = nn.ModuleList([])
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': RelativePositionMultiheadAttention(d_model, nhead, dropout),
                'norm2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                )
            })
            self.transformer_layers.append(layer)

        # Output projection (d_model -> 1 for prediction)
        self.output_projection = nn.Linear(d_model, 1)

        # State update projection (1 -> d_model for autoregressive)
        self.state_projection = nn.Linear(1, d_model)

        # Apply initialization
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive rollout prediction.

        Args:
            x: Input history [B, Tin, X]

        Returns:
            y: Predicted future [B, Tout, X]
        """
        B, Tin, X = x.shape
        D = self.input_projection.out_features

        # Project input: [B, Tin, X] → [B, X, Tin] → [B, X, D_model]
        h = self.input_projection(x.transpose(1, 2))  # [B, X, D]

        # Normalize
        h_mean = h.mean(dim=-1, keepdim=True)
        h_std = h.std(dim=-1, keepdim=True) + 1e-6
        h = (h - h_mean) / h_std

        # Autoregressive rollout
        preds = []
        for _ in range(self.tout):
            # Apply transformer layers
            for layer in self.transformer_layers:
                # Self-attention with residual
                h_norm = layer['norm1'](h)
                h_attn = layer['attn'](h_norm)
                h = h + h_attn

                # Feed-forward with residual
                h_norm = layer['norm2'](h)
                h_ff = layer['ff'](h_norm)
                h = h + h_ff

                # Clip for stability
                h = torch.clamp(h, min=-10.0, max=10.0)

            # Take mean over hidden dim to get [B, X]
            out = self.output_projection(h).squeeze(-1)  # [B, X]
            preds.append(out)

            # Update h (shift): Project out back to D and update
            # out: [B, X] -> [B, X, 1] -> [B, X, D]
            out_state = self.state_projection(out.unsqueeze(-1))  # [B, X, D]
            
            # This model is spatially local, so we don't have a "time" dimension in h
            # instead h is [B, X, D]. To simulate autoregressive behavior in this
            # specific architecture, we update the state h.
            h = out_state

        # Final reshape: [B, Tout, X] -> [B, Tout, X]
        # (preds is a list of [B, X] tensors, torch.stack makes it [B, Tout, X])
        return torch.stack(preds, dim=1)  # [B, Tout, X]


if __name__ == "__main__":
    # Test model
    tin, tout, nx = 16, 32, 128
    batch_size = 4

    model = TransformerRelPE(
        tin=tin,
        nx=nx,
        d_model=64,
        nhead=4,
        num_layers=3,
        dropout=0.0,
        tout=tout
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"TransformerRelPE parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(batch_size, tin, nx)
    y = model(x)

    print(f"Input shape:  {x.shape}")   # [4, 16, 128]
    print(f"Output shape: {y.shape}")   # [4, 32, 128]

    # Test backward pass
    loss = y.mean()
    loss.backward()

    print("\n✓ TransformerRelPE test passed")
