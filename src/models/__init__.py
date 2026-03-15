"""
Model Zoo for SRΨ-Engine Tiny Experiments
==========================================

Available models:
- BaselineMLP: Simple flatten-MLP baseline
- BaselineTransformer: Transformer-style sequence model
- SRPsiEngineTiny: Dynamics-oriented SRΨ-Engine

Ablation Study Models:
- SRPsiEngineReal: SRΨ without complex state (Exp2)
- SRPsiEngineNoR: SRΨ without rhythm operator (Exp3)
- ConvBaseline: Pure convolution baseline (Exp4)
- TransformerRelPE: Transformer with relative position encoding (Exp5)

Author: SRΨ-Engine Tiny Experiment
"""

import torch

from .baseline_mlp import BaselineMLP
from .baseline_transformer import BaselineTransformer
from .srpsi_engine_tiny import SRPsiEngineTiny

# Ablation study models
from .srpsi_real import SRPsiEngineReal
from .srpsi_no_r import SRPsiEngineNoR
from .conv_baseline import ConvBaseline
from .transformer_rel_pe import TransformerRelPE

__all__ = [
    'BaselineMLP',
    'BaselineTransformer',
    'SRPsiEngineTiny',
    # Ablation models
    'SRPsiEngineReal',
    'SRPsiEngineNoR',
    'ConvBaseline',
    'TransformerRelPE',
    'get_model',  # Get model class
    'create_model',  # Create model instance
]


def get_model(model_type: str):
    """
    Factory function to get model class by type string.

    Args:
        model_type: String identifier for model type
            - 'conv_baseline': ConvBaseline class
            - 'transformer_rel_pe': TransformerRelPE class
            - 'srpsi_real': SRPsiEngineReal class
            - 'srpsi_no_r': SRPsiEngineNoR class
            - 'srpsi_tiny': SRPsiEngineTiny class

    Returns:
        Model class (not instantiated)

    Raises:
        ValueError: If model_type is not recognized
    """
    model_map = {
        'conv_baseline': ConvBaseline,
        'transformer_rel_pe': TransformerRelPE,
        'srpsi_real': SRPsiEngineReal,
        'srpsi_no_r': SRPsiEngineNoR,
        'srpsi_tiny': SRPsiEngineTiny,
        'baseline_mlp': BaselineMLP,
        'baseline_transformer': BaselineTransformer,
    }

    model_class = model_map.get(model_type.lower())
    if model_class is None:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available types: {list(model_map.keys())}"
        )

    return model_class


def create_model(model_type: str, cfg: dict, device: torch.device):
    """
    Create model instance based on type and config.

    Args:
        model_type: String identifier for model type
        cfg: Configuration dict (from load_config)
        device: Target device (torch.device)

    Returns:
        model: Model instance (moved to device)

    Raises:
        ValueError: If model_type is not recognized
    """
    import torch.nn as nn

    tin = cfg['task']['tin']
    tout = cfg['task']['tout']
    nx = cfg['task']['nx']
    hidden_dim = cfg['model']['hidden_dim']
    depth = cfg['model']['depth']
    kernel_size = cfg['model']['kernel_size']

    if model_type == 'conv_baseline':
        model = ConvBaseline(
            tin=tin,
            nx=nx,
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            tout=tout
        )

    elif model_type == 'transformer_rel_pe':
        model = TransformerRelPE(
            tin=tin,
            nx=nx,
            d_model=hidden_dim,
            nhead=4,
            num_layers=depth,
            dropout=cfg['model']['dropout'],
            tout=tout
        )

    elif model_type == 'srpsi_real':
        model = SRPsiEngineReal(
            tin=tin,
            nx=nx,
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dt=0.01,
            tout=tout
        )

    elif model_type == 'srpsi_no_r':
        model = SRPsiEngineNoR(
            tin=tin,
            nx=nx,
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dt=0.01,
            tout=tout
        )

    elif model_type == 'srpsi_tiny':
        model = SRPsiEngineTiny(
            tin=tin,
            nx=nx,
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dt=0.01,
            tout=tout
        )

    elif model_type == 'baseline_mlp':
        model = BaselineMLP(tin, tout, nx, hidden_dim=hidden_dim)

    elif model_type == 'baseline_transformer':
        model = BaselineTransformer(
            tin, tout, nx,
            d_model=hidden_dim,
            nhead=4,
            num_layers=depth,
            dropout=cfg['model']['dropout']
        )

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available types: conv_baseline, transformer_rel_pe, srpsi_real, srpsi_no_r, srpsi_tiny"
        )

    return model.to(device)
