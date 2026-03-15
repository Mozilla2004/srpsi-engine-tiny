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
    'get_model',  # Factory function
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
