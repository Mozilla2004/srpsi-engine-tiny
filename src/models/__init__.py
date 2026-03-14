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
]
