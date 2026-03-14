"""
Utility Functions for SRΨ-Engine Tiny
======================================

Common utilities:
- Configuration loading
- Checkpoint management
- Logging setup
- Random seed setting

Author: SRΨ-Engine Tiny Experiment
"""

import yaml
import torch
import random
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file with inheritance support.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'inherit' in cfg:
        inherit_path = os.path.join(os.path.dirname(config_path), cfg['inherit'])
        base_cfg = load_config(inherit_path)
        cfg = merge_configs(base_cfg, cfg)

    return cfg


def merge_configs(base_cfg: Dict, override_cfg: Dict) -> Dict:
    """
    Recursively merge override config into base config.

    Args:
        base_cfg: Base configuration
        override_cfg: Override configuration (takes precedence)

    Returns:
        Merged configuration
    """
    result = base_cfg.copy()

    for key, value in override_cfg.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
):
    """
    Save model checkpoint.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch
        loss: Current loss value
        path: Save path
        **kwargs: Additional metadata to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Load model checkpoint.

    Args:
        path: Checkpoint path
        model: Model instance (weights will be loaded in-place)
        optimizer: Optimizer instance (optional, will be loaded if provided)
        device: Device to load to

    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from: {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown'):.6f}")

    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_cfg: str) -> torch.device:
    """
    Get PyTorch device based on config string.

    Args:
        device_cfg: Device specification ('cuda', 'cpu', 'auto')

    Returns:
        torch.device
    """
    if device_cfg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_cfg)

    print(f"Using device: {device}")
    if device.type == 'cuda' and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif device.type == 'cuda':
        print("  CUDA specified but not available. Falling back to CPU.")
        device = torch.device('cpu')

    return device


def create_output_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create output directory for experiment.

    Args:
        base_dir: Base output directory
        experiment_name: Name of experiment

    Returns:
        Path to output directory
    """
    output_dir = Path(base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)

    return output_dir


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking training metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")

    # Test config loading
    cfg = load_config("config/default.yaml")
    print(f"✓ Config loaded: {cfg['task']['name']}")

    # Test seed setting
    set_seed(42)
    print(f"✓ Random seed set: {torch.rand(1).item():.4f}")

    # Test device
    device = get_device("auto")
    print(f"✓ Device: {device}")

    # Test parameter counting
    model = torch.nn.Linear(10, 10)
    print(f"✓ Model parameters: {count_parameters(model)}")

    print("\n✓ All utilities tests passed")
