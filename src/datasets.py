"""
PyTorch Dataset Wrapper for Field Evolution Data
==================================================

Provides Dataset and DataLoader for rollout-based training.

Author: SRΨ-Engine Tiny Experiment
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional


class FieldRolloutDataset(Dataset):
    """
    Dataset for autoregressive field prediction.

    Splits each trajectory into input (history) and output (future) windows.
    """

    def __init__(self, array: np.ndarray, tin: int, tout: int):
        """
        Args:
            array: Pre-generated trajectories [num_samples, total_steps, nx]
            tin: Input time steps (history window)
            tout: Output time steps (prediction horizon)
        """
        self.array = array
        self.tin = tin
        self.tout = tout
        self.total_steps = array.shape[1]

        assert tin + tout <= self.total_steps, \
            f"tin ({tin}) + tout ({tout}) exceeds total steps ({self.total_steps})"

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict: {
                "x": input field [Tin, X],
                "y": target future field [Tout, X]
            }
        """
        seq = self.array[idx]  # [T, X]

        x_in = seq[:self.tin]                         # [Tin, X]
        y_out = seq[self.tin:self.tin + self.tout]    # [Tout, X]

        return {
            "x": torch.tensor(x_in, dtype=torch.float32),
            "y": torch.tensor(y_out, dtype=torch.float32)
        }


def create_dataloaders(
    data_path: str,
    tin: int,
    tout: int,
    batch_size: int,
    num_train: Optional[int] = None,
    num_val: Optional[int] = None,
    num_test: Optional[int] = None,
    num_workers: int = 0,
    seed: int = 42
) -> tuple:
    """
    Create train/val/test dataloaders from a single dataset file.

    Args:
        data_path: Path to .npy file containing trajectories
        tin: Input time steps
        tout: Output time steps
        batch_size: Batch size
        num_train: Number of training samples (default: all except val/test)
        num_val: Number of validation samples (default: 400)
        num_test: Number of test samples (default: 400)
        num_workers: Number of worker processes
        seed: Random seed for shuffling

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataset
    data = np.load(data_path)  # [N, T, X]
    num_samples = len(data)

    print(f"Loaded dataset from {data_path}")
    print(f"  Shape: {data.shape}")

    # Default split sizes
    if num_val is None:
        num_val = min(400, num_samples // 10)
    if num_test is None:
        num_test = min(400, num_samples // 10)
    if num_train is None:
        num_train = num_samples - num_val - num_test

    # Split dataset
    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:num_train + num_val + num_test]

    print(f"  Train: {train_data.shape[0]} samples")
    print(f"  Val:   {val_data.shape[0]} samples")
    print(f"  Test:  {test_data.shape[0]} samples")

    # Create datasets
    train_dataset = FieldRolloutDataset(train_data, tin, tout)
    val_dataset = FieldRolloutDataset(val_data, tin, tout)
    test_dataset = FieldRolloutDataset(test_data, tin, tout)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset creation
    import sys

    # Generate dummy data
    num_samples = 100
    total_steps = 48
    nx = 128
    tin, tout = 16, 32

    dummy_data = np.random.randn(num_samples, total_steps, nx).astype(np.float32)

    # Create dataset
    dataset = FieldRolloutDataset(dummy_data, tin, tout)

    # Test single sample
    sample = dataset[0]
    print(f"Input shape:  {sample['x'].shape}")   # [16, 128]
    print(f"Output shape: {sample['y'].shape}")   # [32, 128]

    # Test dataloader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))

    print(f"\nBatch input shape:  {batch['x'].shape}")   # [8, 16, 128]
    print(f"Batch output shape: {batch['y'].shape}")   # [8, 32, 128]

    print("\n✓ Dataset test passed")
