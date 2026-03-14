#!/usr/bin/env python3
"""
Analyze trained checkpoint
==========================

Loads and displays information from a trained model checkpoint.

Usage:
    python analyze_checkpoint.py --checkpoint outputs/ablation_study/srpsi_real/srpsi_real/checkpoints/final.pt
"""

import torch
import argparse
from pathlib import Path


def analyze_checkpoint(checkpoint_path: str):
    """Analyze a trained checkpoint"""

    print("=" * 70)
    print(f"Analyzing Checkpoint: {checkpoint_path}")
    print("=" * 70)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Display checkpoint info
    print("\n📦 Checkpoint Information:")
    print(f"  File: {checkpoint_path}")
    size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")

    # Display keys
    print("\n🔑 Checkpoint Keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: (dict with {len(checkpoint[key])} sub-keys)")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: Tensor {tuple(checkpoint[key].shape)}")
        else:
            print(f"  {key}: {type(checkpoint[key]).__name__}")

    # Display training metrics if available
    if 'epoch' in checkpoint:
        print(f"\n📊 Training Info:")
        print(f"  Final Epoch: {checkpoint['epoch']}")

    if 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        print(f"  Train Losses ({len(train_losses)} epochs):")
        print(f"    First:  {train_losses[0]:.4f}")
        print(f"    Last:   {train_losses[-1]:.4f}")
        print(f"    Min:    {min(train_losses):.4f}")
        print(f"    Max:    {max(train_losses):.4f}")

    if 'val_losses' in checkpoint:
        val_losses = checkpoint['val_losses']
        print(f"  Val Losses ({len(val_losses)} epochs):")
        print(f"    First:  {val_losses[0]:.4f}")
        print(f"    Last:   {val_losses[-1]:.4f}")
        print(f"    Min:    {min(val_losses):.4f}")
        print(f"    Max:    {max(val_losses):.4f}")

    if 'best_val_loss' in checkpoint:
        print(f"\n🏆 Best Validation Loss: {checkpoint['best_val_loss']:.4f}")

    if 'best_val_mse' in checkpoint:
        print(f"  Best Val MSE: {checkpoint['best_val_mse']:.6f}")

    if 'best_val_drift' in checkpoint:
        print(f"  Best Val Drift: {checkpoint['best_val_drift']:.6f}")

    # Model architecture
    if 'model_state_dict' in checkpoint:
        print(f"\n🏗️  Model Architecture:")
        state_dict = checkpoint['model_state_dict']

        # Count parameters
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  Total Parameters: {total_params:,}")

        # Layer information
        print(f"\n  Layers ({len(state_dict)}):")
        for name, param in list(state_dict.items())[:10]:  # Show first 10
            print(f"    {name}: {tuple(param.shape)}")
        if len(state_dict) > 10:
            print(f"    ... and {len(state_dict) - 10} more layers")

    # Config
    if 'config' in checkpoint:
        print(f"\n⚙️  Configuration:")
        for key, value in checkpoint['config'].items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trained checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/ablation_study/srpsi_real/srpsi_real/checkpoints/final.pt",
        help="Path to checkpoint file"
    )

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        print("\nLooking for checkpoints...")
        checkpoint_dir = Path("outputs/ablation_study/srpsi_real/srpsi_real/checkpoints")
        if checkpoint_dir.exists():
            print(f"\nAvailable checkpoints in {checkpoint_dir}:")
            for ckpt in sorted(checkpoint_dir.glob("*.pt")):
                print(f"  - {ckpt.name}")
        else:
            print("\n❌ No checkpoint directory found.")
            print("   Please ensure the training has completed and the checkpoint is available.")
        exit(1)

    analyze_checkpoint(args.checkpoint)
