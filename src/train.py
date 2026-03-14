"""
Training Script for SRΨ-Engine Tiny Experiments
=================================================

Supports three model types:
- baseline_mlp: Simple MLP predictor
- baseline_transformer: Transformer-style predictor
- srpsi_engine: SRΨ-Engine Tiny (dynamics-oriented)

Features:
- Multi-component loss (prediction + conservation + shift + smooth)
- Gradient clipping for stability
- Checkpoint saving
- TensorBoard logging
- Validation during training

Usage:
    python src/train.py --config config/burgers.yaml --model srpsi_engine

Author: SRΨ-Engine Tiny Experiment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json

from src.utils import (
    load_config, set_seed, get_device, create_output_dir,
    save_checkpoint, AverageMeter, count_parameters
)
from src.datasets import create_dataloaders
from src.models import (
    BaselineMLP, BaselineTransformer, SRPsiEngineTiny,
    SRPsiEngineReal, SRPsiEngineNoR, ConvBaseline, TransformerRelPE
)
from src.losses import total_loss
from src.metrics import rollout_mse, energy_drift, shift_robustness


def create_model(model_type: str, cfg: dict, device: torch.device) -> nn.Module:
    """
    Create model instance based on type.

    Args:
        model_type: 'baseline_mlp', 'baseline_transformer', or 'srpsi_engine'
        cfg: Configuration dict
        device: Target device

    Returns:
        model: Model instance
    """
    tin = cfg['task']['tin']
    tout = cfg['task']['tout']
    nx = cfg['task']['nx']
    hidden_dim = cfg['model']['hidden_dim']
    depth = cfg['model']['depth']
    kernel_size = cfg['model']['kernel_size']

    if model_type == 'baseline_mlp':
        model = BaselineMLP(tin, tout, nx, hidden_dim=hidden_dim)

    elif model_type == 'baseline_transformer':
        model = BaselineTransformer(
            tin, tout, nx,
            d_model=hidden_dim,
            nhead=4,
            num_layers=depth,
            dropout=cfg['model']['dropout']
        )

    elif model_type == 'srpsi_engine':
        model = SRPsiEngineTiny(
            tin=tin,
            nx=nx,
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dt=0.01,
            tout=tout
        )

    # Ablation study models
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

    elif model_type == 'conv_baseline':
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

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    cfg: dict,
    device: torch.device,
    epoch: int
) -> dict:
    """
    Train for one epoch.

    Args:
        model: Model instance
        dataloader: Training dataloader
        optimizer: Optimizer
        cfg: Configuration dict
        device: Target device
        epoch: Current epoch number

    Returns:
        metrics: Dict of training metrics
    """
    model.train()

    loss_meter = AverageMeter()
    pred_meter = AverageMeter()
    cons_meter = AverageMeter()
    phase_meter = AverageMeter()
    smooth_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        x = batch["x"].to(device)   # [B, Tin, X]
        y = batch["y"].to(device)   # [B, Tout, X]

        # Forward pass
        pred = model(x)

        # Compute loss
        loss, logs = total_loss(
            model, x, pred, y, cfg,
            epoch=epoch,
            compute_shift=(batch_idx % 5 == 0)  # Compute shift loss less frequently
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg['training']['grad_clip']
        )

        optimizer.step()

        # Update meters
        batch_size = x.shape[0]
        loss_meter.update(logs["loss_total"], batch_size)
        pred_meter.update(logs["loss_pred"], batch_size)
        cons_meter.update(logs["loss_cons"], batch_size)
        phase_meter.update(logs["loss_phase"], batch_size)
        smooth_meter.update(logs["loss_smooth"], batch_size)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.6f}",
            "pred": f"{pred_meter.avg:.6f}"
        })

    return {
        "train_loss": loss_meter.avg,
        "train_loss_pred": pred_meter.avg,
        "train_loss_cons": cons_meter.avg,
        "train_loss_phase": phase_meter.avg,
        "train_loss_smooth": smooth_meter.avg,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    cfg: dict,
    device: torch.device
) -> dict:
    """
    Validate model.

    Args:
        model: Model instance
        dataloader: Validation dataloader
        cfg: Configuration dict
        device: Target device

    Returns:
        metrics: Dict of validation metrics
    """
    model.eval()

    loss_meter = AverageMeter()
    rollout_meter = AverageMeter()
    drift_meter = AverageMeter()

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        # Forward pass
        pred = model(x)

        # Prediction loss
        loss, _ = total_loss(model, x, pred, y, cfg, compute_shift=False)
        loss_meter.update(loss.item(), x.shape[0])

        # Rollout MSE
        mse = rollout_mse(pred, y)
        rollout_meter.update(mse, x.shape[0])

        # Energy drift
        drift = energy_drift(pred, y)
        drift_meter.update(drift, x.shape[0])

    return {
        "val_loss": loss_meter.avg,
        "val_rollout_mse": rollout_meter.avg,
        "val_energy_drift": drift_meter.avg,
    }


def main():
    parser = argparse.ArgumentParser(description="Train SRΨ-Engine Tiny")
    parser.add_argument("--config", type=str, default="config/burgers.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default="srpsi_engine",
                        choices=["baseline_mlp", "baseline_transformer", "srpsi_engine", 
                                 "srpsi_real", "srpsi_no_r", "conv_baseline", "transformer_rel_pe"],
                        help="Model type")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to pre-generated data (will generate if not provided)")
    parser.add_argument("--output", type=str, default="outputs/experiment",
                        help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Override config with command line arguments
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    # Set random seed
    set_seed(cfg["seed"])

    # Setup device
    device = get_device(cfg.get("device", "cuda"))

    # Create output directory
    output_dir = create_output_dir(args.output, args.model)
    print(f"Output directory: {output_dir}")

    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)

    # Generate or load data
    if args.data is None:
        print("Generating data...")
        from src.data_gen import generate_burgers_1d

        num_train = cfg['task']['samples_train']
        num_val = cfg['task']['samples_val']
        num_test = cfg['task']['samples_test']
        total_steps = cfg['task']['tin'] + cfg['task']['tout']
        nx = cfg['task']['nx']
        dt = cfg['task']['dt']
        dx = cfg['task']['dx']

        # Generate combined dataset
        all_data = generate_burgers_1d(
            num_samples=num_train + num_val + num_test,
            total_steps=total_steps,
            nx=nx,
            dt=dt,
            dx=dx,
            nu=cfg['task'].get('nu', 0.01),
            seed=cfg['seed']
        )

        data_path = output_dir / "data.npy"
        import numpy as np
        np.save(data_path, all_data)
        print(f"Data saved to: {data_path}")

        args.data = str(data_path)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data,
        tin=cfg['task']['tin'],
        tout=cfg['task']['tout'],
        batch_size=cfg['training']['batch_size'],
        num_train=cfg['task']['samples_train'],
        num_val=cfg['task']['samples_val'],
        num_test=cfg['task']['samples_test'],
        num_workers=0
    )

    # Create model
    print(f"Creating model: {args.model}")
    model = create_model(args.model, cfg, device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    # Training loop
    print("\nStarting training...")

    for epoch in range(start_epoch, cfg['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, cfg, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, cfg, device)

        # Log to TensorBoard
        for key, value in {**train_metrics, **val_metrics}.items():
            writer.add_scalar(key, value, epoch)

        # Print metrics
        print(f"Train Loss: {train_metrics['train_loss']:.6f}")
        print(f"Val Loss:   {val_metrics['val_loss']:.6f}")
        print(f"Val MSE:    {val_metrics['val_rollout_mse']:.6f}")
        print(f"Val Drift:  {val_metrics['val_energy_drift']:.6f}")

        # Save checkpoint
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            print(f"✓ New best model: {best_val_loss:.6f}")

        if (epoch + 1) % cfg['output']['save_interval'] == 0 or is_best:
            ckpt_path = output_dir / "checkpoints" / f"epoch_{epoch + 1}.pt"
            save_checkpoint(
                model, optimizer, epoch, val_metrics['val_loss'],
                str(ckpt_path),
                best_val_loss=best_val_loss
            )

    # Final test evaluation
    print("\nEvaluating on test set...")
    test_metrics = validate(model, test_loader, cfg, device)

    for key, value in test_metrics.items():
        writer.add_scalar(f"test_{key}", value, 0)

    print(f"Test Loss:   {test_metrics['val_loss']:.6f}")
    print(f"Test MSE:    {test_metrics['val_rollout_mse']:.6f}")
    print(f"Test Drift:  {test_metrics['val_energy_drift']:.6f}")

    # Save final model
    final_path = output_dir / "checkpoints" / "final.pt"
    save_checkpoint(
        model, optimizer, cfg['training']['epochs'] - 1, test_metrics['val_loss'],
        str(final_path)
    )

    writer.close()

    print("\n✓ Training complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
