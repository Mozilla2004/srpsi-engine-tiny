"""
SRΨ-v2.0 Training Script
=========================

Training script for the hybrid architecture with physical loss.

Usage:
    python train_v2_hybrid.py --config config/burgers_v2.yaml

Author: TRAE + Claude Code
Version: 2.0-Hybrid
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Import SRΨ-v2.0 components
import sys
sys.path.append(str(Path(__file__).parent))

from src.models.srpsi_v2_hybrid import create_srpsi_v2_model
from src.training.physical_loss import create_physical_loss
from src.data.data_loader import load_burgers_data


class TrainerV2:
    """
    Trainer for SRΨ-v2.0 Hybrid Model

    Features:
    - Physical loss with adaptive weighting
    - Comprehensive logging
    - Checkpoint management
    - Physical metrics tracking
    """

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # Create model
        print("🏗️  Creating SRΨ-v2.0 Hybrid Model...")
        self.model = create_srpsi_v2_model(cfg, device)
        print(f"✅ Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # Create loss function
        print("🔧 Creating Physical Loss...")
        self.loss_fn = create_physical_loss(cfg)
        print("✅ Physical Loss created")

        # Create optimizer
        print("⚙️  Creating Optimizer...")
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg['training']['learning_rate'],
            weight_decay=cfg['training']['weight_decay']
        )
        print(f"✅ Optimizer created (lr={cfg['training']['learning_rate']})")

        # Create scheduler
        print("📅 Creating Learning Rate Scheduler...")
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg['training']['num_epochs'],
            eta_min=cfg['training']['scheduler']['min_lr']
        )
        print("✅ Scheduler created (Cosine Annealing)")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.loss_history = {
            'train': [],
            'val': [],
            'energy_drift': [],
            'momentum_drift': []
        }

        # Create checkpoint directory
        self.checkpoint_dir = Path(cfg['checkpoint']['dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_energy = 0.0
        total_momentum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (batch_x, batch_y) in enumerate(pbar):
            # Move to device
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(batch_x)

            # Compute loss with physical constraints
            loss, loss_dict = self.loss_fn(pred, batch_y, epoch=epoch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_mse += loss_dict['mse']
            total_energy += loss_dict['energy']
            total_momentum += loss_dict['momentum']

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'E': f"{loss_dict['energy']:.4f}",
                'M': f"{loss_dict['momentum']:.4f}"
            })

        # Compute averages
        num_batches = len(train_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'energy_drift': total_energy / num_batches,
            'momentum_drift': total_momentum / num_batches
        }

        return metrics

    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_energy = 0.0
        total_momentum = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

            for batch_x, batch_y in pbar:
                # Move to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                pred = self.model(batch_x)

                # Compute loss
                loss, loss_dict = self.loss_fn(pred, batch_y, epoch=epoch)

                # Track metrics
                total_loss += loss.item()
                total_mse += loss_dict['mse']
                total_energy += loss_dict['energy']
                total_momentum += loss_dict['momentum']

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'E': f"{loss_dict['energy']:.4f}",
                    'M': f"{loss_dict['momentum']:.4f}"
                })

        # Compute averages
        num_batches = len(val_loader)
        metrics = {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'energy_drift': total_energy / num_batches,
            'momentum_drift': total_momentum / num_batches
        }

        return metrics

    def save_checkpoint(self, epoch, val_loss, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': self.cfg
        }

        # Save latest
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved (val_loss={val_loss:.4f})")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keep only last N"""
        keep_n = self.cfg['checkpoint'].get('keep_last_n', 3)

        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > keep_n:
            for ckpt in checkpoints[:-keep_n]:
                ckpt.unlink()

    def train(self, train_loader, val_loader):
        """Full training loop"""
        print("\n" + "="*70)
        print(" " * 20 + "STARTING TRAINING")
        print("="*70 + "\n")

        num_epochs = self.cfg['training']['num_epochs']
        val_interval = self.cfg['validation']['val_interval']

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            print(f"\n📅 Epoch {epoch}/{num_epochs}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Energy Drift: {train_metrics['energy_drift']:.4f}")
            print(f"   Momentum Drift: {train_metrics['momentum_drift']:.4f}")

            # Validate
            if epoch % val_interval == 0:
                val_metrics = self.validate(val_loader, epoch)
                print(f"   Val Loss: {val_metrics['loss']:.4f}")

                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics['loss'], val_metrics)

            # Update learning rate
            self.scheduler.step()

            # Log history
            self.loss_history['train'].append(train_metrics['loss'])
            self.loss_history['energy_drift'].append(train_metrics['energy_drift'])
            self.loss_history['momentum_drift'].append(train_metrics['momentum_drift'])

        print("\n" + "="*70)
        print(" " * 20 + "TRAINING COMPLETED")
        print("="*70)
        print(f"\n🏆 Best Val Loss: {self.best_val_loss:.4f}")


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def prepare_data(cfg, device):
    """Prepare data loaders"""
    print("📊 Loading Burgers data...")

    # Load data
    data = load_burgers_data()

    # Split
    num_train = int(cfg['data']['num_samples'] * cfg['data']['train_split'])
    num_val = int(cfg['data']['num_samples'] * cfg['data']['val_split'])

    # Prepare tensors
    # Input: first tin steps
    # Output: next tout steps (from step tin to tin+tout)
    train_x = torch.tensor(data['u_train'][:num_train, :cfg['task']['tin'], :], dtype=torch.float32)
    train_y = torch.tensor(data['u_train'][:num_train, cfg['task']['tin']:cfg['task']['tin']+cfg['task']['tout'], :], dtype=torch.float32)

    val_x = torch.tensor(data['u_val'][:num_val, :cfg['task']['tin'], :], dtype=torch.float32)
    val_y = torch.tensor(data['u_val'][:num_val, cfg['task']['tin']:cfg['task']['tin']+cfg['task']['tout'], :], dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['hardware']['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['hardware']['num_workers']
    )

    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Val samples: {len(val_dataset)}")

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train SRΨ-v2.0 Hybrid Model")
    parser.add_argument("--config", type=str, default="config/burgers_v2.yaml",
                        help="Path to configuration file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Load config
    print("📋 Loading configuration...")
    cfg = load_config(args.config)
    print(f"✅ Configuration loaded from {args.config}")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(cfg['hardware']['device'] if torch.cuda.is_available() else "cpu")

    print(f"🔧 Device: {device}")

    # Prepare data
    train_loader, val_loader = prepare_data(cfg, device)

    # Create trainer
    trainer = TrainerV2(cfg, device)

    # Train
    trainer.train(train_loader, val_loader)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
