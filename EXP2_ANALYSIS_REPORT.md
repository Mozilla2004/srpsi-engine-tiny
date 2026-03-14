#!/usr/bin/env python3
"""
Extract training metrics from checkpoint and generate report
"""

import torch
import json
from pathlib import Path
from datetime import datetime


def analyze_exp2_checkpoint():
    """Analyze Exp2 (SRΨ Real-only) final checkpoint"""

    checkpoint_path = "outputs/ablation_study/srpsi_real/srpsi_real/checkpoints/final.pt"

    print("=" * 80)
    print(" " * 20 + "EXP2: SRΨ REAL-ONLY - ANALYSIS REPORT")
    print("=" * 80)

    if not Path(checkpoint_path).exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return

    # Load checkpoint
    print("\n📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract information
    epoch = checkpoint.get('epoch', 'N/A')
    loss = checkpoint.get('loss', 'N/A')

    # Model parameters
    state_dict = checkpoint.get('model_state_dict', {})
    total_params = sum(p.numel() for p in state_dict.values())

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    print(f"\n📊 Training Progress:")
    print(f"  Final Epoch: {epoch}/80")
    print(f"  Completion: {epoch/80*100:.1f}%")

    if isinstance(loss, (int, float)):
        print(f"  Final Loss: {loss:.4f}")

    print(f"\n🏗️  Model Architecture:")
    print(f"  Model: SRΨ Real-only (Exp2)")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: ~{Path(checkpoint_path).stat().st_size / (1024*1024):.1f} MB")

    print(f"\n🔑 Key Features:")
    print(f"  ✓ Real-valued state (no complex numbers)")
    print(f"  ✓ Spatial coupling operator (S)")
    print(f"  ✓ Rhythm operator (R)")
    print(f"  ✗ Removed: Imaginary component from Ψ")

    print("\n" + "=" * 80)
    print("CHECKPOINT CONTENT")
    print("=" * 80)

    print(f"\nKeys in checkpoint:")
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"  • {key}: dict with {len(value)} items")
        elif isinstance(value, torch.Tensor):
            print(f"  • {key}: tensor {tuple(value.shape)}")
        elif isinstance(value, int):
            print(f"  • {key}: {value}")
        else:
            print(f"  • {key}: {type(value).__name__}")

    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE DETAILS")
    print("=" * 80)

    # Analyze model structure
    print(f"\nTotal layers: {len(state_dict)}")

    # Count parameters by component
    encoder_params = sum(p.numel() for name, p in state_dict.items() if 'encoder' in name)
    srpsi_params = sum(p.numel() for name, p in state_dict.items() if 'blocks' in name)
    output_params = sum(p.numel() for name, p in state_dict.items() if 'output' in name)

    print(f"\nParameter distribution:")
    print(f"  Encoder:   {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  SRΨ Blocks: {srpsi_params:,} ({srpsi_params/total_params*100:.1f}%)")
    print(f"  Output:     {output_params:,} ({output_params/total_params*100:.1f}%)")

    # SRΨ block details
    srpsi_layers = {name: shape for name, p in state_dict.items() if 'blocks.0' in name}
    if srpsi_layers:
        print(f"\nSRΨ Block 0 structure ({len(srpsi_layers)} layers):")
        for name, shape in list(srpsi_layers.items())[:8]:
            print(f"  • {name.split('.')[-1]}: {shape}")
        if len(srpsi_layers) > 8:
            print(f"  • ... and {len(srpsi_layers) - 8} more")

    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)

    if 'config' in checkpoint:
        config = checkpoint['config']
        print("\nHyperparameters:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print("\n⚠️  Configuration not saved in checkpoint")
        print("   Refer to config/burgers.yaml for training settings")

    print("\n" + "=" * 80)
    print("COMPARISON NOTES")
    print("=" * 80)

    print("\n📍 Exp2 vs Exp1 (SRΨ Full):")
    print("  • Removed: Complex-valued state representation")
    print("  • Expected: Reduced shift robustness (no phase)")
    print("  • Expected: Similar or slightly worse MSE")

    print("\n📍 Exp2 vs Exp3 (SRΨ w/o R):")
    print("  • Kept: Rhythm operator (R)")
    print("  • Expected: Better stability and lower energy drift")
    print("  • Expected: Faster convergence")

    print("\n" + "=" * 80)
    print("FILE INFORMATION")
    print("=" * 80)

    file_path = Path(checkpoint_path)
    print(f"\nPath: {file_path}")
    print(f"Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Modified: {datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

    # List all checkpoints
    checkpoint_dir = file_path.parent
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    print(f"\nTotal checkpoints: {len(checkpoints)}")
    print(f"First: {checkpoints[0].name}")
    print(f"Last:  {checkpoints[-1].name}")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)

    print("\n💡 Next Steps:")
    print("  1. Run evaluation script to get metrics on test set")
    print("  2. Compare with Exp1, Exp3, Exp4, Exp5")
    print("  3. Generate ablation study analysis report")
    print("  4. Visualize training curves")

    # Save summary to JSON
    summary = {
        "experiment": "Exp2 - SRΨ Real-only",
        "checkpoint_file": str(checkpoint_path),
        "final_epoch": int(epoch) if isinstance(epoch, int) else epoch,
        "final_loss": float(loss) if isinstance(loss, (int, float)) else loss,
        "total_parameters": total_params,
        "checkpoint_size_mb": file_path.stat().st_size / (1024*1024),
        "analysis_time": datetime.now().isoformat()
    }

    summary_file = "outputs/ablation_study/srpsi_real/summary.json"
    Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    analyze_exp2_checkpoint()
