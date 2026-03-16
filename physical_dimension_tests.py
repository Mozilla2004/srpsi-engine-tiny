#!/usr/bin/env python3
"""
Physical Dimension Tests for Ablation Study
============================================

Validate TRAE's insights:
- "Conv's low Loss is 'false victory'"
- "SRΨ's higher Loss is 'physical cost' for maintaining invariants"

Tests:
1. Shift Robustness (平移鲁棒性)
2. Energy Drift (能量守恒)
3. Noise Robustness (抗扰动)
4. Field-State Coherence (场状态一致性)

Author: SRΨ-Engine Tiny Experiment
Date: 2026-03-15
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import get_model
from utils import load_config


class PhysicalTester:
    """Test models on physical dimensions beyond training loss"""

    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self.test_data = None
        self.results = {}

    def load_all_models(self):
        """Load all 4 trained models"""
        print("\n" + "="*70)
        print(" " * 20 + "LOADING TRAINED MODELS")
        print("="*70)

        model_configs = {
            "Exp2_SRΨ_Real": {
                "ckpt": "outputs/ablation_study/srpsi_real/srpsi_real/checkpoints/final.pt",
                "model_type": "srpsi_real",
                "config": "config/burgers.yaml"
            },
            "Exp3_SRΨ_w/o_R": {
                "ckpt": "outputs/ablation_study/srpsi_no_r/srpsi_no_r/checkpoints/final.pt",
                "model_type": "srpsi_no_r",
                "config": "config/burgers.yaml"
            },
            "Exp4_Conv": {
                "ckpt": "checkpoints_colab/exp4_conv_final.pt",
                "model_type": "conv_baseline",
                "config": "config/burgers.yaml"
            },
            "Exp5_Transformer": {
                "ckpt": "checkpoints_colab/exp5_transformer_final.pt",
                "model_type": "transformer_rel_pe",
                "config": "config/burgers.yaml"
            }
        }

        for name, config in model_configs.items():
            print(f"\n📦 Loading {name}...")

            ckpt_path = Path(config["ckpt"])
            if not ckpt_path.exists():
                print(f"  ❌ Checkpoint not found: {ckpt_path}")
                continue

            try:
                # Load checkpoint
                ckpt = torch.load(config["ckpt"], map_location=self.device)

                # Load model config
                cfg = load_config(config["config"])

                # Create model
                model = get_model(config["model_type"])
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(self.device)
                model.eval()

                # Store model
                self.models[name] = {
                    "model": model,
                    "model_type": config["model_type"],
                    "train_loss": float(ckpt.get("loss", 0)),
                    "epoch": int(ckpt.get("epoch", 0))
                }

                print(f"  ✅ Loaded successfully")
                print(f"     - Epoch: {self.models[name]['epoch']}/80")
                print(f"     - Train Loss: {self.models[name]['train_loss']:.2f}")

            except Exception as e:
                print(f"  ❌ Failed to load: {e}")
                continue

        print(f"\n✅ Successfully loaded {len(self.models)}/4 models")

    def load_test_data(self):
        """Load test data"""
        print("\n" + "="*70)
        print(" " * 25 + "LOADING TEST DATA")
        print("="*70)

        data_path = Path("data/burgers_1d.npy")
        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            print("Please run: python src/data_gen.py")
            return False

        data = np.load(data_path, allow_pickle=True).item()

        # Extract test data
        self.test_data = {
            "u_init": data["u_test"][:, :10],  # Initial 10 steps
            "u_true": data["u_test"][:, 10:],   # Next 38 steps (ground truth)
            "viscosity": data["nu_test"],
            "x_grid": data["x"]
        }

        print(f"✅ Test data loaded")
        print(f"   - Samples: {self.test_data['u_init'].shape[0]}")
        print(f"   - Input window: 10 steps")
        print(f"   - Output window: 38 steps")
        print(f"   - Grid points: {self.test_data['x_grid'].shape[0]}")

        return True

    # =========================================================================
    # TEST 1: Shift Robustness (平移鲁棒性)
    # =========================================================================

    def test_shift_robustness(self):
        """
        Test: How does model perform when input is spatially shifted?

        Hypothesis (TRAE):
        - Conv: Error grows rapidly (local convolution breaks)
        - SRΨ: Error grows slowly (S operator maintains coherence)

        Metric: MSE vs shift amount
        """
        print("\n" + "="*70)
        print(" " * 15 + "TEST 1: SHIFT ROBUSTNESS (平移鲁棒性)")
        print("="*70)
        print("\n🎯 Testing: Spatial shift invariance")
        print("   Shift amounts: 0, 4, 8, 16, 32 grid points")

        if len(self.models) == 0:
            print("❌ No models loaded")
            return

        if self.test_data is None:
            print("❌ No test data loaded")
            return

        u_init = torch.tensor(self.test_data["u_init"], dtype=torch.float32)
        u_true = torch.tensor(self.test_data["u_true"], dtype=torch.float32)

        shifts = [0, 4, 8, 16, 32]
        results = {name: [] for name in self.models.keys()}

        for shift in shifts:
            print(f"\n📍 Shift = {shift} grid points")

            # Shift input
            u_init_shifted = torch.roll(u_init, shift, dims=-1)

            for name, model_info in self.models.items():
                model = model_info["model"]

                with torch.no_grad():
                    u_pred = model(u_init_shifted.to(self.device))
                    u_pred = u_pred.cpu().numpy()

                # Compute MSE on unshifted output (model should predict shifted)
                mse = np.mean((u_pred - u_true.numpy()) ** 2)
                results[name].append(mse)

                print(f"  {name:<20} MSE: {mse:.4f}")

        # Store results
        self.results["shift_robustness"] = {
            "shifts": shifts,
            "results": results
        }

        # Analysis
        print("\n" + "-"*70)
        print(" " * 25 + "ANALYSIS")
        print("-"*70)

        # Compute error growth rate
        for name, errors in results.items():
            if len(errors) >= 2:
                # Growth rate: (error at shift 32) / (error at shift 0)
                growth_rate = errors[-1] / (errors[0] + 1e-10)
                print(f"\n{name}:")
                print(f"  - Error at shift 0:   {errors[0]:.4f}")
                print(f"  - Error at shift 32:  {errors[-1]:.4f}")
                print(f"  - Growth rate:        {growth_rate:.2f}x")

        # Rank by robustness (lower growth rate = better)
        growth_rates = {
            name: errors[-1] / (errors[0] + 1e-10)
            for name, errors in results.items()
        }
        ranked = sorted(growth_rates.items(), key=lambda x: x[1])

        print("\n🏆 Ranking by Shift Robustness (lower growth rate = better):")
        for rank, (name, rate) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank}. {name:<20} Growth rate: {rate:.2f}x")

    # =========================================================================
    # TEST 2: Energy Drift (能量守恒)
    # =========================================================================

    def test_energy_drift(self):
        """
        Test: How well does model conserve energy over long rollouts?

        Hypothesis (TRAE):
        - SRΨ: Best energy conservation (R operator models time evolution)
        - Conv: May drift (no explicit time mechanism)

        Metric: Std dev of energy over 100-step rollout
        """
        print("\n" + "="*70)
        print(" " * 20 + "TEST 2: ENERGY DRIFT (能量守恒)")
        print("="*70)
        print("\n🎯 Testing: Long-term energy conservation")
        print("   Rollout: 50 steps (auto-regressive)")

        if len(self.models) == 0:
            print("❌ No models loaded")
            return

        if self.test_data is None:
            print("❌ No test data loaded")
            return

        # Use one sample for rollout
        u_init = torch.tensor(self.test_data["u_init"][:1], dtype=torch.float32)

        rollout_steps = 50
        results = {name: [] for name in self.models.keys()}

        for name, model_info in self.models.items():
            print(f"\n🔄 Rolling out {name}...")

            model = model_info["model"]
            u_current = u_init.to(self.device)
            energy_history = []

            for step in range(rollout_steps):
                with torch.no_grad():
                    u_next = model(u_current)

                    # Compute energy: L2 norm
                    energy = torch.sqrt(torch.sum(u_next ** 2)).item()
                    energy_history.append(energy)

                    # Auto-regressive: use last prediction as next input
                    # For simplicity, just shift window
                    u_current = torch.cat([
                        u_current[:, 1:],
                        u_next[:, -1:]  # Take only last timestep
                    ], dim=1)

            # Compute drift: standard deviation of energy
            energy_mean = np.mean(energy_history)
            energy_std = np.std(energy_history)
            drift_ratio = energy_std / (energy_mean + 1e-10)

            results[name] = {
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "drift_ratio": drift_ratio,
                "history": energy_history
            }

            print(f"  ✅ Mean energy: {energy_mean:.4f}")
            print(f"  ✅ Energy std:  {energy_std:.4f}")
            print(f"  ✅ Drift ratio: {drift_ratio:.4f}")

        # Store results
        self.results["energy_drift"] = results

        # Analysis
        print("\n" + "-"*70)
        print(" " * 25 + "ANALYSIS")
        print("-"*70)

        # Rank by energy conservation (lower drift = better)
        drift_ratios = {
            name: info["drift_ratio"]
            for name, info in results.items()
        }
        ranked = sorted(drift_ratios.items(), key=lambda x: x[1])

        print("\n🏆 Ranking by Energy Conservation (lower drift = better):")
        for rank, (name, ratio) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank}. {name:<20} Drift ratio: {ratio:.4f}")

    # =========================================================================
    # TEST 3: Noise Robustness (抗扰动)
    # =========================================================================

    def test_noise_robustness(self):
        """
        Test: How well does model recover from noisy input?

        Hypothesis (TRAE):
        - SRΨ: Recovers via Ψ mechanism (field-state projection)
        - Conv: Noise propagates, error amplifies

        Metric: MSE vs noise level
        """
        print("\n" + "="*70)
        print(" " * 18 + "TEST 3: NOISE ROBUSTNESS (抗扰动)")
        print("="*70)
        print("\n🎯 Testing: Recovery from Gaussian noise")
        print("   Noise levels: 0.0, 0.05, 0.10, 0.15")

        if len(self.models) == 0:
            print("❌ No models loaded")
            return

        if self.test_data is None:
            print("❌ No test data loaded")
            return

        u_init = torch.tensor(self.test_data["u_init"], dtype=torch.float32)
        u_true = torch.tensor(self.test_data["u_true"], dtype=torch.float32)

        noise_levels = [0.0, 0.05, 0.10, 0.15]
        results = {name: [] for name in self.models.keys()}

        for noise_std in noise_levels:
            print(f"\n📊 Noise std = {noise_std:.2f}")

            # Add noise
            u_noisy = u_init + torch.randn_like(u_init) * noise_std

            for name, model_info in self.models.items():
                model = model_info["model"]

                with torch.no_grad():
                    u_pred = model(u_noisy.to(self.device))
                    u_pred = u_pred.cpu().numpy()

                # Compute MSE
                mse = np.mean((u_pred - u_true.numpy()) ** 2)
                results[name].append(mse)

                print(f"  {name:<20} MSE: {mse:.4f}")

        # Store results
        self.results["noise_robustness"] = {
            "noise_levels": noise_levels,
            "results": results
        }

        # Analysis
        print("\n" + "-"*70)
        print(" " * 25 + "ANALYSIS")
        print("-"*70)

        # Compute error sensitivity
        for name, errors in results.items():
            if len(errors) >= 2:
                # Sensitivity: (error at noise 0.15) / (error at noise 0.0)
                sensitivity = errors[-1] / (errors[0] + 1e-10)
                print(f"\n{name}:")
                print(f"  - Error at noise 0.00: {errors[0]:.4f}")
                print(f"  - Error at noise 0.15: {errors[-1]:.4f}")
                print(f"  - Sensitivity:        {sensitivity:.2f}x")

        # Rank by robustness (lower sensitivity = better)
        sensitivities = {
            name: errors[-1] / (errors[0] + 1e-10)
            for name, errors in results.items()
        }
        ranked = sorted(sensitivities.items(), key=lambda x: x[1])

        print("\n🏆 Ranking by Noise Robustness (lower sensitivity = better):")
        for rank, (name, rate) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank}. {name:<20} Sensitivity: {rate:.2f}x")

    # =========================================================================
    # TEST 4: Field-State Coherence (场状态一致性)
    # =========================================================================

    def test_field_coherence(self):
        """
        Test: How smooth and coherent is the predicted field?

        Hypothesis (TRAE):
        - SRΨ: Higher coherence (S operator maintains structure)
        - Conv: May have artifacts/discontinuities

        Metric: Laplacian (smoothness) + multi-step consistency
        """
        print("\n" + "="*70)
        print(" " * 15 + "TEST 4: FIELD-STATE COHERENCE (场状态一致性)")
        print("="*70)
        print("\n🎯 Testing: Field smoothness and multi-step consistency")

        if len(self.models) == 0:
            print("❌ No models loaded")
            return

        if self.test_data is None:
            print("❌ No test data loaded")
            return

        u_init = torch.tensor(self.test_data["u_init"][:10], dtype=torch.float32)

        results = {}

        for name, model_info in self.models.items():
            print(f"\n🌊 Analyzing {name}...")

            model = model_info["model"]

            with torch.no_grad():
                u_pred = model(u_init.to(self.device))
                u_pred = u_pred.cpu().numpy()

            # Compute smoothness: Laplacian (2nd spatial derivative)
            # Simple finite difference: u[i+1] - 2*u[i] + u[i-1]
            laplacian = np.zeros_like(u_pred)
            laplacian[:, :, 1:-1] = (
                u_pred[:, :, 2:] - 2 * u_pred[:, :, 1:-1] + u_pred[:, :, :-2]
            )

            smoothness = np.mean(np.abs(laplacian))

            # Compute temporal consistency (variance across output timesteps)
            temporal_var = np.var(u_pred, axis=1).mean()

            results[name] = {
                "smoothness": smoothness,
                "temporal_consistency": temporal_var
            }

            print(f"  ✅ Smoothness (Laplacian): {smoothness:.4f}")
            print(f"  ✅ Temporal consistency:   {temporal_consistency:.4f}")

        # Store results
        self.results["field_coherence"] = results

        # Analysis
        print("\n" + "-"*70)
        print(" " * 25 + "ANALYSIS")
        print("-"*70)

        # Rank by smoothness (lower Laplacian = smoother = better)
        smoothness = {
            name: info["smoothness"]
            for name, info in results.items()
        }
        ranked = sorted(smoothness.items(), key=lambda x: x[1])

        print("\n🏆 Ranking by Field Smoothness (lower Laplacian = better):")
        for rank, (name, value) in enumerate(ranked, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank}. {name:<20} Smoothness: {value:.4f}")

    # =========================================================================
    # SUMMARY & SAVE
    # =========================================================================

    def print_summary(self):
        """Print comprehensive summary of all tests"""
        print("\n" + "="*70)
        print(" " * 15 + "PHYSICAL DIMENSION TEST SUMMARY")
        print("="*70)

        print("\n📊 Comparison Table:")
        print("-"*70)

        # Collect all metrics
        summary = []

        for name in self.models.keys():
            row = {"model": name}

            # Training Loss (from checkpoint)
            row["train_loss"] = self.models[name]["train_loss"]

            # Shift Robustness (growth rate)
            if "shift_robustness" in self.results:
                errors = self.results["shift_robustness"]["results"][name]
                row["shift_growth"] = errors[-1] / (errors[0] + 1e-10)

            # Energy Drift
            if "energy_drift" in self.results:
                row["energy_drift"] = self.results["energy_drift"][name]["drift_ratio"]

            # Noise Robustness (sensitivity)
            if "noise_robustness" in self.results:
                errors = self.results["noise_robustness"]["results"][name]
                row["noise_sensitivity"] = errors[-1] / (errors[0] + 1e-10)

            # Field Coherence (smoothness)
            if "field_coherence" in self.results:
                row["smoothness"] = self.results["field_coherence"][name]["smoothness"]

            summary.append(row)

        # Print table
        print(f"\n{'Model':<20} {'TrainLoss':<12} {'Shift':<10} {'Energy':<10} {'Noise':<10} {'Smooth':<10}")
        print("-"*70)

        for row in summary:
            print(f"{row['model']:<20} ", end="")
            print(f"{row['train_loss']:<12.2f} ", end="")

            if "shift_growth" in row:
                print(f"{row['shift_growth']:<10.2f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")

            if "energy_drift" in row:
                print(f"{row['energy_drift']:<10.4f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")

            if "noise_sensitivity" in row:
                print(f"{row['noise_sensitivity']:<10.2f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")

            if "smoothness" in row:
                print(f"{row['smoothness']:<10.4f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")

            print()

        print("\n" + "="*70)
        print(" " * 20 + "KEY INSIGHTS")
        print("="*70)

        # Compare Conv vs SRΨ Real
        if len(summary) >= 2:
            conv_data = next((r for r in summary if "Conv" in r["model"]), None)
            srpsi_data = next((r for r in summary if "SRΨ_Real" in r["model"]), None)

            if conv_data and srpsi_data:
                print("\n🎯 Conv Baseline vs SRΨ Real:")

                print(f"\n  Training Loss:")
                print(f"    Conv:  {conv_data['train_loss']:.2f}")
                print(f"    SRΨ:   {srpsi_data['train_loss']:.2f}")
                print(f"    → Conv wins by {srpsi_data['train_loss']/conv_data['train_loss']:.2f}x")

                if "shift_growth" in conv_data and "shift_growth" in srpsi_data:
                    print(f"\n  Shift Robustness (lower growth = better):")
                    print(f"    Conv:  {conv_data['shift_growth']:.2f}x")
                    print(f"    SRΨ:   {srpsi_data['shift_growth']:.2f}x")
                    if srpsi_data['shift_growth'] < conv_data['shift_growth']:
                        print(f"    → SRΨ wins by {conv_data['shift_growth']/srpsi_data['shift_growth']:.2f}x ✅")
                    else:
                        print(f"    → Conv wins by {srpsi_data['shift_growth']/conv_data['shift_growth']:.2f}x")

    def save_results(self, output_path="results/physical_dimension_tests.json"):
        """Save test results to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare serializable results
        serializable = {}

        for test_name, test_data in self.results.items():
            if test_name == "energy_drift":
                serializable[test_name] = {}
                for model_name, model_data in test_data.items():
                    serializable[test_name][model_name] = {
                        "energy_mean": model_data["energy_mean"],
                        "energy_std": model_data["energy_std"],
                        "drift_ratio": model_data["drift_ratio"]
                        # history is too long, skip
                    }
            else:
                serializable[test_name] = test_data

        # Add model info
        serializable["models"] = {}
        for name, info in self.models.items():
            serializable["models"][name] = {
                "train_loss": float(info["train_loss"]),
                "epoch": int(info["epoch"]),
                "model_type": info["model_type"]
            }

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Run all physical dimension tests"""
    print("\n" + "="*70)
    print(" " * 10 + "SRΨ-ENGINE: PHYSICAL DIMENSION TESTS")
    print("="*70)
    print("\n🎯 Objective: Validate TRAE's insights")
    print("   - 'Conv's low Loss is false victory'")
    print("   - 'SRΨ's higher Loss is physical cost for invariants'")

    # Initialize tester
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tester = PhysicalTester(device=device)

    # Load models
    tester.load_all_models()

    if len(tester.models) == 0:
        print("\n❌ No models loaded. Exiting.")
        return

    # Load test data
    if not tester.load_test_data():
        return

    # Run all tests
    print("\n" + "="*70)
    print(" " * 20 + "RUNNING ALL TESTS")
    print("="*70)

    tester.test_shift_robustness()
    tester.test_energy_drift()
    tester.test_noise_robustness()
    tester.test_field_coherence()

    # Print summary
    tester.print_summary()

    # Save results
    tester.save_results()

    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70)


if __name__ == "__main__":
    main()
