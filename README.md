# SRΨ-Engine Tiny: Minimal Experimental Project

**A minimal experimental project for testing whether a dynamics-oriented block with structure, rhythm, and stable projection inductive biases can improve long-horizon rollout, conservation control, and shift robustness in 1D field evolution tasks.**

---

## Overview

This project implements and compares three model architectures on 1D field evolution prediction tasks:

1. **Baseline A (MLP)**: Simple predictor that flattens spatiotemporal input
2. **Baseline B (Transformer)**: Transformer-style predictor with self-attention
3. **Model C (SRΨ-Engine Tiny)**: Dynamics-oriented model with structure/rhythm/stability operators

### Key Hypotheses

We test whether SRΨ-Engine's inductive biases improve:
- **Long-term rollout stability**: Error accumulation over multi-step prediction
- **Conservation control**: Energy drift management
- **Shift/phase robustness**: Translation equivariance
- **Recovery capability**: Stability under small perturbations

---

## Installation

```bash
# Clone repository
cd /path/to/srpsi-engine-tiny

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA (optional, but recommended)

---

## Project Structure

```
srpsi-engine-tiny/
├── config/              # Configuration files
│   ├── default.yaml
│   └── burgers.yaml
├── data/                # Generated datasets
├── outputs/             # Training results, logs, plots
├── src/
│   ├── data_gen.py      # Synthetic data generation
│   ├── datasets.py      # PyTorch Dataset wrapper
│   ├── utils.py         # Utilities (checkpoint, logging, etc.)
│   ├── losses.py        # Multi-component loss functions
│   ├── metrics.py       # Evaluation metrics
│   ├── train.py         # Training script
│   ├── eval.py          # Evaluation script
│   ├── plot.py          # Visualization utilities
│   └── models/
│       ├── baseline_mlp.py
│       ├── baseline_transformer.py
│       └── srpsi_engine_tiny.py
├── scripts/
│   ├── run_train.sh     # Train all three models
│   ├── run_eval.sh      # Evaluate and compare
│   └── run_ablation.sh  # Ablation studies
└── README.md
```

---

## Quick Start

### 1. Generate Data

```bash
python src/data_gen.py \
    --task burgers_1d \
    --num_samples 4800 \
    --total_steps 48 \
    --nx 128 \
    --output data/burgers_1d.npy
```

### 2. Train Models

**Option A: Train all models (recommended)**
```bash
bash scripts/run_train.sh
```

**Option B: Train individual models**
```bash
# Baseline MLP
python src/train.py \
    --config config/burgers.yaml \
    --model baseline_mlp \
    --output outputs/burgers_1d/baseline_mlp

# Baseline Transformer
python src/train.py \
    --config config/burgers.yaml \
    --model baseline_transformer \
    --output outputs/burgers_1d/baseline_transformer

# SRΨ-Engine Tiny
python src/train.py \
    --config config/burgers.yaml \
    --model srpsi_engine \
    --output outputs/burgers_1d/srpsi_engine
```

### 3. Evaluate and Compare

```bash
bash scripts/run_eval.sh
```

---

## Configuration

Edit `config/burgers.yaml` to adjust:

```yaml
task:
  nx: 128              # Spatial resolution
  tin: 16              # Input time steps
  tout: 32             # Prediction time steps
  samples_train: 4000

training:
  batch_size: 32
  epochs: 80
  lr: 0.0005

loss:
  lambda_cons: 0.2     # Conservation weight
  lambda_phase: 0.1    # Shift consistency weight
  lambda_smooth: 0.05  # Smoothness weight

model:
  hidden_dim: 64
  depth: 3             # Number of SRΨ blocks
  kernel_size: 5
```

---

## Model Architectures

### Baseline MLP
- **Idea**: Flatten input → FC layers → reshape output
- **Parameters**: ~200K (depends on hidden_dim)
- **Expected behavior**: Good short-term fit, poor long-term stability

### Baseline Transformer
- **Idea**: Time frames as tokens → self-attention → decode future
- **Parameters**: ~300K (depends on d_model, num_layers)
- **Expected behavior**: Strong short-term, may drift in long rollouts

### SRΨ-Engine Tiny
- **Idea**: Encode as complex field ψ → apply S/R/N/Φ operators → decode
- **Components**:
  - **S (Structure)**: Local spatial coupling
  - **R (Rhythm)**: Phase rotation dynamics
  - **N (Nonlinear)**: Nonlinear modulation
  - **Φ (Stable Projection)**: Energy control
- **Parameters**: ~150K (depends on hidden_dim, depth)
- **Expected behavior**: Competitive short-term, better long-term stability

---

## Loss Function

Total loss = weighted sum of four components:

```python
L = L_pred
    + λ_cons · L_conservation      # Energy drift penalty
    + λ_phase · L_shift_consistency # Translation equivariance
    + λ_smooth · L_smoothness       # Temporal smoothness
```

---

## Evaluation Metrics

1. **Rollout MSE**: Average prediction error over horizon
2. **Late-Horizon MSE**: Error in later half of predictions
3. **Energy Drift**: |E_pred(t) - E_true(t)|
4. **Shift Robustness**: ||model(shift(x)) - shift(model(x))||

---

## Expected Results

We anticipate the following pattern:

| Model | Short-Term | Long-Term | Energy Drift | Shift Robustness |
|-------|-----------|-----------|--------------|------------------|
| MLP   | Good      | Poor      | High         | Poor             |
| Transformer | Excellent | Moderate  | Moderate     | Moderate         |
| **SRΨ** | Good      | **Best**   | **Low**      | **Best**         |

The key question is whether SRΨ-Engine achieves **better stability without sacrificing accuracy**.

---

## Output Files

After training, each model directory contains:

```
outputs/burgers_1d/<model_name>/
├── checkpoints/
│   ├── epoch_20.pt
│   ├── epoch_40.pt
│   └── final.pt
├── logs/              # TensorBoard logs
├── plots/             # Training curves
├── config.yaml        # Used configuration
└── data.npy           # Generated dataset (if not provided)
```

View TensorBoard logs:
```bash
tensorboard --logdir outputs/burgers_1d/<model_name>/logs
```

---

## Milestones

### Milestone 1: ✅ Can Run
- [x] Data generation works
- [x] All three models train
- [x] Logging and checkpoints work
- [x] Basic evaluation produces outputs

### Milestone 2: Can Compare
- [ ] Unified evaluation script
- [ ] Four core comparison plots
- [ ] Clear model comparison metrics

### Milestone 3: Can Explain
- [ ] Internal state inspection (ψ norms, phase gates)
- [ ] Ablation studies (S/R/N/Φ contributions)
- [ ] Architectural analysis

---

## Troubleshooting

**Out of memory**: Reduce `batch_size` or `nx` in config

**Training unstable**: Reduce `lr` or increase `grad_clip`

**Loss NaN**: Check data generation, reduce `dt` for Burgers integration

**Slow training**: Reduce `eval.perturb_shift` frequency (computed every 5 batches by default)

---

## Citation

```bibtex
@misc{srpsi_engine_tiny_2026,
  title={SRΨ-Engine Tiny: Minimal Experimental Project},
  author={Genesis-OS Research Team},
  year={2026},
  note={Internal working draft}
}
```

---

## License

Internal research project.

---

**Contact**: Genesis-OS Research Team
**Status**: Prototype Alpha - Not Production-Ready
