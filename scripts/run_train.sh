#!/bin/bash
# Train all three models on Burgers 1D task

set -e  # Exit on error

echo "================================"
echo "SRΨ-Engine Tiny: Training Script"
echo "================================"

# Common arguments
CONFIG="config/burgers.yaml"
OUTPUT_BASE="outputs/burgers_1d"

# Model 1: Baseline MLP
echo ""
echo "Training Baseline MLP..."
python src/train.py \
    --config $CONFIG \
    --model baseline_mlp \
    --output $OUTPUT_BASE/baseline_mlp

# Model 2: Baseline Transformer
echo ""
echo "Training Baseline Transformer..."
python src/train.py \
    --config $CONFIG \
    --model baseline_transformer \
    --output $OUTPUT_BASE/baseline_transformer

# Model 3: SRΨ-Engine Tiny
echo ""
echo "Training SRΨ-Engine Tiny..."
python src/train.py \
    --config $CONFIG \
    --model srpsi_engine \
    --output $OUTPUT_BASE/srpsi_engine

echo ""
echo "================================"
echo "✓ All models trained successfully!"
echo "================================"
echo ""
echo "Results saved to: $OUTPUT_BASE"
