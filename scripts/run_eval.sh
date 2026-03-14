#!/bin/bash
# Evaluate all three models and generate comparison plots

set -e

echo "================================"
echo "SRΨ-Engine Tiny: Evaluation Script"
echo "================================"

CONFIG="config/burgers.yaml"
OUTPUT_BASE="outputs/burgers_1d"

# Run evaluation script (to be implemented)
echo ""
echo "Running evaluation..."

python src/eval.py \
    --config $CONFIG \
    --output_dir $OUTPUT_BASE

echo ""
echo "================================"
echo "✓ Evaluation complete!"
echo "================================"
echo ""
echo "Comparison plots saved to: $OUTPUT_BASE/comparison"
