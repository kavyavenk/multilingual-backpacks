#!/bin/bash
# Complete training and evaluation pipeline for tiny Backpack model
# Run this on your GCP GPU instance

set -e  # Exit on error

echo "=================================================="
echo "MULTILINGUAL BACKPACK: TRAIN & EVALUATE PIPELINE"
echo "=================================================="
echo ""

# Configuration
CONFIG="train_europarl_tiny"
OUT_DIR="out/tiny"
DATA_DIR="data/europarl"

# Step 1: Check GPU availability
echo "Step 1: Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo "✓ GPU available"
else
    echo "⚠ Warning: nvidia-smi not found. Running on CPU may be slow."
fi
echo ""

# Step 2: Check if data is prepared
echo "Step 2: Checking Europarl data..."
if [ ! -f "$DATA_DIR/train.bin" ] || [ ! -f "$DATA_DIR/val.bin" ]; then
    echo "Data not found. Preparing Europarl dataset..."
    python data/europarl/prepare.py --language_pair en-fr
    echo "✓ Data prepared"
else
    echo "✓ Data already prepared"
fi
echo ""

# Step 3: Train model
echo "Step 3: Training tiny Backpack model (~500K params)..."
echo "Config: config/$CONFIG.py"
echo "Output: $OUT_DIR"
echo ""
python train.py \
    --config "$CONFIG" \
    --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR"

echo ""
echo "✓ Training complete"
echo ""

# Step 4: Evaluate model
echo "Step 4: Running evaluation suite..."
echo ""
python run_full_evaluation.py \
    --out_dir "$OUT_DIR" \
    --device cuda

echo ""
echo "=================================================="
echo "PIPELINE COMPLETE!"
echo "=================================================="
echo ""
echo "Results saved to: $OUT_DIR/evaluation_results.json"
echo "Training logs: $OUT_DIR/training_log.json"
echo ""
echo "To view loss curves:"
echo "  python experiments/plot_loss_curves.py --log_file $OUT_DIR/training_log.json"
echo ""
