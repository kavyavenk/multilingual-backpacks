# Full-Sized Model Training Guide

## Overview

This guide explains how to train the full-sized Backpack and Transformer baseline models with identical parameters (except for sense vectors).

## Model Configurations

Both models use **identical parameters**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `block_size` | 512 | Context length |
| `n_layer` | 6 | Transformer layers |
| `n_head` | 6 | Attention heads per layer |
| `n_embd` | 384 | Embedding dimension |
| `n_senses` | 16 (Backpack) / 1 (Transformer) | Only difference - Transformer doesn't use senses |
| `dropout` | 0.1 | Dropout rate |
| `bias` | False | No bias in LayerNorm/Linear |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 3e-4 | Learning rate |
| `max_iters` | 50,000 | Maximum training iterations |
| `weight_decay` | 1e-1 | Weight decay |
| `eval_interval` | 500 | Evaluate every N iterations |
| `eval_iters` | 200 | Number of eval batches |

**Parameter Counts:**
- **Backpack**: ~1.64B parameters (1,642,985,488)
- **Transformer**: ~203M parameters (202,985,488)
- The difference is due to sense embeddings (vocab_size × n_embd × n_senses)

## Training Commands

### Train Backpack Model

```bash
python train.py \
    --model_type backpack \
    --config train_europarl_scratch \
    --out_dir out/backpack_full \
    --data_dir europarl \
    --init_from scratch \
    --device cuda \
    --dtype float16 \
    --compile
```

### Train Transformer Baseline

```bash
python train.py \
    --model_type transformer \
    --config train_europarl_transformer_baseline \
    --out_dir out/transformer_full \
    --data_dir europarl \
    --init_from scratch \
    --device cuda \
    --dtype float16 \
    --compile
```

## Checkpoint Management

### Automatic Checkpoint Saving

The training script automatically saves checkpoints:
1. **Best validation loss**: Saves whenever validation loss improves
2. **Periodic saves**: Saves every 5 evaluation intervals (every 2,500 iterations)

Checkpoints are saved to:
- `{out_dir}/ckpt.pt` - Contains:
  - Model state dict
  - Optimizer state dict
  - Current iteration number
  - Best validation loss
  - Training log (losses, top activating words)

### Resume Training

If training is interrupted (e.g., GPU disconnection), resume with:

```bash
# Resume Backpack training
python train.py \
    --model_type backpack \
    --config train_europarl_scratch \
    --out_dir out/backpack_full \
    --data_dir europarl \
    --init_from resume \
    --device cuda \
    --dtype float16 \
    --compile

# Resume Transformer training
python train.py \
    --model_type transformer \
    --config train_europarl_transformer_baseline \
    --out_dir out/transformer_full \
    --data_dir europarl \
    --init_from resume \
    --device cuda \
    --dtype float16 \
    --compile
```

The resume functionality automatically:
- Restores model weights
- Restores optimizer state (learning rate schedule, momentum, etc.)
- Restores iteration number (continues from where it left off)
- Restores best validation loss
- Restores training log

## Training on GCP

### Quick Start Script

Use the provided script to train on GCP:

```bash
# Train Backpack
./run_on_gcp_gpu.sh backpack_full

# Train Transformer
./run_on_gcp_gpu.sh transformer_full
```

Or manually:

1. **Create GPU instance**:
   ```bash
   ./create_gpu_instance.sh
   ```

2. **Connect to instance**:
   ```bash
   ./connect_gcp_gpu.sh
   ```

3. **On the instance, train**:
   ```bash
   # Backpack
   python train.py --model_type backpack --config train_europarl_scratch \
       --out_dir out/backpack_full --data_dir europarl --init_from scratch \
       --device cuda --dtype float16 --compile

   # Transformer
   python train.py --model_type transformer --config train_europarl_transformer_baseline \
       --out_dir out/transformer_full --data_dir europarl --init_from scratch \
       --device cuda --dtype float16 --compile
   ```

4. **If disconnected, resume**:
   ```bash
   # Change --init_from scratch to --init_from resume
   python train.py --model_type backpack --config train_europarl_scratch \
       --out_dir out/backpack_full --data_dir europarl --init_from resume \
       --device cuda --dtype float16 --compile
   ```

## Monitoring Training

### Training Log

Training progress is logged to `{out_dir}/training_log.json`:
```json
{
  "iterations": [0, 500, 1000, ...],
  "train_loss": [4.5, 3.2, 2.8, ...],
  "val_loss": [4.6, 3.3, 2.9, ...],
  "top_activating_words": [
    {
      "iteration": 500,
      "words": [...]
    }
  ]
}
```

### Check Training Status

```bash
# View training log
cat out/backpack_full/training_log.json | python -m json.tool

# Check latest checkpoint
ls -lh out/backpack_full/ckpt.pt
```

## Evaluation

After training, evaluate the models:

```bash
# Evaluate Backpack
python evaluate.py \
    --model_path out/backpack_full/ckpt.pt \
    --config train_europarl_scratch \
    --model_type backpack

# Evaluate Transformer
python evaluate.py \
    --model_path out/transformer_full/ckpt.pt \
    --config train_europarl_transformer_baseline \
    --model_type transformer
```

## Notes

1. **Training Time**: With 50,000 iterations, expect several days of training on a single GPU. Checkpoints allow you to pause and resume.

2. **GPU Memory**: Both models fit on a T4 GPU (16GB) with batch_size=32. If you run out of memory, reduce `batch_size` in the config files.

3. **Parameter Verification**: The configs are verified to be identical except for `n_senses`. The Transformer baseline uses `n_senses=1` for compatibility but doesn't actually use it.

4. **Checkpoint Frequency**: Checkpoints are saved:
   - Every time validation loss improves (best model)
   - Every 2,500 iterations (periodic backup)

5. **Resume Safety**: Always use `--init_from resume` when resuming. The script will automatically detect and load the checkpoint from `{out_dir}/ckpt.pt`.

