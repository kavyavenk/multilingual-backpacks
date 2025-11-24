# Loss Curves and Top Activating Words Guide

This guide explains how to use the new loss curve tracking and top activating words functionality.

## Features:

1. **Automatic Loss Logging**: Training and validation losses are automatically saved to `training_log.json`
2. **Top Activating Words**: For Backpack models, tracks which words have highest sense vector activations
3. **Visualization Scripts**: Plot loss curves and analyze word activations

## Quick Start

### 1. Train a Tiny Model (~500K parameters)

```bash
python train.py \
    --config train_europarl_tiny \
    --out_dir out/tiny_experiment \
    --data_dir europarl
```

**Note**: The tiny config (`train_europarl_tiny`) is designed for quick experiments. Actual parameter count depends on vocab_size. With a vocab of ~2000 tokens, you'll get approximately 500K-600K parameters.

### 2. View Loss Curves

After training, plot the loss curves:

```bash
python experiments/plot_loss_curves.py \
    --log_file out/tiny_experiment/training_log.json \
    --output figures/loss_curves.pdf
```

### 3. Analyze Top Activating Words

For Backpack models, analyze which words have highest sense activations:

```bash
python experiments/analyze_top_activating_words.py \
    --log_file out/tiny_experiment/training_log.json \
    --top_n 20 \
    --plot_evolution \
    --output figures/top_words.pdf
```

## Training Log Format

The `training_log.json` file contains:

```json
{
  "iterations": [0, 100, 200, ...],
  "train_loss": [4.5, 3.2, 2.8, ...],
  "val_loss": [4.6, 3.3, 2.9, ...],
  "top_activating_words": [
    {
      "iteration": 100,
      "words": [
        {"token_id": 123, "word": "hello", "activation": 0.85},
        ...
      ]
    },
    ...
  ]
}
```

## Configuration

### Tiny Model Config

The `train_europarl_tiny` config uses:
- `block_size=128`: Small context window
- `n_layer=2`: Only 2 transformer layers
- `n_head=2`: 2 attention heads
- `n_embd=48`: Small embedding dimension
- `n_senses=4`: Minimal sense vectors
- `eval_interval=100`: Frequent evaluation for better curves

### Adjusting for True 500K Parameters

To get exactly ~500K parameters, you may need to:
1. Use a smaller vocabulary (subset of tokens)
2. Further reduce `n_embd` or `n_senses`
3. Use `experiments/calculate_model_params.py` to verify parameter count

## Visualization Options

### Single Model Loss Curves

```bash
python experiments/plot_loss_curves.py \
    --log_file out/model1/training_log.json \
    --output loss_curves.pdf
```

### Compare Multiple Models

```bash
python experiments/plot_loss_curves.py \
    --log_file out/model1/training_log.json \
    --compare out/model2/training_log.json out/model3/training_log.json \
    --labels "Backpack" "Transformer" "Baseline" \
    --output comparison.pdf
```

### Top Activating Words Analysis

```bash
# Print top 20 words
python experiments/analyze_top_activating_words.py \
    --log_file out/tiny_experiment/training_log.json \
    --top_n 20

# Plot evolution over training
python experiments/analyze_top_activating_words.py \
    --log_file out/tiny_experiment/training_log.json \
    --top_n 10 \
    --plot_evolution \
    --output top_words_evolution.pdf
```

## Example Workflow

```bash
# 1. Train tiny model
python train.py --config train_europarl_tiny --out_dir out/tiny --data_dir europarl

# 2. Plot loss curves
python experiments/plot_loss_curves.py --log_file out/tiny/training_log.json --output figures/tiny_loss.pdf

# 3. Analyze top words
python experiments/analyze_top_activating_words.py \
    --log_file out/tiny/training_log.json \
    --top_n 15 \
    --plot_evolution \
    --output figures/tiny_top_words.pdf
```

## Notes

- Loss logging happens automatically during training
- Top activating words are tracked every `eval_interval/2` iterations (for Backpack models only)
- The JSON log file is updated incrementally, so you can monitor training progress
- For very long training runs, the log file may become large; consider periodic cleanup

## Troubleshooting

**No top activating words in log?**
- Make sure you're training a Backpack model (not transformer baseline)
- Check that tokenizer loaded successfully
- Top words are only tracked periodically (every `eval_interval/2`)

**Loss curves look flat?**
- Check that `eval_interval` is appropriate (not too large)
- Verify training is actually progressing (check console output)
- Ensure model is learning (loss should decrease)

**Parameter count too high?**
- Use `experiments/calculate_model_params.py` to verify
- Reduce vocab_size, n_embd, or n_senses
- Consider using a smaller tokenizer vocabulary

