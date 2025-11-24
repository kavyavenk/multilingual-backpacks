# Standard Transformer Baseline Implementation

## Quick Start

To train the standard transformer baseline (for comparison with Backpack):

```bash
python train.py \
    --config train_europarl_transformer_baseline \
    --out_dir out-transformer-baseline \
    --data_dir europarl \
    --init_from scratch
```

## What Was Implemented

### 1. StandardTransformerLM Class (`model.py`)

A new model class that is identical to `BackpackLM` but without sense vectors:

- ✅ Regular token embeddings (instead of sense embeddings)
- ✅ Same transformer blocks (CausalSelfAttention + MLP)
- ✅ Same position embeddings
- ✅ Same language modeling head
- ✅ Same optimizer configuration
- ✅ Same text generation capability

**Key Difference**: No sense predictor, no weighted combination of sense vectors.

### 2. Configuration File (`config/train_europarl_transformer_baseline.py`)

Configuration identical to Backpack scratch training:
- Same architecture: 6 layers, 6 heads, 384 embedding dim
- Same training hyperparameters
- Same data and tokenizer

### 3. Training Script Updates (`train.py`)

- Automatically detects transformer baseline configs
- Instantiates `StandardTransformerLM` when config name contains "transformer_baseline"
- Supports resuming from checkpoint

## Parameter Comparison

**Standard Transformer**: 202,819,968 parameters (~203M)
**Backpack Model**: 1,642,985,488 parameters (~1.643B)

**Ratio**: Backpack is 8.1× larger

## Architecture Comparison

| Component | Standard Transformer | Backpack Model |
|-----------|---------------------|----------------|
| Token Embeddings | 96M (vocab × embd) | 1.54B (vocab × embd × 16) |
| Sense Predictor | None | 154K |
| Transformer Blocks | 10.6M | 10.6M (same!) |
| LM Head | 96M | 96M (same!) |
| **Total** | **203M** | **1.643B** |

## Usage Examples

### Train Transformer Baseline
```bash
python train.py \
    --config train_europarl_transformer_baseline \
    --out_dir out-transformer-baseline \
    --data_dir europarl
```

### Train Backpack Model (for comparison)
```bash
python train.py \
    --config train_europarl_scratch \
    --out_dir out-backpack-scratch \
    --data_dir europarl
```

### Evaluate Both Models
```bash
# Evaluate transformer baseline
python evaluate.py --out_dir out-transformer-baseline

# Evaluate Backpack model
python evaluate.py --out_dir out-backpack-scratch
```

### Compare Results
```bash
python experiments/analyze_results.py \
    --model_dirs out-transformer-baseline out-backpack-scratch \
    --output_dir comparison_results
```

## Verification

Test that the transformer baseline works:

```bash
python -c "
from model import StandardTransformerLM
from configurator import get_config

config = get_config('train_europarl_transformer_baseline')
config.vocab_size = 250002
model = StandardTransformerLM(config)
print(f'✓ Transformer baseline: {sum(p.numel() for p in model.parameters()):,} parameters')
"
```

Expected: ~202,819,968 parameters

## Files Created/Modified

1. **`model.py`**: Added `StandardTransformerLM` class
2. **`config/train_europarl_transformer_baseline.py`**: New config file
3. **`train.py`**: Updated to support transformer baseline
4. **`BASELINE_EXPLANATION.md`**: Detailed explanation document

## Next Steps

1. Train both models on Europarl
2. Compare validation losses
3. Evaluate on lexical relatedness tasks
4. Analyze differences in cross-lingual performance

