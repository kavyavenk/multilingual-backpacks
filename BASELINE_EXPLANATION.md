# Baseline Models Explanation

## Overview

This project uses two types of baselines for comparison:

1. **Pretrained Backpack Model** (for finetuning experiments)
2. **Standard Transformer** (for scratch training experiments)

---

## 1. Pretrained Backpack Baseline (Finetuning)

### What It Is
- **Model**: Hewitt et al.'s Small Backpack Language Model (`stanfordnlp/backpack-gpt2`)
- **Pretraining**: Trained on OpenWebText (English corpus)
- **Architecture**: Backpack model with sense vectors (not a standard transformer)

### Purpose
- Baseline for finetuning experiments
- Compare: multilingual finetuned performance vs original English-only performance
- Assess if multilingual finetuning maintains English performance while adding cross-lingual capabilities

### Usage
```bash
python train.py \
    --config train_europarl_finetune \
    --out_dir out-europarl-finetune \
    --data_dir europarl \
    --init_from backpack-small
```

### What Happens
1. Loads pretrained Backpack model from HuggingFace
2. Finetunes on Europarl French-English parallel data
3. Evaluates on lexical relatedness tasks
4. Compares to original English-only performance

---

## 2. Standard Transformer Baseline (Scratch Training)

### What It Is
- **Model**: `StandardTransformerLM` - a standard transformer language model
- **Architecture**: Identical to Backpack model EXCEPT:
  - Uses regular token embeddings (not sense embeddings)
  - No sense predictor network
  - No weighted combination of sense vectors
- **Training**: Trained from scratch on Europarl (same data as Backpack)

### Purpose
- Baseline for scratch training experiments
- Compare: Backpack model (with sense vectors) vs Standard transformer (without sense vectors)
- Same architecture, same data, same training setup - only difference is sense vectors

### Key Differences from Backpack

**Backpack Model Forward Pass:**
```python
# 1. Get sense embeddings (16 vectors per token)
sense_embs = self.sense_embeddings(idx)  # (B, T, 16, n_embd)

# 2. Predict sense weights
sense_weights = self.sense_predictor(context)  # (B, T, 16)

# 3. Weighted combination
x = weighted_sum(sense_embs, sense_weights)  # (B, T, n_embd)

# 4. Add position embeddings
x = x + pos_emb

# 5. Transformer blocks
x = self.blocks(x)
```

**Standard Transformer Forward Pass:**
```python
# 1. Get regular embeddings (1 vector per token)
token_embs = self.token_embeddings(idx)  # (B, T, n_embd)

# 2. Add position embeddings
x = token_embs + pos_emb

# 3. Transformer blocks (SAME as Backpack!)
x = self.blocks(x)
```

**Key Point**: The transformer blocks (`self.blocks`) are **identical** in both models!

### Parameter Comparison

**Standard Transformer** (scratch):
- Total: **202,819,968 parameters** (~203M)
- Token embeddings: 96,000,768 (250,002 × 384)
- Transformer blocks: 10,621,440 (same as Backpack)
- LM head: 96,000,768
- Other: ~197,000

**Backpack Model** (scratch):
- Total: **1,642,985,488 parameters** (~1.643B)
- Sense embeddings: 1,536,012,288 (250,002 × 384 × 16)
- Transformer blocks: 10,621,440 (same as Standard Transformer)
- LM head: 96,000,768
- Other: ~351,000

**Ratio**: Backpack is **8.1× larger** than Standard Transformer

The difference comes entirely from sense embeddings (16× more embedding parameters).

### Usage
```bash
python train.py \
    --config train_europarl_transformer_baseline \
    --out_dir out-transformer-baseline \
    --data_dir europarl \
    --init_from scratch
```

### Implementation

The `StandardTransformerLM` class is implemented in `model.py`:

```python
class StandardTransformerLM(nn.Module):
    """
    Standard Transformer Language Model (baseline)
    
    Identical to BackpackLM but without sense vectors.
    Uses regular token embeddings instead of sense embeddings.
    """
    
    def __init__(self, config):
        # Regular token embeddings (not sense embeddings)
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        
        # Same position embeddings, transformer blocks, LM head as Backpack
        self.pos_embeddings = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        # Simple: token embeddings + position embeddings → transformer blocks
        token_embs = self.token_embeddings(idx)
        pos_emb = self.pos_embeddings(pos)
        x = token_embs + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        return logits, loss
```

### Configuration

The transformer baseline uses the same config as Backpack scratch training:
- File: `config/train_europarl_transformer_baseline.py`
- Same architecture: 6 layers, 6 heads, 384 embedding dim
- Same training: batch size 32, learning rate 3e-4, etc.
- Only difference: `n_senses=1` (not used, but kept for compatibility)

---

## Comparison Summary

| Aspect | Pretrained Backpack | Standard Transformer | Your Backpack |
|--------|---------------------|---------------------|---------------|
| **Model Type** | Backpack (with senses) | Standard Transformer | Backpack (with senses) |
| **Pretraining** | OpenWebText (English) | None | None |
| **Training Data** | Europarl (finetune) | Europarl (scratch) | Europarl (scratch) |
| **Parameters** | ~3.35B | ~203M | ~1.643B |
| **Purpose** | Finetuning baseline | Scratch baseline | Your model |

---

## Why These Baselines?

### For Finetuning:
- **Question**: Does multilingual finetuning hurt English performance?
- **Baseline**: Original English-only Backpack model
- **Compare**: Finetuned multilingual vs original English-only

### For Scratch Training:
- **Question**: Do sense vectors help with multilingual learning?
- **Baseline**: Standard transformer (same arch, same data)
- **Compare**: Backpack (with senses) vs Transformer (without senses)

Both models trained identically, only difference is sense vectors!

---

## Files

- **Model Implementation**: `model.py` (contains both `BackpackLM` and `StandardTransformerLM`)
- **Transformer Config**: `config/train_europarl_transformer_baseline.py`
- **Training Script**: `train.py` (automatically detects transformer baseline config)

---

## Verification

To verify the transformer baseline works:

```bash
# Test parameter count
python -c "from model import StandardTransformerLM; from configurator import get_config; config = get_config('train_europarl_transformer_baseline'); config.vocab_size = 250002; model = StandardTransformerLM(config); print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"
```

Expected output: ~202,819,968 parameters

