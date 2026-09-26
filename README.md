# Multilingual Backpack Language Models

This project implements multilingual Backpack Language Models for French-English, based on the nanoBackpackLM architecture. The project focuses on:

1. Training small Backpack models from scratch on Europarl (French-English parallel data)
2. Finetuning pre-trained Backpack models on multilingual data
3. Evaluating multilingual word representation capabilities
4. Analyzing sense vectors across languages

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Verification](#verification)
- [Code Overview](#code-overview)
- [Sense Vector Analysis](#sense-vector-analysis)
- [Baseline Models](#baseline-models)
- [MultiSimLex Evaluation](#multisimlex-evaluation)
- [Debiasing](#debiasing)

---

## Project Structure

```
.
├── data/
│   ├── europarl/          # Europarl dataset preparation
│   │   ├── prepare.py           # Main data preparation
│   │   ├── segregate_languages.py  # Create separate language files with tags
│   │   └── README.md            # Europarl-specific documentation
├── config/                # Configuration files for training
├── experiments/           # Evaluation and analysis scripts
├── model.py              # Backpack model architecture
├── train.py              # Training script
├── sample.py             # Sampling/inference script
├── evaluate.py           # Evaluation scripts
├── compare_models.py     # Model comparison script
└── verify_evaluation.py  # Verification script
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Europarl Dataset

```bash
python data/europarl/prepare.py --language_pair en-fr
```

This will download and tokenize the French-English parallel data. The script will:
- Download Europarl or OPUS100 dataset
- Tokenize using XLM-RoBERTa tokenizer
- Create `train.bin` and `val.bin` files
- Save metadata including vocabulary size

**Optional: Create segregated language files for reference:**
```bash
python data/europarl/segregate_languages.py --language_pair en-fr --create_alignment
```

---

## Quick Start

```bash
python train.py \
    --config train_europarl_scratch \
    --out_dir out-europarl-scratch \
    --data_dir europarl \
    --device cuda
```


### Evaluate

```bash
python evaluate.py --out_dir out-europarl-scratch
```

---

## Training

### Model Configurations

Both models use identical parameters:

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

### Training Commands

#### Train Backpack Model

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

#### Train Transformer Baseline

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

### Checkpoint Management

#### Automatic Checkpoint Saving

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

#### Resume Training

If training is interrupted, resume with:

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
```

The resume functionality automatically:
- Restores model weights
- Restores optimizer state (learning rate schedule, momentum, etc.)
- Restores iteration number (continues from where it left off)
- Restores best validation loss
- Restores training log

## Evaluation

```bash
# Evaluate Backpack
python evaluate.py \
    --out_dir out/backpack_full \
    --device cpu


### Evaluation Metrics

1. **Perplexity**

3. **Word Similarity**:
   - MultiSimLex evaluation (monolingual and cross-lingual)

3. **Sentence-Level Similarity**:
   - Cross-lingual sentence similarity (cosine similarity)




## Code Overview

### Core Architecture (`model.py`)

- **BackpackLM**: Implements Backpack Language Model with sense vectors
- **StandardTransformerLM**: Standard transformer baseline (no sense vectors)

### Training (`train.py`)

- Training loop with validation
- Checkpoint saving and resuming
- Training log generation

### Evaluation (`'run_ckpt_eval.py')

- Perplexity
- Delta_mu

## MultiSimLex Evaluation

MultiSimLex is a multilingual word similarity benchmark that evaluates how well models capture semantic similarity between word pairs.

```bash
# Run MultiSimLex evaluation
python evaluate.py \
    --out_dir out/backpack_full \
    --multisimlex \
    --languages en fr

# With cross-lingual evaluation
python evaluate.py \
    --out_dir out/backpack_full \
    --multisimlex \
    --cross_lingual \
    --languages en fr
```

## Debiasing





## References

- nanoBackpackLM repository: https://github.com/SwordElucidator/nanoBackpackLM
- Backpack Language Models paper repository: https://github.com/john-hewitt/backpacks-flash-attn
- XLM-RoBERTa: https://huggingface.co/xlm-roberta-base
- MultiSimLex: Multilingual word similarity benchmark https://aclanthology.org/2020.cl-4.5/
