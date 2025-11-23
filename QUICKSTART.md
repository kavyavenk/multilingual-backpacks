# Quick Start Guide

This guide will help you set up and run experiments for multilingual Backpack Language Models.

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Prepare the Europarl dataset:**
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

This creates separate files (`en.txt`, `fr.txt`) with tagged sentences and an alignment file for reference.

## Training

### Option 1: Train from Scratch

Train a small Backpack model from scratch on Europarl:

```bash
python train.py \
    --config train_europarl_scratch \
    --out_dir out-europarl-scratch \
    --data_dir europarl \
    --device cuda
```

### Option 2: Finetune Pre-trained Model

If you have a pre-trained Backpack model, finetune it:

```bash
python train.py \
    --config train_europarl_finetune \
    --out_dir out-europarl-finetune \
    --data_dir europarl \
    --init_from backpack-small \
    --device cuda
```

**Note:** Pre-trained model loading needs to be implemented based on available checkpoints.

## Evaluation

### Word-level Evaluation

Evaluate word representations and sense vectors:

```bash
python evaluate.py \
    --out_dir out-europarl-scratch \
    --device cuda
```

This will:
- Extract word representations for English and French words
- Analyze sense vectors
- Compute cross-lingual word similarities
- Evaluate sentence-level representations

### Sense Vector Analysis

Analyze what each sense vector represents:

```bash
python experiments/sense_vector.py \
    --out_dir out-europarl-scratch \
    --device cuda
```

## Sampling

Generate text from trained models:

```bash
python sample.py \
    --out_dir out-europarl-scratch \
    --start "Hello" \
    --num_samples 3 \
    --max_new_tokens 50
```

For French:
```bash
python sample.py \
    --out_dir out-hansards-scratch \
    --start "Bonjour" \
    --num_samples 3 \
    --max_new_tokens 50
```

## Expected Results

Based on the project proposal feedback, you should:

1. **Train small backpacks from scratch** on Europarl and evaluate
2. **Finetune the released backpack** on Europarl and see what happens
3. **Evaluate multilingual word representation** capabilities
4. **Analyze sense vectors** across languages to see if they capture cross-lingual patterns
5. **Use segregated language files** for reference and analysis

### Key Findings to Report

- How well do sense vectors align across languages?
- Can the model represent both English and French effectively?
- How do sentence-level representations compare for translation pairs?
- What patterns emerge in the sense vectors for multilingual words?

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in the config file:
```python
batch_size=16  # or smaller
```

### Dataset Not Found

If Europarl dataset is not available, the script will try OPUS100. Alternatively:
- Download Europarl manually from https://www.statmt.org/europarl/
- Use any French-English parallel corpus
- Modify `data/europarl/prepare.py` to load your data
- Use different language pairs: `--language_pair en-de` for English-German, etc.

### Slow Training

- Use `--compile` flag for faster training (PyTorch 2.0+)
- Reduce model size in config
- Use smaller `block_size`
- Train on GPU if available

## Next Steps

1. Run training experiments
2. Analyze sense vectors for multilingual words
3. Compare word and sentence representations across languages
4. Document findings in your project report

