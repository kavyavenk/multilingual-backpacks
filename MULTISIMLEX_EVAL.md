# MultiSimLex Evaluation Guide

## Overview

MultiSimLex is a multilingual word similarity benchmark that evaluates how well models capture semantic similarity between word pairs. This evaluation replaces ad-hoc word pair testing with standardized, human-annotated similarity scores.

## Performance Benchmarks

### Monolingual Evaluation (English/French)

- **Excellent (≥0.70 EN, ≥0.65 FR)**: Strong multilingual models (XLM-RoBERTa, mBERT)
- **Good (≥0.60 EN, ≥0.55 FR)**: Good multilingual models
- **Baseline (≥0.45 EN, ≥0.40 FR)**: Basic word embeddings (Word2Vec, GloVe)
- **Needs Improvement (<baseline)**: Random or very weak models

### Cross-lingual Evaluation

- **Excellent (≥0.60)**: Strong cross-lingual alignment
- **Good (≥0.50)**: Good cross-lingual alignment
- **Baseline (≥0.35)**: Basic cross-lingual models
- **Needs Improvement (<baseline)**: Poor cross-lingual alignment

## Usage

### Basic Evaluation

```bash
# Run MultiSimLex evaluation for English and French
python evaluate.py \
    --out_dir out/tiny \
    --multisimlex \
    --languages en fr

# Run with cross-lingual evaluation
python evaluate.py \
    --out_dir out/tiny \
    --multisimlex \
    --cross_lingual \
    --languages en fr
```

### Full Evaluation Suite

```bash
# Run all evaluations including MultiSimLex
python evaluate.py \
    --out_dir out/tiny \
    --device cuda \
    --tokenizer_name xlm-roberta-base \
    --multisimlex \
    --cross_lingual \
    --languages en fr
```

## Output Format

The evaluation will print:

1. **Spearman correlation**: How well model similarities match human judgments
2. **P-value**: Statistical significance
3. **Number of pairs**: How many word pairs were evaluated
4. **Benchmark comparison**: Performance level (Excellent/Good/Baseline/Needs Improvement)

Example output:
```
============================================================
MultiSimLex Evaluation - EN
============================================================
Processing 999 word pairs...

Results:
  Spearman correlation: 0.6234
  P-value: 0.0000
  Number of pairs: 999
  Skipped pairs: 0

Benchmark Comparison:
  Performance level: GOOD
  Excellent threshold: 0.70
  Good threshold: 0.60
  Baseline threshold: 0.45
```

## What Gets Evaluated

### Monolingual Evaluation
- Tests semantic similarity within a single language
- Compares model's cosine similarity with human-annotated scores
- Evaluates if the model captures word meanings correctly

### Cross-lingual Evaluation
- Tests if translation pairs have high similarity
- Evaluates multilingual alignment quality
- Important for multilingual models trained on parallel data

## Expected Results

### For Tiny Models (~500K params)
- **Monolingual**: 0.30-0.45 (Baseline to Needs Improvement)
- **Cross-lingual**: 0.25-0.40 (Baseline to Needs Improvement)

### For Larger Models (scratch config)
- **Monolingual**: 0.45-0.60 (Baseline to Good)
- **Cross-lingual**: 0.35-0.50 (Baseline to Good)

### For Finetuned Models
- **Monolingual**: 0.55-0.70 (Good to Excellent)
- **Cross-lingual**: 0.45-0.60 (Good to Excellent)

## Integration with Training

The MultiSimLex evaluation can be run:
1. **After training**: Evaluate final model performance
2. **During training**: Periodically evaluate to track improvement
3. **For comparison**: Compare Backpack vs Transformer baseline

## Notes

- Requires `datasets` library: `pip install datasets`
- MultiSimLex dataset is automatically downloaded on first use
- Some word pairs may be skipped if words aren't in tokenizer vocabulary
- Cross-lingual evaluation requires aligned word pairs in the dataset

## References

- Vulic et al. (2020): "Multi-SimLex: A Large-Scale Multilingual Evaluation Dataset for Lexical Similarity"
- Dataset: https://huggingface.co/datasets/Helsinki-NLP/multisimlex

