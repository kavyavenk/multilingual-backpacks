# Sense Vector Analysis Guide

## Quick Start

The improved sense vector analysis is already integrated into the evaluation scripts. Here's how to use it:

### Option 1: Run Full Evaluation (Recommended)

This runs sense vector analysis along with all other evaluations:

```bash
# For Backpack model
python run_full_evaluation.py --out_dir out/backpack_full --device cpu

# Skip MultiSimLex if dataset unavailable
python run_full_evaluation.py --out_dir out/backpack_full --device cpu --skip_multisimlex

# Skip translation evaluation if you only want sense analysis
python run_full_evaluation.py --out_dir out/backpack_full --device cpu --skip_translation
```

### Option 2: Compare Models Side-by-Side

Compare Backpack vs Transformer sense vectors:

```bash
python compare_models.py \
    --backpack_dir out/backpack_full \
    --transformer_dir out/transformer_full \
    --device cpu \
    --translation_samples 500
```

### Option 3: Use in Python Scripts

```python
from evaluate import load_model, analyze_sense_vectors, analyze_cross_lingual_sense_alignment
from transformers import AutoTokenizer
import torch

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, config = load_model('out/backpack_full', device)
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Analyze individual words
words = ['hello', 'bonjour', 'world', 'monde']
results = analyze_sense_vectors(
    model, tokenizer, words, device, 
    top_k=5,  # Show top 5 predictions per sense
    verbose=True  # Print detailed output
)

# Analyze cross-lingual alignment for translation pairs
word_pairs = [
    ('hello', 'bonjour'),
    ('world', 'monde'),
    ('parliament', 'parlement'),
]
alignment_results = analyze_cross_lingual_sense_alignment(
    model, tokenizer, word_pairs, device,
    top_k=5,
    verbose=True
)
```

## What You'll See

### Individual Word Analysis

For each word, you'll get:

```
======================================================================
Word: 'hello' (tokenized as: hello)
======================================================================

Top-5 Predictions per Sense:
----------------------------------------------------------------------

Sense  0 (entropy: 8.234, norm: 12.456):
  bonjour            45.23%
  salut              12.45%
  hi                  8.92%
  bonsoir            5.67%
  coucou              4.12%

Sense  1 (entropy: 7.891, norm: 11.234):
  sep                38.45%
  manque             15.23%
  risque              9.87%
  ...

======================================================================
Quantitative Metrics:
----------------------------------------------------------------------
  Mean Entropy:        8.123 ± 0.234
  Mean Magnitude:     11.845 ± 1.234
  Avg Sense Similarity: 0.456
  Number of Senses:    16

======================================================================
Sense Similarity Matrix (cosine similarity):
----------------------------------------------------------------------
       S 0  S 1  S 2  S 3 ...
 S 0  1.00 0.45 0.32 0.28 ...
 S 1  0.45 1.00 0.51 0.33 ...
 ...
```

### Cross-Lingual Alignment

For translation pairs:

```
======================================================================
Cross-Lingual Sense Alignment: 'hello' ↔ 'bonjour'
======================================================================

Alignment Pairs (EN Sense → FR Sense, similarity):
----------------------------------------------------------------------
  EN Sense  0 (bonjour        ) ↔ FR Sense  0 (bonjour        ): 0.892
  EN Sense  1 (sep            ) ↔ FR Sense  1 (sep            ): 0.756
  EN Sense  2 (salut          ) ↔ FR Sense  2 (salut          ): 0.823
  ...

======================================================================
Cross-Lingual Alignment Metrics:
----------------------------------------------------------------------
  Average Alignment Similarity: 0.823
  Max Alignment Similarity:       0.892
  Min Alignment Similarity:       0.645
  Number of Aligned Pairs:        16/16
```

## Parameters

### `analyze_sense_vectors()`

- `model`: BackpackLM model instance
- `tokenizer`: Tokenizer instance (e.g., XLM-RoBERTa)
- `words`: List of words to analyze (e.g., `['hello', 'bonjour']`)
- `device`: Device to run on (`'cpu'` or `'cuda'`)
- `top_k`: Number of top predictions to show per sense (default: 5)
- `verbose`: Whether to print detailed output (default: True)

### `analyze_cross_lingual_sense_alignment()`

- `model`: BackpackLM model instance
- `tokenizer`: Tokenizer instance
- `word_pairs`: List of (word_en, word_fr) tuples
- `device`: Device to run on
- `top_k`: Number of top predictions to show per sense (default: 5)
- `verbose`: Whether to print detailed output (default: True)

## Output Format

The function returns a dictionary with:

```python
{
    'word': {
        'predictions': [
            {
                'tokens': ['bonjour', 'salut', ...],
                'probs': [0.4523, 0.1245, ...],
                'top_prob': 0.4523
            },
            ...
        ],
        'entropies': [8.234, 7.891, ...],
        'sense_similarities': [[1.0, 0.45, ...], ...],
        'prediction_overlap': [[1.0, 0.2, ...], ...],
        'sense_norms': [12.456, 11.234, ...],
        'metrics': {
            'mean_entropy': 8.123,
            'std_entropy': 0.234,
            'mean_magnitude': 11.845,
            'std_magnitude': 1.234,
            'avg_sense_similarity': 0.456,
            'n_senses': 16
        }
    }
}
```

## Understanding the Metrics

### Entropy
- **High entropy** (>8): Sense predicts diverse tokens (more general)
- **Low entropy** (<6): Sense predicts specific tokens (more focused)

### Sense Similarity
- **High similarity** (>0.7): Senses are similar (redundant)
- **Low similarity** (<0.3): Senses capture different meanings (diverse)

### Cross-Lingual Alignment
- **High alignment** (>0.7): Translation pairs have aligned senses
- **Low alignment** (<0.4): Senses don't align across languages

## Examples

### Example 1: Analyze a single word

```python
from evaluate import load_model, analyze_sense_vectors
from transformers import AutoTokenizer

model, config = load_model('out/backpack_full', 'cpu')
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

results = analyze_sense_vectors(model, tokenizer, ['hello'], 'cpu', top_k=10)
```

### Example 2: Compare English and French words

```python
# Analyze both languages
en_words = ['hello', 'world', 'parliament']
fr_words = ['bonjour', 'monde', 'parlement']

en_results = analyze_sense_vectors(model, tokenizer, en_words, 'cpu')
fr_results = analyze_sense_vectors(model, tokenizer, fr_words, 'cpu')
```

### Example 3: Cross-lingual alignment

```python
from evaluate import analyze_cross_lingual_sense_alignment

word_pairs = [
    ('hello', 'bonjour'),
    ('world', 'monde'),
    ('parliament', 'parlement'),
]

alignment = analyze_cross_lingual_sense_alignment(
    model, tokenizer, word_pairs, 'cpu'
)
```

## Tips

1. **Use `verbose=False`** when you only need the data structure (not printed output)
2. **Increase `top_k`** to see more predictions per sense (useful for analysis)
3. **Check entropy values** to understand sense diversity
4. **Use cross-lingual alignment** to verify multilingual sense correspondence
5. **Compare similarity matrices** to identify redundant or unique senses

## Troubleshooting

### "Could not tokenize word"
- The word might not be in the tokenizer vocabulary
- Try using the tokenized form or a different word

### "Model is not BackpackLM"
- Sense vector analysis only works for Backpack models
- Use `load_model()` which automatically detects model type

### Memory issues
- Use `device='cpu'` if GPU memory is limited
- Analyze fewer words at a time
- Set `verbose=False` to reduce memory usage
