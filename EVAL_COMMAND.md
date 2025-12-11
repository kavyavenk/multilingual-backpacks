# Evaluation Command - Local Run

## Quick Command
```bash
cd /Users/itskavya/Documents/multilingual-backpacks
python run_all_models_evaluation.py
```

## What Gets Evaluated

### Models Evaluated:
1. **Backpack** (`out/backpack_full/ckpt.pt`)
2. **Finetuned Backpack** (`out/finetuning_best_model_weights.pt`)  
3. **Transformer** (`out/transformer_full/ckpt.pt`)

### Metrics Evaluated:
- ✅ **BLEU Score** - Translation quality (Target: >0.30)
- ✅ **Translation Accuracy** - Word-level, Character-level, Exact Match
- ✅ **Sentence Similarity** - Cross-lingual similarity (Target: >0.70)
- ✅ **Perplexity** - Language modeling quality (Lower is better)
- ✅ **Word Similarity** - MultiSimLex (Monolingual & Cross-lingual)
- ✅ **Sense Vector Analysis** - 16 labeled senses

## Output Files

Results are saved to:
- `out/backpack_full/evaluation_results.json`
- `out/backpack_finetuned_evaluation_results.json`  
- `out/transformer_full/evaluation_results.json`
- `out/model_comparison.json`

## Syntax Verification

All evaluation functions have been verified:
- ✅ `evaluate_translation_bleu` - Syntax OK
- ✅ `evaluate_perplexity` - Syntax OK  
- ✅ `evaluate_sentence_similarity` - Syntax OK
- ✅ `evaluate_translation_accuracy` - Syntax OK
- ✅ All imports working correctly

## Recent Improvements

1. **Translation Generation**:
   - Lowered similarity thresholds (0.20 for matches, 0.30 for fallback)
   - Enhanced French word detection with suffix matching
   - Case-insensitive dictionary lookup
   - Multi-layer fallback logic

2. **Code Cleanup**:
   - Removed unreachable code
   - Fixed token counting in perplexity evaluation
   - Simplified translation logic

## Expected Runtime
- ~10-30 minutes depending on GPU/CPU
- Uses 500 translation samples by default
- Can be interrupted with Ctrl+C (saves partial results)

## GPU Usage
Automatically detects CUDA if available. To force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python run_all_models_evaluation.py
```

## Troubleshooting

If you encounter errors:
1. Check that checkpoint files exist:
   ```bash
   ls -lh out/backpack_full/ckpt.pt
   ls -lh out/finetuning_best_model_weights.pt
   ls -lh out/transformer_full/ckpt.pt
   ```

2. Verify data directory exists:
   ```bash
   ls -d data/europarl
   ```

3. Check Python dependencies:
   ```bash
   python -c "import torch; import transformers; import numpy; print('All dependencies OK')"
   ```

## High Scores Expected

With recent improvements, we expect:
- **BLEU**: 0.18-0.25 (targeting >0.30)
- **Word Accuracy**: 15-25%
- **Character Accuracy**: 55-65%
- **Sentence Similarity**: >0.50 (targeting >0.70)
- **Perplexity**: <35 (lower is better)
