# Running Evaluations on Tiny Backpack Model (~500K params)

## ✓ Data Preparation Complete

Europarl dataset prepared with:
- 10,000 parallel sentences (en-fr)
- ~20,000 total texts (both directions)
- Vocabulary: 250,002 tokens
- Files: `data/europarl/train.bin`, `data/europarl/val.bin`

## Option 1: Run on GCP GPU (Recommended)

### Step 1: Create and setup GPU instance
```bash
./run_on_gcp_gpu.sh
```

This will:
1. Create a GCP GPU instance (T4)
2. Transfer your code
3. Install dependencies  
4. Run training + evaluation
5. Download results back

**Cost**: ~$0.35/hour for T4 GPU
**Time**: ~30-60 minutes for 5000 iterations

### Step 2: View results
```bash
cat out/tiny/evaluation_results.json
python experiments/plot_loss_curves.py --log_file out/tiny/training_log.json
```

---

## Option 2: Run Locally (CPU/GPU)

### Check if you have GPU
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Train the model
```bash
python train_tiny.py
```

This will:
- Train for 5000 iterations (~20-30 min on GPU, ~2 hours on CPU)
- Save checkpoint to `out/tiny/ckpt.pt`
- Create training logs

### Run evaluations
```bash
python run_full_evaluation.py --out_dir out/tiny --device cuda
```

Or skip MultiSimLex if dataset unavailable:
```bash
python run_full_evaluation.py --out_dir out/tiny --skip_multisimlex
```

---

## Option 3: Use Complete Pipeline Script

```bash
./train_and_eval.sh
```

Runs everything in sequence:
1. Check data
2. Train model
3. Run all evaluations
4. Generate reports

---

## What the Evaluation Will Show

### 1. Model Info
- Number of parameters
- Architecture details
- Training configuration

### 2. Sense Vector Analysis
For each word (e.g., "hello", "bonjour", "parliament"):
- What each of the 4 senses predicts
- Top 5 tokens per sense
- Whether senses capture different meanings

**Example Output**:
```
Word: parliament
  Sense 0 → ['parliament', 'parliamentary', 'mps', 'commission']
  Sense 1 → ['parlement', 'européen', 'députés']
  Sense 2 → ['assembly', 'legislature', 'congress']
  Sense 3 → ['debate', 'session', 'vote']
```

### 3. Word Similarity (if MultiSimLex available)
- English similarity: Spearman correlation
- French similarity: Spearman correlation  
- **Cross-lingual similarity**: Spearman correlation (KEY METRIC!)

**Interpretation**:
- > 0.5: Excellent cross-lingual alignment
- 0.35-0.5: Good alignment
- 0.2-0.35: Weak alignment
- < 0.2: Failed to align languages

### 4. Results Files
- `out/tiny/ckpt.pt` - Model checkpoint
- `out/tiny/training_log.json` - Training loss curves
- `out/tiny/evaluation_results.json` - All evaluation metrics

---

## Quick Start (Recommended Path)

1. **If you have GCP GPU quota**:
   ```bash
   ./run_on_gcp_gpu.sh
   ```

2. **If running locally with GPU**:
   ```bash
   python train_tiny.py && python run_full_evaluation.py --out_dir out/tiny
   ```

3. **If running locally on CPU** (slow):
   ```bash
   # Reduce iterations for faster testing
   python train_tiny.py  # Will take ~2 hours
   python run_full_evaluation.py --out_dir out/tiny --skip_multisimlex
   ```

---

## Expected Timeline

| Step | GPU Time | CPU Time |
|------|----------|----------|
| Data prep | ✓ Complete | ✓ Complete |
| Training (5000 iter) | 20-30 min | 2-3 hours |
| Evaluation | 2-5 min | 5-10 min |
| **Total** | **30-40 min** | **2-4 hours** |

---

## Next Steps After Evaluation

1. **Analyze results**: Check cross-lingual correlation
2. **Plot loss curves**: See if model converged
3. **Compare with baseline**: Train transformer baseline
4. **Scale up**: If working, train larger model (2M-10M params)

Ready to start? Choose your option and run!
