# Quick Start Guide: Training on Colab GPU

## Steps to Run on Colab

### 1. Upload to Colab
Open the notebook in Colab:
- Upload `train_tiny_gpu.ipynb` to Google Colab
- Or open directly from GitHub after pushing

### 2. Connect to T4 GPU
- Runtime → Change runtime type → T4 GPU → Save

### 3. Run All Cells
The notebook will automatically:
1. ✓ Check GPU (nvidia-smi)
2. ✓ Clone your repo
3. ✓ Install dependencies
4. ✓ Prepare Europarl data (10k samples)
5. ✓ Configure for GPU
6. ✓ Train model (5000 iterations, ~20-30 min)
7. ✓ Plot loss curves
8. ✓ Run evaluations
9. ✓ Download results

### 4. Expected Results

**Training (~25 minutes on T4)**:
- Initial loss: ~12.4
- Final loss: ~3-5 (lower is better)
- Parameters: ~500K

**Evaluations**:
- Cross-lingual similarity score
- Sense vector analysis
- Loss curves visualization

### 5. Download Results
The notebook will package everything as `tiny_model_results.tar.gz`:
- Model checkpoint
- Training logs
- Evaluation metrics
- Loss curve plots

## Alternative: Use Existing Prepared Data

If you want to skip data preparation (faster):

1. Upload your local prepared data to Colab:
   ```python
   from google.colab import files
   # Upload data/europarl/train.bin, val.bin, meta.pkl
   ```

2. Skip the "Prepare Data" section in the notebook

## Monitoring Training

Watch the output for:
```
iter 0: loss 12.4443
iter 100: loss 8.2341
iter 200: loss 6.1234
...
iter 5000: loss 3.4567
```

Lower loss = better learning!

## Troubleshooting

**Out of memory?**
- Reduce batch_size in cell 7 (32 → 16)

**Too slow?**
- Reduce max_iters (5000 → 1000)
- Use smaller dataset (--max_samples 5000)

**GPU not detected?**
- Check Runtime → Change runtime type → GPU enabled
