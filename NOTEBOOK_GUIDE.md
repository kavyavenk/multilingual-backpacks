# Jupyter Notebook Evaluation Guide

## `evaluate_tiny_model.ipynb`

Complete evaluation suite for the tiny Backpack model with GPU support.

## Features

✅ **All evaluation methods included:**
- Word-level representations (English & French)
- Cross-lingual word similarity
- Sense vector analysis (Backpack models)
- Sentence-level representations
- Cross-lingual sentence similarity
- MultiSimLex benchmark evaluation
- Training loss curve visualization
- Top activating words analysis

✅ **GPU Support:**
- Automatically detects and uses CUDA if available
- Falls back to CPU if GPU not available
- Optimized for GPU evaluation

✅ **Access to all GitHub code:**
- Imports from local codebase
- Uses all evaluation functions from `evaluate.py`
- Works with both Backpack and Transformer models

## Usage

### 1. Start Jupyter

```bash
# Install jupyter if needed
pip install jupyter

# Start Jupyter notebook server
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### 2. Open the Notebook

- Navigate to `evaluate_tiny_model.ipynb`
- Click to open

### 3. Run All Cells

- **Option 1**: Run cells one by one (Shift+Enter)
- **Option 2**: Run all cells: `Cell → Run All`
- **Option 3**: Use "Run All Above" to run up to a specific cell

## Prerequisites

Make sure you have:

1. **Trained model checkpoint**:
   ```bash
   # Train the tiny model first
   python train.py --config train_europarl_tiny --out_dir out/tiny --data_dir europarl
   ```

2. **Required libraries**:
   ```bash
   pip install torch transformers scipy matplotlib datasets numpy
   ```

3. **Model checkpoint exists**:
   ```bash
   ls out/tiny/ckpt.pt  # Should exist
   ```

## Notebook Structure

1. **Setup** (Cells 1-3):
   - Import all dependencies
   - GPU detection and setup
   - Load model and tokenizer

2. **Word-Level Evaluation** (Cells 4-6):
   - Extract word representations
   - Compute cross-lingual similarities

3. **Sense Vector Analysis** (Cells 7-8):
   - Analyze sense vectors (Backpack only)
   - Show top predictions per sense

4. **Sentence-Level Evaluation** (Cells 9-11):
   - Extract sentence representations
   - Compute cross-lingual sentence similarities

5. **MultiSimLex Benchmark** (Cells 12-14):
   - Monolingual evaluation (EN, FR)
   - Cross-lingual evaluation
   - Benchmark comparison

6. **Training Curves** (Cells 15-16):
   - Plot loss curves from training log
   - Display top activating words

7. **Summary Report** (Cell 17):
   - Complete evaluation summary
   - All results in one place

## Expected Output

The notebook will produce:

- **Word representations**: Shape information for each word
- **Similarity scores**: Cosine similarities between word/sentence pairs
- **Sense analysis**: Top-k predictions for each sense vector
- **MultiSimLex scores**: Spearman correlations with benchmark levels
- **Loss curves**: Visual plots of training progress
- **Summary**: Complete evaluation report

## GPU Usage

The notebook automatically:
- Detects CUDA availability
- Moves model to GPU
- Uses GPU for all computations
- Falls back to CPU if GPU unavailable

To force CPU usage, modify cell 2:
```python
device = 'cpu'  # Force CPU
```

## Troubleshooting

**Import errors?**
- Make sure you're running from the project root directory
- Check that all files are in the repository

**Model not found?**
- Train the model first: `python train.py --config train_europarl_tiny --out_dir out/tiny --data_dir europarl`
- Check that `out/tiny/ckpt.pt` exists

**GPU not detected?**
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch if needed
- Notebook will automatically use CPU as fallback

**MultiSimLex fails?**
- Install datasets: `pip install datasets`
- Check internet connection (downloads dataset on first use)

## Quick Start

```bash
# 1. Train model (if not already done)
python train.py --config train_europarl_tiny --out_dir out/tiny --data_dir europarl

# 2. Start Jupyter
jupyter notebook

# 3. Open evaluate_tiny_model.ipynb

# 4. Run all cells (Cell → Run All)
```

## Notes

- The notebook uses the codebase directly (no need to install as package)
- All evaluation functions are imported from `evaluate.py`
- Results are displayed inline with visualizations
- Can be run multiple times to track progress
- Works with both Backpack and Transformer baseline models

