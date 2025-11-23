# Project Setup Summary

This document summarizes what we have set up so far for our multilingual Backpack Language Model project.

## Project Overview

Based on our proposal feedback, this project implements:

1. **Training small Backpack models from scratch** on Europarl (French-English parallel data)
2. **Finetuning pre-trained Backpack models** on multilingual data
3. **Evaluating multilingual word representation** capabilities
4. **Analyzing sense vectors** across languages

## Project Structure

```
nlp-project-multilingual-backpacks/
├── README.md                    # Main project documentation
├── QUICKSTART.md                # Quick start guide
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
│
├── model.py                     # Backpack LM architecture
├── train.py                     # Training script
├── sample.py                    # Text generation script
├── evaluate.py                  # Evaluation script
├── configurator.py              # Configuration management
│
├── config/                      # Configuration files
│   ├── train_europarl_scratch.py    # Train from scratch config
│   └── train_europarl_finetune.py  # Finetune config
│
├── data/                        # Data preparation scripts
│   ├── europarl/                # Europarl dataset preparation
│   │   ├── prepare.py           # Main data preparation
│   │   ├── segregate_languages.py  # Create separate language files
│   │   └── README.md            # Europarl-specific documentation
│   └── hansards/                # (Legacy) Hansards dataset
│
└── experiments/                 # Analysis scripts
    ├── sense_vector.py          # Sense vector analysis
    └── analyze_results.py       # Results comparison
```

## Key Components

### 1. Backpack Model Architecture (`model.py`)

The `BackpackLM` class implements:
- **Sense vectors**: Each word has `n_senses` (default 16) sense vectors
- **Sense predictor**: Predicts weights for combining sense vectors based on context
- **Transformer backbone**: Standard transformer architecture for contextualization
- **Multilingual support**: Works with multilingual tokenizers (XLM-RoBERTa)

Key methods:
- `forward()`: Forward pass with sense vector combination
- `get_sense_vectors()`: Extract sense vectors for analysis
- `generate()`: Text generation

### 2. Data Preparation (`data/europarl/prepare.py`)

Prepares French-English parallel data:
- Downloads Europarl dataset
- Tokenizes with XLM-RoBERTa tokenizer
- Creates train/val splits
- Saves binary format for efficient loading
- Optional language segregation for reference

### 3. Training (`train.py`)

Supports:
- Training from scratch
- Resuming from checkpoint
- Finetuning pre-trained models (when available)

Features:
- Automatic checkpointing
- Validation evaluation
- Configurable via config files

### 4. Evaluation (`evaluate.py`)

Evaluates:
- **Word-level representations**: Extract sense vectors for words
- **Sentence-level representations**: Average/last/CLS token representations
- **Cross-lingual similarity**: Compare translation pairs
- **Sense vector analysis**: What each sense predicts

### 5. Sense Vector Analysis (`experiments/sense_vector.py`)

Analyzes:
- What each sense vector predicts (top-k tokens)
- Cross-lingual sense alignment
- Multilingual sense patterns

## Usage Workflow

### Step 1: Prepare Data
```bash
python data/europarl/prepare.py --language_pair en-fr
```

### Step 2: Train Model
```bash
# From scratch
python train.py --config train_europarl_scratch --out_dir out-europarl-scratch --data_dir europarl

# Finetune (when pre-trained model available)
python train.py --config train_europarl_finetune --out_dir out-europarl-finetune --data_dir europarl --init_from backpack-small
```

### Step 3: Evaluate
```bash
python evaluate.py --out_dir out-europarl-scratch
```

### Step 4: Analyze Sense Vectors
```bash
python experiments/sense_vector.py --out_dir out-europarl-scratch
```

### Step 5: Generate Samples
```bash
python sample.py --out_dir out-europarl-scratch --start "Hello" --num_samples 3
```

## Key Findings to Investigate

Based on your proposal feedback, focus on:

1. **Multilingual Word Representation**
   - Do sense vectors align across languages?
   - Can the model represent both English and French effectively?
   - How do translation pairs compare in embedding space?

2. **Sentence-level Representations**
   - Average across sequence length (as suggested in feedback)
   - Compare sentence representations for translation pairs
   - Evaluate semantic similarity

3. **Training Comparison**
   - Compare training from scratch vs. finetuning
   - Analyze how multilingual data affects English performance
   - Document any improvements or regressions

4. **Sense Vector Analysis**
   - What patterns emerge in sense vectors?
   - Do different senses capture different meanings/contexts?
   - Are there cross-lingual sense correspondences?

## Implementation Notes

### Differences from Standard Transformers

1. **Sense Embeddings**: Each token has multiple sense vectors instead of a single embedding
2. **Sense Prediction**: Context-dependent weighting of sense vectors
3. **Weighted Combination**: Final representation is a weighted sum of sense vectors

### Multilingual Considerations

1. **Tokenizer**: Uses XLM-RoBERTa tokenizer (handles both English and French)
2. **Language Separation**: Uses `<|lang_sep|>` token to help model distinguish languages
3. **Parallel Data**: Interleaves English-French sentence pairs for training

### Evaluation Strategy

As per feedback:
- **Word-level**: Extract sense vectors and compare translation pairs
- **Sentence-level**: Average representations across sequence length
- **Cross-lingual**: Measure similarity between translation pairs

## Next Steps

1. **Run experiments**:
   - Train from scratch on Europarl
   - Finetune if pre-trained model available
   - Evaluate both approaches

2. **Analyze results**:
   - Compare word representations
   - Analyze sense vectors
   - Evaluate sentence representations

3. **Document findings**:
   - How well does Backpack work for multilingual data?
   - What patterns emerge in sense vectors?
   - How do results compare to monolingual training?

## References

- nanoBackpackLM: https://github.com/SwordElucidator/nanoBackpackLM
- Backpack Language Models paper
- Your project proposal and feedback

## Troubleshooting

See `QUICKSTART.md` for common issues and solutions.

