# Europarl Dataset Setup

This document explains the Europarl dataset setup and the language segregation feature.

## Overview

The project now uses **Europarl** instead of Hansards as the primary dataset. Europarl is a parallel corpus from European Parliament proceedings, available in many language pairs.

## Key Features

### 1. Data Preparation (`data/europarl/prepare.py`)

Prepares the dataset for training:
- Downloads Europarl or OPUS100 dataset
- Tokenizes using XLM-RoBERTa tokenizer
- Creates `train.bin` and `val.bin` for efficient training
- Saves metadata

**Usage:**
```bash
python data/europarl/prepare.py --language_pair en-fr
```

### 2. Language Segregation (`data/europarl/segregate_languages.py`)

Creates separate language files with tags for reference:
- `en.txt`: English sentences with tags
- `fr.txt`: French sentences with tags
- `metadata.json`: Dataset metadata
- `alignment.json`: Parallel sentence mappings (optional)

**Tag Format:**
```
<sentence_id>|<language_pair>|<source>\t<sentence_text>
```

Example:
```
0|en-fr|europarl	This is an English sentence.
0|en-fr|europarl	Ceci est une phrase française.
```

**Usage:**
```bash
# Basic segregation
python data/europarl/segregate_languages.py --language_pair en-fr

# With alignment file
python data/europarl/segregate_languages.py --language_pair en-fr --create_alignment
```

### 3. Reading Segregated Files (`data/europarl/read_segregated.py`)

Helper script to work with segregated files:
- Read parallel pairs
- Get statistics
- Search by sentence ID
- Access alignment data

**Usage:**
```bash
# Show statistics and examples
python data/europarl/read_segregated.py --data_dir data/europarl --show_pairs 5

# Get specific sentence
python data/europarl/read_segregated.py --data_dir data/europarl --sentence_id 0
```

## Complete Workflow

### Step 1: Prepare Training Data
```bash
python data/europarl/prepare.py --language_pair en-fr
```

This creates:
- `data/europarl/train.bin`
- `data/europarl/val.bin`
- `data/europarl/meta.pkl`

### Step 2: Create Segregated Files (Reference)
```bash
python data/europarl/segregate_languages.py --language_pair en-fr --create_alignment
```

This creates:
- `data/europarl/en.txt` - English sentences with tags
- `data/europarl/fr.txt` - French sentences with tags
- `data/europarl/metadata.json` - Dataset info
- `data/europarl/alignment.json` - Parallel mappings

### Step 3: Train Model
```bash
python train.py --config train_europarl_scratch --out_dir out-europarl-scratch --data_dir europarl
```

### Step 4: Use Segregated Files for Analysis

The segregated files can be used for:
- **Evaluation**: Create evaluation sets with known parallel pairs
- **Analysis**: Study sentence-level patterns
- **Visualization**: Visualize parallel sentence pairs
- **Debugging**: Check data quality
- **Research**: Analyze cross-lingual patterns

## Language Pairs

Europarl supports many language pairs. Common ones:
- `en-fr` (English-French)
- `en-de` (English-German)
- `en-es` (English-Spanish)
- `en-it` (English-Italian)
- `fr-de` (French-German)
- And many more...

To use a different pair:
```bash
python data/europarl/prepare.py --language_pair en-de
python data/europarl/segregate_languages.py --language_pair en-de
```

## File Structure

After running both scripts:

```
data/europarl/
├── prepare.py              # Data preparation script
├── segregate_languages.py  # Language segregation script
├── read_segregated.py      # Helper to read segregated files
├── README.md               # Documentation
│
├── train.bin              # Training tokens (from prepare.py)
├── val.bin                # Validation tokens (from prepare.py)
├── meta.pkl               # Metadata (from prepare.py)
│
├── en.txt                 # English sentences with tags (from segregate_languages.py)
├── fr.txt                 # French sentences with tags (from segregate_languages.py)
├── metadata.json          # Dataset metadata (from segregate_languages.py)
└── alignment.json          # Parallel mappings (optional, from segregate_languages.py)
```

## Benefits of Segregation

1. **Reference**: Easy access to parallel sentences for analysis
2. **Evaluation**: Create evaluation sets with known pairs
3. **Debugging**: Check data quality and alignment
4. **Research**: Study cross-lingual patterns
5. **Visualization**: Visualize parallel sentence pairs
6. **Flexibility**: Work with individual languages separately

## Example: Using Segregated Files

```python
from data.europarl.read_segregated import SegregatedDataReader

# Initialize
reader = SegregatedDataReader('data/europarl', 'en-fr')

# Get statistics
stats = reader.get_statistics()
print(f"Total aligned pairs: {stats['aligned_pairs']}")

# Get parallel pairs for evaluation
pairs = reader.get_parallel_pairs(limit=100)
for sent_id, en_text, fr_text in pairs:
    # Use for evaluation, analysis, etc.
    pass

# Get specific sentence
pair = reader.search_by_id('42')
if pair:
    en_text, fr_text = pair
    print(f"EN: {en_text}")
    print(f"FR: {fr_text}")
```

## Migration from Hansards

The old Hansards configuration files are still available but have been updated:
- `config/train_hansards_scratch.py` → Updated to use Europarl
- `config/train_hansards_finetune.py` → Updated to use Europarl

New Europarl-specific configs:
- `config/train_europarl_scratch.py`
- `config/train_europarl_finetune.py`

Both work the same way - just use `--data_dir europarl` instead of `--data_dir hansards`.

