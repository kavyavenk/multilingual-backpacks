# Code Overview: Multilingual Backpack Project

## Table of Contents
1. [Metrics for Understanding the Backpack](#metrics)
2. [Code Explanation](#code-explanation)
3. [Progress on Proposed Plan](#progress)

---

## Metrics for Understanding the Backpack

### 1. **Word-Level Metrics** (`evaluate.py`)

#### a. **Word Representations**
- **What it measures**: Extracts sense vectors for individual words
- **Implementation**: `get_word_representations()` function
- **Output**: Dictionary mapping words to their sense vectors `(n_senses, n_embd)`
- **Use case**: Understand how words are represented as multiple sense vectors

#### b. **Cross-lingual Word Similarity**
- **What it measures**: Cosine similarity between translation pairs (e.g., "hello" vs "bonjour")
- **Implementation**: `evaluate_word_similarity()` function
- **Metric**: Cosine similarity between averaged sense vectors
- **Use case**: Check if translation pairs are close in embedding space

#### c. **Sense Vector Analysis**
- **What it measures**: What each sense vector predicts when projected through the LM head
- **Implementation**: `analyze_sense_vectors()` function
- **Output**: Top-k token predictions for each sense
- **Use case**: Understand what each sense vector represents semantically

### 2. **Sentence-Level Metrics** (`evaluate.py`)

#### a. **Sentence Representations**
- **What it measures**: Sentence-level embeddings using different pooling methods
- **Implementation**: `get_sentence_representation()` function
- **Methods**:
  - `mean`: Average across all tokens (as suggested in feedback)
  - `last`: Last token representation
  - `cls`: First token representation
- **Use case**: Compare sentence-level semantics across languages

#### b. **Cross-lingual Sentence Similarity**
- **What it measures**: Cosine similarity between translation sentence pairs
- **Implementation**: `evaluate_sentence_similarity()` function
- **Metric**: Cosine similarity between sentence representations
- **Use case**: Evaluate if the model captures semantic equivalence across languages

### 3. **Sense Vector Analysis Metrics** (`experiments/sense_vector.py`)

#### a. **Sense Projection Analysis**
- **What it measures**: What each sense vector predicts (top-k tokens)
- **Implementation**: `SenseVectorExperiment.sense_projection()`
- **Output**: List of top-k predictions for each sense
- **Use case**: Understand semantic meaning of each sense vector

#### b. **Multilingual Sense Alignment**
- **What it measures**: Compare sense vectors for translation pairs
- **Implementation**: `SenseVectorExperiment.analyze_multilingual_senses()`
- **Output**: Side-by-side comparison of English and French sense predictions
- **Use case**: Check if senses align across languages

### 4. **Visualization Metrics** (`experiments/visualize_senses.py`)

#### a. **Sense Vector Visualization**
- **What it measures**: 2D visualization of sense vectors using PCA or t-SNE
- **Implementation**: `visualize_sense_vectors()` function
- **Methods**: PCA or t-SNE dimensionality reduction
- **Use case**: Visual inspection of sense vector clustering and relationships

#### b. **Multilingual Sense Comparison**
- **What it measures**: Visual comparison of sense vectors for translation pairs
- **Implementation**: `compare_multilingual_senses()` function
- **Output**: Scatter plot showing English (blue) vs French (red) sense vectors
- **Use case**: Visual assessment of cross-lingual sense alignment

### 5. **Training Metrics** (`train.py`)

#### a. **Training Loss**
- **What it measures**: Cross-entropy loss on training data
- **Implementation**: Computed during forward pass
- **Use case**: Monitor training progress

#### b. **Validation Loss**
- **What it measures**: Cross-entropy loss on validation data
- **Implementation**: `estimate_loss()` function
- **Use case**: Track model generalization and prevent overfitting

#### c. **Best Validation Loss**
- **What it measures**: Lowest validation loss achieved
- **Implementation**: Checkpoint saving logic
- **Use case**: Model selection and comparison

### 6. **Results Analysis Metrics** (`experiments/analyze_results.py`)

#### a. **Model Comparison**
- **What it measures**: Compare metrics across different training runs
- **Implementation**: `compare_models()` function
- **Output**: Comparison of validation losses, iteration counts
- **Use case**: Compare scratch vs finetuned models

#### b. **Training Curves**
- **What it measures**: Plot training and validation loss over time
- **Implementation**: `plot_training_curves()` function
- **Use case**: Visualize training dynamics

#### c. **Sense Alignment Scores**
- **What it measures**: Overlap metric for sense predictions across languages
- **Implementation**: `analyze_sense_alignment()` function
- **Metric**: Jaccard similarity between top-k predictions
- **Use case**: Quantify cross-lingual sense alignment

---

## Code Explanation

### Core Architecture (`model.py`)

#### **BackpackLM Class**
The main model class implementing the Backpack architecture:

**Key Components:**
1. **Sense Embeddings** (`self.sense_embeddings`):
   - Each token has `n_senses` (default 16) sense vectors
   - Shape: `(vocab_size, n_senses * n_embd)`
   - Stores multiple representations per word

2. **Sense Predictor** (`self.sense_predictor`):
   - Neural network that predicts weights for combining sense vectors
   - Takes context (position embeddings initially) as input
   - Outputs `n_senses` weights (softmax normalized)

3. **Transformer Backbone**:
   - Standard transformer blocks (`Block` class)
   - Causal self-attention (`CausalSelfAttention`)
   - MLP layers (`MLP`)
   - Layer normalization

4. **Forward Pass Flow**:
   ```
   Input tokens → Sense embeddings → Sense weights (from predictor)
   → Weighted sum of sense vectors → Position embeddings
   → Transformer blocks → Layer norm → LM head → Logits
   ```

**Key Methods:**
- `forward()`: Main forward pass with loss computation
- `get_sense_vectors()`: Extract sense vectors for analysis
- `generate()`: Text generation with sampling
- `configure_optimizers()`: Set up AdamW optimizer with weight decay

### Training (`train.py`)

**Features:**
- Supports training from scratch, resuming, and finetuning
- Automatic checkpointing (saves best validation model)
- Configurable via config files
- Mixed precision training support (float16/bfloat16)
- Gradient clipping

**Training Loop:**
1. Load data (memmap for efficient loading)
2. Initialize model (scratch/resume/pretrained)
3. Set up optimizer
4. Training loop:
   - Sample batch
   - Forward pass
   - Backward pass
   - Gradient clipping
   - Optimizer step
   - Periodic evaluation and checkpointing

### Evaluation (`evaluate.py`)

**Comprehensive evaluation suite:**

1. **Word-level evaluation**:
   - Extract sense vectors for words
   - Compute cross-lingual similarities
   - Analyze what each sense predicts

2. **Sentence-level evaluation**:
   - Generate sentence representations (mean/last/cls pooling)
   - Compute sentence similarities
   - Cross-lingual sentence comparison

3. **Sense analysis**:
   - Project sense vectors through LM head
   - Get top-k predictions per sense
   - Compare senses across languages

### Data Preparation (`data/europarl/prepare.py`, `data/hansards/prepare.py`)

**Europarl preparation:**
- Downloads Europarl parallel corpus
- Tokenizes with XLM-RoBERTa tokenizer
- Creates train/val splits
- Saves binary format for efficient loading

**Features:**
- Language pair selection (en-fr)
- Optional language segregation
- Alignment file creation

### Configuration (`configurator.py`)

**ModelConfig dataclass:**
- Architecture parameters (layers, heads, embedding size, senses)
- Training hyperparameters (learning rate, batch size, etc.)
- Evaluation settings
- System settings (device, dtype, compile)

**Config files** (`config/`):
- `train_europarl_scratch.py`: Small model for scratch training
- `train_europarl_finetune.py`: Larger model for finetuning
- Similar configs for Hansards dataset

### Experiments (`experiments/`)

1. **sense_vector.py**: 
   - Sense projection analysis
   - Multilingual sense comparison

2. **visualize_senses.py**:
   - PCA/t-SNE visualization
   - Multilingual comparison plots

3. **analyze_results.py**:
   - Model comparison
   - Training curve plotting
   - Sense alignment analysis

### Sampling (`sample.py`)

**Text generation:**
- Load trained model
- Generate text from prompts
- Supports temperature and top-k sampling
- Can generate from file or string input

---

## Progress on Proposed Plan

### ✅ **Completed Components**

#### 1. **Backpack Model Architecture** 
- [x] Implemented `BackpackLM` class with sense vectors
- [x] Sense embeddings (n_senses per token)
- [x] Sense predictor network
- [x] Transformer backbone
- [x] Forward pass with weighted sense combination
- [x] Text generation capability

#### 2. **Training Infrastructure** 
- [x] Training script (`train.py`)
- [x] Support for training from scratch
- [x] Support for resuming from checkpoint
- [x] Framework for finetuning (config ready, loading needs implementation)
- [x] Checkpointing system
- [x] Validation evaluation
- [x] Multiple config files (scratch and finetune)

#### 3. **Data Preparation** 
- [x] Europarl data preparation script
- [x] Tokenization with XLM-RoBERTa
- [x] Train/val split creation
- [x] Binary format for efficient loading
- [x] Language segregation utilities

#### 4. **Evaluation Metrics** 
- [x] Word-level representation extraction
- [x] Sentence-level representation (mean pooling as per feedback)
- [x] Cross-lingual word similarity
- [x] Cross-lingual sentence similarity
- [x] Sense vector analysis (what each sense predicts)
- [x] Multilingual sense comparison

#### 5. **Analysis Tools** 
- [x] Sense vector visualization (PCA/t-SNE)
- [x] Multilingual sense comparison visualization
- [x] Results analysis and comparison
- [x] Training curve plotting

#### 6. **Configuration System** 
- [x] Configurator module
- [x] Config files for different scenarios
- [x] Easy switching between scratch and finetune configs

### ⚠️ **Partially Completed**

#### 1. **Finetuning**
- [x] Config files ready (`train_europarl_finetune.py`, `train_hansards_finetune.py`)
- [x] Training script supports `--init_from backpack-small`
- [ ] **Missing**: Actual loading of pretrained Backpack models
- **Status**: Framework is ready, but needs pretrained model checkpoint loading implementation

#### 2. **Training Logging**
- [x] Basic logging to console
- [ ] **Missing**: JSON log file creation for training curves
- **Status**: `analyze_results.py` expects JSON logs but `train.py` doesn't create them yet

### ❌ **Not Yet Implemented**

#### 1. **Advanced Evaluation Metrics**
- [ ] Bilingual lexicon induction evaluation
- [ ] Word sense disambiguation tasks
- [ ] Semantic similarity benchmarks (e.g., STS)
- [ ] Downstream task evaluation (e.g., classification)

#### 2. **Sense Vector Weight Analysis**
- [ ] Analysis of when different senses are activated
- [ ] Context-dependent sense weight visualization
- [ ] Sense weight distribution analysis

#### 3. **Comparative Analysis**
- [ ] Comparison with baseline transformer models
- [ ] Ablation studies (e.g., effect of n_senses)
- [ ] Performance on monolingual vs multilingual data

#### 4. **Documentation**
- [ ] Detailed results documentation
- [ ] Findings and insights documentation
- [ ] Performance benchmarks

---

## Summary

### What We Have:
1. **Complete Backpack architecture** with sense vectors
2. **Full training pipeline** (scratch training works, finetuning framework ready)
3. **Comprehensive evaluation suite** with word-level, sentence-level, and sense-level metrics
4. **Analysis tools** for visualization and comparison
5. **Data preparation** for Europarl and Hansards

### What's Ready but Needs Implementation:
1. **Pretrained model loading** for finetuning
2. **Training log file creation** for better analysis

### What's Missing:
1. **Advanced evaluation benchmarks**
2. **Comparative studies**
3. **Detailed results documentation**

### Key Metrics Available:
- Word-level similarity (cross-lingual)
- Sentence-level similarity (cross-lingual)
- Sense vector predictions
- Sense vector visualization
- Training/validation loss
- Multilingual sense alignment analysis

The codebase provides a solid foundation for understanding the Backpack model's behavior in multilingual settings, with comprehensive metrics for word-level, sentence-level, and sense-level analysis.

