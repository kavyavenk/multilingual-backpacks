# Sense Distinction Analysis Guide

## Problem: Why Are Senses So Similar?

The original analysis only looked at **next-token predictions** through the LM head, which shows what tokens each sense predicts. However, this doesn't capture the rich semantic and syntactic structure that Backpack senses should represent.

## Solution: Multi-Dimensional Analysis

We've improved the sense analysis to distinguish senses by analyzing:

1. **Semantic Relatedness** (Embedding-space similarity)
2. **Next Wordpiece Patterns** (Syntactic continuation)
3. **Syntactic Roles** (Verb objects, noun modifiers, proper nouns)
4. **Contextual Activation** (When is each sense activated?)

## New Analysis Features

### 1. Semantic Relatedness Analysis

Instead of just looking at predictions, we analyze **embedding-space similarity**:

```python
# Find words semantically similar to each sense vector
similar_words = get_sense_embedding_similarities(model, tokenizer, sense_vectors)
```

This shows what words are **semantically related** to each sense, not just what it predicts next.

### 2. Syntactic Pattern Analysis

We categorize predictions by syntactic role:

- **Articles**: `the`, `le`, `la`, `un`, `une`
- **Prepositions**: `de`, `du`, `of`, `in`, `on`, `à`, `pour`
- **Proper Nouns**: Capitalized words (likely proper nouns)
- **Verb Objects**: Words that follow verbs
- **Noun Modifiers (nmod)**: Words that modify nouns

### 3. Contextual Activation Patterns

We analyze **when** each sense is activated:

- In verb-object contexts: "support **the proposal**"
- In noun-modifier contexts: "the **important** proposal"
- In proper noun contexts: "**European** Parliament"

### 4. Next Wordpiece Analysis

We analyze what **wordpieces** each sense predicts, which reveals:
- Syntactic continuation patterns
- Language-specific patterns
- Structural elements

## Usage

### Basic Usage (with new features)

```python
from evaluate import analyze_sense_vectors

results = analyze_sense_vectors(
    model, tokenizer, ['hello', 'bonjour'], device,
    top_k=10,
    analyze_relatedness=True,  # Analyze semantic relatedness
    analyze_syntax=True         # Analyze syntactic patterns
)
```

### Advanced Analysis Script

Run the comprehensive distinction analysis:

```bash
python analyze_sense_distinctions.py --out_dir out/backpack_full --device cpu
```

This will show:
- Semantic relatedness for each sense
- Contextual activation patterns
- Syntactic relationships (verb objects, nmod, proper nouns)
- Sense similarity matrix
- Summary of what distinguishes each sense

## What You'll See

### Example Output

```
======================================================================
Word: 'hello'
======================================================================

Top-10 Predictions per Sense (Next Wordpiece):
----------------------------------------------------------------------

Sense  0 (entropy: 8.234, norm: 12.456):
  Next wordpiece predictions:
    bonjour            45.23%
    salut              12.45%
    ...
  Semantically related words (embedding space):
    bonjour            (similarity: 0.892)
    salut              (similarity: 0.856)
    hi                 (similarity: 0.823)
    ...
  Syntactic patterns: Prepositions: de, à, pour | Proper nouns: Bonjour

Sense  1 (entropy: 7.891, norm: 11.234):
  Next wordpiece predictions:
    sep                38.45%
    manque             15.23%
    ...
  Semantically related words (embedding space):
    sep                (similarity: 0.945)
    manque             (similarity: 0.712)
    ...
  Syntactic patterns: Articles: le, la, les
```

## Interpreting Results

### Semantic Relatedness
- **High similarity (>0.8)**: Sense captures semantic meaning similar to those words
- **Medium similarity (0.5-0.8)**: Related concepts
- **Low similarity (<0.5)**: Different semantic space

### Syntactic Patterns
- **Articles/Prepositions**: Sense predicts structural elements
- **Proper Nouns**: Sense associated with named entities
- **Verb Objects**: Sense predicts what follows verbs
- **Noun Modifiers**: Sense predicts modifiers

### Contextual Activation
- Shows **when** each sense is most active
- Reveals **syntactic roles** each sense plays
- Identifies **redundant senses** (similar activation patterns)

## Distinguishing Senses

Based on the analysis, you can now identify:

1. **Semantic Senses**: High semantic relatedness to specific word groups
   - Example: Sense that's related to "parliament", "debate", "discussion"

2. **Syntactic Senses**: Predict specific syntactic elements
   - Example: Sense that predicts articles/prepositions

3. **Proper Noun Senses**: Associated with named entities
   - Example: Sense that predicts capitalized words

4. **Structural Senses**: Predict structural elements
   - Example: Sense that predicts language separators

5. **Redundant Senses**: Similar patterns to other senses
   - Example: Multiple senses predicting the same tokens

## Improving Sense Diversity

If senses are too similar, consider:

1. **Training adjustments**:
   - Add diversity loss to encourage different sense usage
   - Use different initialization strategies
   - Adjust sense predictor architecture

2. **Architecture changes**:
   - Reduce number of senses if many are redundant
   - Add explicit sense specialization objectives
   - Use attention-based sense selection

3. **Data augmentation**:
   - Include more diverse contexts
   - Add explicit syntactic annotations
   - Include more proper nouns and named entities

## Files

- `analyze_sense_distinctions.py`: Comprehensive distinction analysis script
- `evaluate.py`: Updated `analyze_sense_vectors()` function with new features
- `SENSE_DISTINCTIONS_GUIDE.md`: This guide

## Next Steps

1. Run the distinction analysis to see how senses differ
2. Identify redundant senses
3. Adjust training or architecture based on findings
4. Re-analyze after improvements
