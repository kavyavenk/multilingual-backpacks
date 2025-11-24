# Quick Reference: Model Parameters

## Scratch Model (`train_europarl_scratch`)

### Key Specifications

**Total Parameters**: **1,642,985,488** (1.643B)

**Architecture**:
- 6 transformer layers
- 6 attention heads per layer
- 384 embedding dimension
- 16 sense vectors per token
- 512 context length
- 250,002 vocabulary size (XLM-RoBERTa-base)

**Largest Components**:
1. Sense embeddings: **1.54B** (93.5%)
2. LM head: **96M** (5.8%)
3. Transformer blocks: **10.6M** (0.65%)

**Training**:
- Batch size: 32
- Learning rate: 3e-4
- Max iterations: 10,000
- Weight decay: 0.1
- Gradient clip: 1.0

**Model Size**: ~3.13 GB (float16)

**Comparison**: 8.1x larger than equivalent standard transformer

---

## Finetune Model (`train_europarl_finetune`)

### Key Specifications

**Total Parameters**: **3,350,369,296** (3.350B)

**Architecture**:
- 12 transformer layers
- 12 attention heads per layer
- 768 embedding dimension
- 16 sense vectors per token
- 1024 context length
- 250,002 vocabulary size (XLM-RoBERTa-base)

**Largest Components**:
1. Sense embeddings: **3.07B** (91.7%)
2. LM head: **192M** (5.7%)
3. Transformer blocks: **85M** (2.5%)

**Training**:
- Batch size: 16
- Learning rate: 1e-5 (lower for finetuning)
- Max iterations: 5,000
- Weight decay: 0.1
- Gradient clip: 1.0

**Model Size**: ~6.39 GB (float16)

**Comparison**: 7.1x larger than equivalent standard transformer

---

## Parameter Calculation

Run the verification script:

```bash
# Scratch model
python experiments/calculate_model_params.py train_europarl_scratch

# Finetune model
python experiments/calculate_model_params.py train_europarl_finetune
```

---

## Key Differences

| Aspect | Scratch | Finetune |
|--------|---------|----------|
| Layers | 6 | 12 |
| Heads | 6 | 12 |
| Embedding Dim | 384 | 768 |
| Context Length | 512 | 1024 |
| Parameters | 1.64B | 3.35B |
| Batch Size | 32 | 16 |
| Learning Rate | 3e-4 | 1e-5 |
| Max Iters | 10,000 | 5,000 |

