# Evaluating stanfordnlp/backpack-gpt2 from HuggingFace

## Quick Start

```bash
cd /Users/itskavya/Documents/multilingual-backpacks
python evaluate_huggingface_backpack.py
```

## What This Does

1. **Loads the HuggingFace model**: Downloads and loads `stanfordnlp/backpack-gpt2`
2. **Runs standard evaluations**:
   - Translation BLEU scores (using generation)
   - Translation accuracy (word/char level)
   - Sentence similarity
   - Sense vector analysis (if supported)

## Requirements

```bash
pip install transformers torch
```

## Using in Your Own Code

```python
from evaluate import load_huggingface_model, generate_translation, evaluate_translation_bleu
from transformers import AutoTokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load HuggingFace model
model, config = load_huggingface_model('stanfordnlp/backpack-gpt2', device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('stanfordnlp/backpack-gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate translation
source_text = "Hello, how are you?"
translation = generate_translation(
    model, tokenizer, source_text, device,
    use_sense_retrieval=False,  # Use actual generation
    max_new_tokens=50,
    temperature=0.3,
    top_k=10
)
print(f"Translation: {translation}")
```

## Notes

- **Architecture Differences**: The HuggingFace model may have a different architecture than our custom BackpackLM
- **Sense Vectors**: Sense vector analysis may not work if the model doesn't expose sense vectors in the same way
- **Generation**: The model uses `model.generate()` for autoregressive generation
- **Tokenization**: Uses GPT-2 tokenizer (since backpack-gpt2 is based on GPT-2)

## Output

Results are saved to: `out/huggingface_backpack_evaluation.json`

## Troubleshooting

1. **Import Error**: Make sure transformers is installed: `pip install transformers`
2. **Model Download**: First run will download the model (~500MB)
3. **CUDA Out of Memory**: Use CPU: `device='cpu'`
4. **Sense Analysis Fails**: This is expected if the HuggingFace model doesn't expose sense vectors the same way
