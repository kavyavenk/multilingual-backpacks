"""
Training script for tiny Backpack model optimized for Colab T4 GPU
Run this in Google Colab or any environment with CUDA GPU
"""

import os
import time
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm

from model import BackpackLM
from configurator import ModelConfig

# ============================================================================
# 1. GPU SETUP
# ============================================================================
print("="*70)
print("GPU SETUP")
print("="*70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
else:
    print("WARNING: Running on CPU (will be slow)")

print()

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================
print("="*70)
print("DATA PREPARATION")
print("="*70)

print("Loading Europarl dataset...")
dataset = load_dataset("europarl_bilingual", "en-fr", split="train")

# Use tiny subset as specified by professor
max_samples = 10000
if len(dataset) > max_samples:
    dataset = dataset.select(range(max_samples))

print(f"Using {len(dataset)} parallel sentences")

# Initialize tokenizer
tokenizer_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
print(f"Tokenizer loaded: {tokenizer_name}")

# Process parallel sentences
print("Processing sentences...")
all_texts = []

for item in tqdm(dataset, desc="Processing"):
    en_text = item['translation']['en']
    fr_text = item['translation']['fr']
    
    # Concatenate with language separator
    combined = en_text + " <|lang_sep|> " + fr_text
    all_texts.append(combined)
    
    # Also add reverse direction
    combined_reverse = fr_text + " <|lang_sep|> " + en_text
    all_texts.append(combined_reverse)

print(f"Total texts: {len(all_texts)}")

# Tokenize
print("Tokenizing...")
all_tokens = []

for text in tqdm(all_texts, desc="Tokenizing"):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
    all_tokens.extend(tokens)

# Convert to numpy
data = np.array(all_tokens, dtype=np.uint32)
print(f"Total tokens: {len(data):,}")

# Train/val split
train_cutoff = int(len(data) * 0.9)
train_data = data[:train_cutoff]
val_data = data[train_cutoff:]

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")
print()

# ============================================================================
# 3. MODEL CONFIGURATION
# ============================================================================
print("="*70)
print("MODEL CONFIGURATION")
print("="*70)

config = ModelConfig(
    # Model architecture - ~500K params
    vocab_size=tokenizer.vocab_size,
    block_size=128,
    n_layer=2,
    n_head=2,
    n_embd=48,
    n_senses=4,
    dropout=0.1,
    bias=False,
    
    # Training
    batch_size=32,  # Larger batch for GPU
    learning_rate=3e-4,
    max_iters=1000,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    
    # Evaluation
    eval_interval=100,
    eval_iters=50,
    log_interval=10,
    
    # System
    device=device,
    dtype='float16' if device == 'cuda' else 'float32',
    compile=False,
    
    # Data
    dataset='europarl',
    tokenizer_name=tokenizer_name,
    languages=['en', 'fr'],
)

print(f"Vocab size: {config.vocab_size:,}")
print(f"Embedding dim: {config.n_embd}")
print(f"Senses: {config.n_senses}")
print(f"Batch size: {config.batch_size}")
print(f"Max iterations: {config.max_iters}")

# Initialize model
model = BackpackLM(config)
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {n_params:,} ({n_params/1e6:.2f}M)")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=(config.beta1, config.beta2),
    weight_decay=config.weight_decay
)

print()

# ============================================================================
# 4. TRAINING UTILITIES
# ============================================================================
def get_batch(data, block_size, batch_size, device):
    """Generate a batch"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, config, device):
    """Estimate loss"""
    model.eval()
    out = {}
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, config.block_size, config.batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    
    model.train()
    return out

# ============================================================================
# 5. TRAINING LOOP
# ============================================================================
print("="*70)
print("TRAINING")
print("="*70)
print()

training_log = {
    'iterations': [],
    'train_loss': [],
    'val_loss': [],
}

model.train()
start_time = time.time()

for iter in range(config.max_iters + 1):
    # Evaluate
    if iter % config.eval_interval == 0:
        losses = estimate_loss(model, config.eval_iters, train_data, val_data, config, device)
        elapsed = time.time() - start_time
        
        print(f"Iter {iter:4d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | Time: {elapsed/60:.1f}min")
        
        training_log['iterations'].append(iter)
        training_log['train_loss'].append(losses['train'])
        training_log['val_loss'].append(losses['val'])
    
    if iter == config.max_iters:
        break
    
    # Training step
    X, Y = get_batch(train_data, config.block_size, config.batch_size, device)
    
    logits, loss = model(X, Y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    optimizer.step()

total_time = time.time() - start_time

print()
print("="*70)
print(f"TRAINING COMPLETE - {total_time/60:.1f} minutes")
print(f"Final train loss: {training_log['train_loss'][-1]:.4f}")
print(f"Final val loss: {training_log['val_loss'][-1]:.4f}")
print("="*70)
print()

# ============================================================================
# 6. SAVE MODEL
# ============================================================================
os.makedirs('out/tiny', exist_ok=True)

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': config,
    'iter': config.max_iters,
}

torch.save(checkpoint, 'out/tiny/ckpt.pt')
print("✓ Model saved to out/tiny/ckpt.pt")

with open('out/tiny/training_log.json', 'w') as f:
    json.dump(training_log, f, indent=2)
print("✓ Training log saved")
print()

# ============================================================================
# 7. PLOT LOSS CURVES
# ============================================================================
plt.figure(figsize=(10, 5))
plt.plot(training_log['iterations'], training_log['train_loss'], label='Train Loss', marker='o')
plt.plot(training_log['iterations'], training_log['val_loss'], label='Val Loss', marker='s')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Progress - Tiny Backpack Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('out/tiny/loss_curves.png', dpi=150, bbox_inches='tight')
print("✓ Loss curves saved to out/tiny/loss_curves.png")
plt.show()

print()
print("="*70)
print("READY FOR EVALUATION")
print("Run: python run_full_evaluation.py --out_dir out/tiny")
print("="*70)
