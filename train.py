"""
Training script for Backpack Language Models
Supports both training from scratch and finetuning
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import math
import pickle
import argparse
import json
import numpy as np
from contextlib import nullcontext
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import BackpackLM, StandardTransformerLM
from configurator import ModelConfig, get_config
from transformers import AutoTokenizer, AutoModelForCausalLM


from google.colab import drive
drive.mount('/content/drive')

g_drive_model_path = "/content/drive/MyDrive/best_model_weights.pt"

# Optional import for BackpackTokenizer (only needed for pretrained models)
try:
    from tokenization_backpack import BackpackTokenizer
except ImportError:
    BackpackTokenizer = None

# Data loading
def get_batch(split, data, block_size, batch_size, device, device_type):
    """Generate a small batch of data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    
    if device_type == 'cuda':
        # Pin memory for faster CPU->GPU transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def load_data(data_dir):
    """Load tokenized data"""
    data_dir = os.path.join('data', data_dir)
    print("Loading train.bin")
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r')
    
    print("Loading val.bin")
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r')
    
    # Load metadata
    print("Loading meta.pkl")
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    return train_data, val_data, meta


@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device, device_type):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        data = train_data if split == 'train' else val_data
        for k in range(eval_iters):
            X, Y = get_batch(split, data, block_size, batch_size, device, device_type)
            print("X bounds: ", X.min().item(), X.max().item())
            print("Y bounds: ", Y.min().item(), Y.max().item())
            len_cap = min(X.size(1), Y.size(1), 512, model.config.block_size)
            X = X[:, :len_cap].clamp(0, model.config.vocab_size - 1)
            Y = Y[:, :len_cap].clamp(0, model.config.vocab_size - 1)
            
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            
        out[split] = losses.mean()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser(description='Train Language Model')
    parser.add_argument('--model_type', type=str, default='backpack', choices=['backpack','transformer'])
    parser.add_argument('--config', type=str, default='train_hansards_scratch', help='Config name')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--data_dir', type=str, default='hansards', help='Data directory')
    parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch', 'resume', 'backpack-small'],
                       help='Initialize from: scratch, resume, or pretrained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float32', 'bfloat16', 'float16'])
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    config.device = args.device
    config.dtype = args.dtype
    config.compile = args.compile
    
    # Load data
    print("Loading data...")
    train_data, val_data, meta = load_data(args.data_dir)
    config.vocab_size = meta['vocab_size']
    
    # Determine device
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Initialize model
    print("Initializing model...")
    
    # Check if this is a transformer baseline config
    is_transformer_baseline = 'transformer_baseline' in args.config or 'transformer' in args.config.lower()
    
    if args.init_from == 'scratch':
        if is_transformer_baseline:
            print("Standard Transformer baseline from scratch")
            model = StandardTransformerLM(config)
            print("Standard Transformer initialized (scratch)")
        else:
            print("Backpack from scratch")
            model = BackpackLM(config)
            print("Backpack initialized (scratch)")
    elif args.init_from == 'resume':
        # Load checkpoint
        print("About to load ckpt")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        print("Loading ckpt")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        if is_transformer_baseline:
            model = StandardTransformerLM(config)
        else:
            model = BackpackLM(config)
        model.load_state_dict(checkpoint['model'])
    elif args.init_from == 'backpack-small':
        # Load pretrained Backpack model (if available)
        print("Loading pretrained Backpack")
        model_name = "stanfordnlp/backpack-gpt2"
        if BackpackTokenizer is not None:
            tokenizer = BackpackTokenizer.from_pretrained(model_name)
        else:
            print("Warning: BackpackTokenizer not available, using AutoTokenizer")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
        print("max pos emb: ", model.config.max_position_embeddings)
        config.block_size = min(config.block_size, model.config.max_position_embeddings)
        print("block size ", config.block_size)

        
    
    model.to(args.device)
    
    # Compile model if requested
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Initialize optimizer (use model's configure_optimizers if available)
    if hasattr(model, 'configure_optimizers'):
        optimizer = model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=device_type
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            weight_decay=config.weight_decay,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
    
    # Training loop
    print("Starting training...")
    iter_num = 0
    best_val_loss = 1e9
    
    # Initialize JSON log file for loss curves
    log_file = os.path.join(args.out_dir, 'training_log.json')
    os.makedirs(args.out_dir, exist_ok=True)
    training_log = {
        'iterations': [],
        'train_loss': [],
        'val_loss': [],
        'top_activating_words': []  # Will store periodically
    }
    
    # Load tokenizer for top activating words analysis (if Backpack model)
    tokenizer = None
    if not is_transformer_baseline:
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name if hasattr(config, 'tokenizer_name') else 'xlm-roberta-base')
        except:
            print("Warning: Could not load tokenizer for top activating words analysis")
    
    while True:
        # Evaluate
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config.eval_iters, train_data, val_data, 
                                  config.block_size, config.batch_size, args.device, device_type)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log to JSON file
            training_log['iterations'].append(iter_num)
            training_log['train_loss'].append(float(losses['train'].item()))
            training_log['val_loss'].append(float(losses['val'].item()))
            
            # Save log file
            with open(log_file, 'w') as f:
                json.dump(training_log, f, indent=2)
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    os.makedirs(args.out_dir, exist_ok=True)
                    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))
                    print("model saved w in runtime")

                    torch.save(checkpoint, g_drive_model_path)
                    print(f"google drive model path: {g_drive_model_path}")

        
        
        # Forward backward update
        X, Y = get_batch('train', train_data, config.block_size, config.batch_size, args.device, device_type)
        X, Y = X.to(args.device), Y.to(args.device)
        # Clamp token IDs to valid vocabulary range
        X = X.clamp(0, config.vocab_size - 1)
        Y = Y.clamp(0, config.vocab_size - 1)
        with ctx:
            logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        # Logging
        if iter_num % config.log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}")
        
        # Track top activating words (for Backpack models only, periodically)
        track_words_interval = max(config.eval_interval // 2, 50)  # At least every 50 iterations
        if not is_transformer_baseline and tokenizer is not None and iter_num % track_words_interval == 0 and iter_num > 0:
            try:
                top_words = get_top_activating_words(model, X, tokenizer, device=args.device, top_k=10)
                training_log['top_activating_words'].append({
                    'iteration': iter_num,
                    'words': top_words
                })
                # Save updated log
                with open(log_file, 'w') as f:
                    json.dump(training_log, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not compute top activating words: {e}")
        
        iter_num += 1
        
        # Termination
        if iter_num > config.max_iters:
            break
    
    print("Training complete!")
    print(f"Training log saved to: {log_file}")


@torch.no_grad()
def get_top_activating_words(model, batch, tokenizer, device='cuda', top_k=10):
    """
    Get top activating words based on sense weights.
    Returns words with highest average sense weight activation.
    """
    if not isinstance(model, BackpackLM):
        return []
    
    model.eval()
    B, T = batch.size()
    
    # Get sense weights for this batch
    sense_embs = model.sense_embeddings(batch)  # (B, T, n_senses * n_embd)
    sense_embs = sense_embs.view(B, T, model.n_senses, model.config.n_embd)
    
    pos = torch.arange(0, T, dtype=torch.long, device=device)
    pos_emb = model.pos_embeddings(pos)  # (T, n_embd)
    context = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, T, n_embd)
    sense_weights = model.sense_predictor(context)  # (B, T, n_senses)
    sense_weights = F.softmax(sense_weights, dim=-1)  # (B, T, n_senses)
    
    # Average sense weights across batch and sequence
    # Shape: (T, n_senses) -> average activation per token position
    avg_weights = sense_weights.mean(dim=0)  # (T, n_senses)
    max_weights_per_token = avg_weights.max(dim=1)[0]  # (T,) - max sense weight for each token
    
    # Get top-k tokens by activation
    top_k_vals, top_k_indices = torch.topk(max_weights_per_token, min(top_k, T))
    
    # Decode tokens to words
    top_words = []
    for idx in top_k_indices.cpu().numpy():
        token_id = batch[0, idx].item()  # Get token from first batch item
        try:
            word = tokenizer.decode([token_id])
            activation = max_weights_per_token[idx].item()
            top_words.append({
                'token_id': int(token_id),
                'word': word,
                'activation': float(activation)
            })
        except:
            continue
    
    model.train()
    return top_words


if __name__ == '__main__':
    import numpy as np
    main()

