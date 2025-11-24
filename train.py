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
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import BackpackLM
from configurator import ModelConfig, get_config
from transformers import AutoTokenizer, AutoModelForCausalLM

# Data loading
def get_batch(split, data, block_size, batch_size, device, device_type, vocab_size):
    """Generate a small batch of data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    x = x.clamp(0, vocab_size - 1).contiguous()
    y = y.clamp(0, vocab_size - 1).contiguous()

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
def estimate_loss(model, eval_iters, train_data, val_data, block_size, batch_size, device, device_type, vocab_size):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        data = train_data if split == 'train' else val_data
        for k in range(eval_iters):
            X, Y = get_batch(split, data, block_size, batch_size, device, device_type, model.config.vocab_size)
            print("X bounds: ", X.min().item(), X.max().item())
            print("Y bounds: ", Y.min().item(), Y.max().item())
            len_cap = min(X.size(1), Y.size(1), 512, model.config.n_positions)
            X = X[:, :len_cap].clamp(0, model.config.vocab_size - 1)
            Y = Y[:, :len_cap].clamp(0, model.config.vocab_size - 1)
            
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            
        out[split] = losses.mean()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser(description='Train Backpack Language Model')
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
    if args.init_from == 'scratch':
        print("Backpack from scratch")
        model = BackpackLM(config)
        print("Backpack initialized (scratch)")
    elif args.init_from == 'resume':
        # Load checkpoint
        print("About to load ckpt")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        print("Loading ckpt")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        model = BackpackLM(config)
        model.load_state_dict(checkpoint['model'])
    elif args.init_from == 'backpack-small':
        # Load pretrained Backpack model (if available)
        print("Loading pretrained Backpack")
        model_name = "stanfordnlp/backpack-gpt2"
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
    
    # Initialize optimizer
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
    
    while True:
        # Evaluate
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config.eval_iters, train_data, val_data, 
                                  config.block_size, config.batch_size, args.device, device_type, model.config.vocab_size)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
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
        
        # Forward backward update
        X, Y = get_batch('train', train_data, config.block_size, config.batch_size, args.device, device_type, config.vocab_size)
        X, Y = X.to(args.device), Y.to(args.device)
        with ctx:
            logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        # Logging
        if iter_num % config.log_interval == 0:
            print(f"iter {iter_num}: loss {loss.item():.4f}")
        
        iter_num += 1
        
        # Termination
        if iter_num > config.max_iters:
            break
    
    print("Training complete!")


if __name__ == '__main__':
    import numpy as np
    main()

