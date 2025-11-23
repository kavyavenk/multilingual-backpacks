"""
Sampling script for Backpack Language Models
Generate text from trained models
"""

import os
import pickle
import argparse
import torch
from model import BackpackLM
from transformers import AutoTokenizer


def load_model(out_dir, device):
    """Load trained model"""
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint['config']
    
    model = BackpackLM(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Sample from Backpack Language Model')
    parser.add_argument('--out_dir', type=str, required=True, help='Model output directory')
    parser.add_argument('--start', type=str, default="Hello", help='Starting text or FILE:path')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--tokenizer_name', type=str, default='xlm-roberta-base', help='Tokenizer name')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load model
    print(f"Loading model from {args.out_dir}...")
    model, config = load_model(args.out_dir, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Get starting text
    if args.start.startswith('FILE:'):
        with open(args.start[5:], 'r') as f:
            start_text = f.read()
    else:
        start_text = args.start
    
    # Encode starting text
    start_ids = tokenizer.encode(start_text, add_special_tokens=True)
    start_ids = torch.tensor([start_ids], dtype=torch.long, device=device)
    
    print(f"\nStarting text: {start_text}")
    print(f"Generating {args.num_samples} sample(s)...\n")
    
    # Generate samples
    for i in range(args.num_samples):
        with torch.no_grad():
            generated_ids = model.generate(
                start_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        print(f"Sample {i+1}:")
        print(generated_text)
        print("-" * 80)


if __name__ == '__main__':
    main()

