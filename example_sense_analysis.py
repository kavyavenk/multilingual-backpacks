#!/usr/bin/env python3
"""
Example script demonstrating improved sense vector analysis
"""

import argparse
import torch
from evaluate import load_model, analyze_sense_vectors, analyze_cross_lingual_sense_alignment
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Example sense vector analysis')
    parser.add_argument('--out_dir', type=str, default='out/backpack_full',
                       help='Directory containing trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show per sense')
    args = parser.parse_args()
    
    print("="*70)
    print("SENSE VECTOR ANALYSIS EXAMPLE")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from {args.out_dir}...")
    model, config = load_model(args.out_dir, args.device)
    
    # Load tokenizer
    tokenizer_name = config.tokenizer_name if hasattr(config, 'tokenizer_name') else 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print(f"Model type: {type(model).__name__}")
    if hasattr(config, 'n_senses'):
        print(f"Number of senses: {config.n_senses}")
    
    # Example 1: Analyze individual words
    print("\n" + "="*70)
    print("EXAMPLE 1: Individual Word Analysis")
    print("="*70)
    
    words = ['hello', 'bonjour', 'world', 'monde']
    print(f"\nAnalyzing words: {words}")
    results = analyze_sense_vectors(
        model, tokenizer, words, args.device,
        top_k=args.top_k,
        verbose=True
    )
    
    # Example 2: Cross-lingual alignment
    print("\n" + "="*70)
    print("EXAMPLE 2: Cross-Lingual Sense Alignment")
    print("="*70)
    
    word_pairs = [
        ('hello', 'bonjour'),
        ('world', 'monde'),
        ('parliament', 'parlement'),
    ]
    
    print(f"\nAnalyzing translation pairs: {word_pairs}")
    alignment_results = analyze_cross_lingual_sense_alignment(
        model, tokenizer, word_pairs, args.device,
        top_k=args.top_k,
        verbose=True
    )
    
    # Example 3: Access results programmatically
    print("\n" + "="*70)
    print("EXAMPLE 3: Accessing Results Programmatically")
    print("="*70)
    
    if 'hello' in results:
        hello_data = results['hello']
        print(f"\nMetrics for 'hello':")
        metrics = hello_data['metrics']
        print(f"  Mean Entropy: {metrics['mean_entropy']:.3f}")
        print(f"  Avg Sense Similarity: {metrics['avg_sense_similarity']:.3f}")
        print(f"  Number of Senses: {metrics['n_senses']}")
        
        print(f"\nTop predictions for Sense 0:")
        sense_0 = hello_data['predictions'][0]
        for token, prob in zip(sense_0['tokens'][:3], sense_0['probs'][:3]):
            print(f"  {token:20s} {prob*100:6.2f}%")
    
    if ('hello', 'bonjour') in alignment_results:
        alignment_data = alignment_results[('hello', 'bonjour')]
        print(f"\nCross-lingual alignment metrics for 'hello'/'bonjour':")
        metrics = alignment_data['metrics']
        print(f"  Average Alignment Similarity: {metrics['avg_alignment_sim']:.3f}")
        print(f"  Number of Aligned Pairs: {metrics['n_aligned_pairs']}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == '__main__':
    main()
