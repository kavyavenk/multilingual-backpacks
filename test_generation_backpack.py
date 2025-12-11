#!/usr/bin/env python3
"""
Test generation evaluation on GPU for full backpack model
"""
import os
import torch
import json
from evaluate import (
    load_model,
    load_test_data,
    generate_translation,
    calculate_bleu_score,
    calculate_sacrebleu
)
from transformers import AutoTokenizer
import numpy as np

def evaluate_generation_bleu(model, tokenizer, test_pairs, device, max_samples=500):
    """Evaluate using generation (not sense retrieval)"""
    print(f"\n{'='*60}")
    print("GENERATION-BASED TRANSLATION BLEU EVALUATION")
    print(f"{'='*60}")
    
    if max_samples and max_samples < len(test_pairs):
        test_pairs = test_pairs[:max_samples]
    
    print(f"Evaluating {len(test_pairs)} sentence pairs using generation...")
    
    references = []
    candidates = []
    bleu_scores = []
    
    print("Generating translations...")
    for i, (source_text, target_text) in enumerate(test_pairs):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_pairs)} pairs...")
        
        try:
            # Use generation (not sense retrieval)
            generated_text = generate_translation(
                model, tokenizer, source_text, device,
                max_new_tokens=100,
                temperature=0.3,
                top_k=10,
                greedy=True,  # Greedy decoding for best quality
                use_sense_retrieval=False  # KEY: Use generation
            )
            
            references.append(target_text)
            candidates.append(generated_text)
            
            bleu_score = calculate_bleu_score(target_text, generated_text)
            if bleu_score is not None:
                bleu_scores.append(bleu_score)
        except Exception as e:
            print(f"  Error translating pair {i+1}: {e}")
            continue
    
    if len(bleu_scores) == 0:
        print("Error: No valid translations generated")
        return None
    
    # Calculate statistics
    avg_bleu = np.mean(bleu_scores)
    median_bleu = np.median(bleu_scores)
    std_bleu = np.std(bleu_scores)
    
    print(f"\nResults:")
    print(f"  Number of pairs evaluated: {len(bleu_scores)}")
    print(f"  Average BLEU score: {avg_bleu:.4f}")
    print(f"  Median BLEU score: {median_bleu:.4f}")
    print(f"  Std deviation: {std_bleu:.4f}")
    print(f"  Min BLEU: {np.min(bleu_scores):.4f}")
    print(f"  Max BLEU: {np.max(bleu_scores):.4f}")
    
    # SacreBLEU
    sacrebleu_result = calculate_sacrebleu(references, candidates)
    if sacrebleu_result:
        print(f"\nSacreBLEU (corpus-level): {sacrebleu_result['score']:.4f}")
    
    return {
        'n_pairs': len(bleu_scores),
        'avg_bleu': float(avg_bleu),
        'median_bleu': float(median_bleu),
        'std_bleu': float(std_bleu),
        'min_bleu': float(np.min(bleu_scores)),
        'max_bleu': float(np.max(bleu_scores)),
        'sacrebleu': sacrebleu_result
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*70)
    print("GENERATION EVALUATION - FULL BACKPACK MODEL")
    print("="*70)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    # Load model
    model_path = 'out/backpack_full'
    print(f"\nLoading model from {model_path}...")
    model, config = load_model(model_path, device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded ({n_params:,} parameters)")
    
    # Load tokenizer
    tokenizer_name = getattr(config, 'tokenizer_name', 'xlm-roberta-base')
    print(f"\nLoading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("✓ Tokenizer loaded")
    
    # Load test data
    print("\nLoading test data...")
    test_pairs = load_test_data(
        data_dir='data/europarl',
        language_pair='en-fr',
        max_samples=500,
        split='validation'
    )
    
    if not test_pairs:
        print("❌ No test data found!")
        return
    
    print(f"✓ Loaded {len(test_pairs)} test pairs")
    
    # Run evaluation
    results = evaluate_generation_bleu(model, tokenizer, test_pairs, device, max_samples=500)
    
    # Save results
    output_file = 'out/backpack_full/generation_evaluation_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'backpack_full',
            'method': 'generation',
            'device': device,
            'parameters': n_params,
            **results
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()