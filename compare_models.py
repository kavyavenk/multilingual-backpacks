"""
Compare Backpack and Transformer models side by side
Evaluates both models on the same test data and compares results
"""

import os
import argparse
import torch
import json
from evaluate import (
    load_model,
    analyze_sense_vectors,
    load_test_data,
    evaluate_translation_bleu,
    evaluate_translation_accuracy
)
from transformers import AutoTokenizer


def compare_models(backpack_dir, transformer_dir, device='cpu', 
                   translation_samples=500, data_dir='data/europarl', 
                   language_pair='en-fr', skip_translation=False):
    """
    Compare Backpack and Transformer models on the same evaluation tasks.
    
    Args:
        backpack_dir: Directory containing Backpack model checkpoint
        transformer_dir: Directory containing Transformer model checkpoint
        device: Device to run on
        translation_samples: Number of test samples for translation evaluation
        data_dir: Directory containing Europarl data
        language_pair: Language pair for evaluation
        skip_translation: Skip translation evaluation
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON: BACKPACK vs TRANSFORMER")
    print("="*70)
    
    results = {
        'backpack': {},
        'transformer': {},
        'comparison': {}
    }
    
    # Load both models
    print("\nLoading Backpack model...")
    backpack_model, backpack_config = load_model(backpack_dir, device)
    backpack_params = sum(p.numel() for p in backpack_model.parameters())
    
    print("\nLoading Transformer model...")
    transformer_model, transformer_config = load_model(transformer_dir, device)
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    
    print(f"\n{'='*70}")
    print("MODEL INFORMATION")
    print(f"{'='*70}")
    print(f"Backpack:")
    print(f"  Parameters: {backpack_params:,}")
    print(f"  Embedding dim: {backpack_config.n_embd}")
    print(f"  Senses: {backpack_config.n_senses}")
    print(f"\nTransformer:")
    print(f"  Parameters: {transformer_params:,}")
    print(f"  Embedding dim: {transformer_config.n_embd}")
    
    # Load tokenizer
    tokenizer_name = backpack_config.tokenizer_name if hasattr(backpack_config, 'tokenizer_name') else 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 1. Sense Vector Analysis (Backpack only)
    print(f"\n{'='*70}")
    print("1. SENSE VECTOR ANALYSIS (Backpack only)")
    print(f"{'='*70}")
    test_words = [
        'hello', 'bonjour',
        'world', 'monde',
        'parliament', 'parlement',
        'support', 'soutenir',
        'proposal', 'proposition',
    ]
    
    backpack_senses = analyze_sense_vectors(backpack_model, tokenizer, test_words, device, top_k=5)
    results['backpack']['sense_analysis'] = backpack_senses
    
    # 2. Translation Evaluation (Both models)
    if not skip_translation:
        print(f"\n{'='*70}")
        print("2. TRANSLATION EVALUATION (BLEU & ACCURACY)")
        print(f"{'='*70}")
        
        # Load test data
        print(f"\nLoading test data from {data_dir}...")
        test_pairs = load_test_data(
            data_dir=data_dir,
            language_pair=language_pair,
            max_samples=translation_samples,
            split='validation'
        )
        
        if test_pairs:
            # Evaluate Backpack
            print(f"\n{'='*70}")
            print("2a. BACKPACK MODEL TRANSLATION")
            print(f"{'='*70}")
            try:
                backpack_bleu = evaluate_translation_bleu(
                    backpack_model, tokenizer, test_pairs, device,
                    max_samples=translation_samples,
                    max_new_tokens=100,
                    temperature=1.0,
                    top_k=None
                )
                results['backpack']['translation_bleu'] = backpack_bleu
            except Exception as e:
                print(f"  ERROR: {e}")
                results['backpack']['translation_bleu'] = None
            
            try:
                backpack_acc = evaluate_translation_accuracy(
                    backpack_model, tokenizer, test_pairs, device,
                    max_samples=translation_samples,
                    max_new_tokens=100,
                    temperature=1.0,
                    top_k=None
                )
                results['backpack']['translation_accuracy'] = backpack_acc
            except Exception as e:
                print(f"  ERROR: {e}")
                results['backpack']['translation_accuracy'] = None
            
            # Evaluate Transformer
            print(f"\n{'='*70}")
            print("2b. TRANSFORMER MODEL TRANSLATION")
            print(f"{'='*70}")
            try:
                transformer_bleu = evaluate_translation_bleu(
                    transformer_model, tokenizer, test_pairs, device,
                    max_samples=translation_samples,
                    max_new_tokens=100,
                    temperature=1.0,
                    top_k=None
                )
                results['transformer']['translation_bleu'] = transformer_bleu
            except Exception as e:
                print(f"  ERROR: {e}")
                results['transformer']['translation_bleu'] = None
            
            try:
                transformer_acc = evaluate_translation_accuracy(
                    transformer_model, tokenizer, test_pairs, device,
                    max_samples=translation_samples,
                    max_new_tokens=100,
                    temperature=1.0,
                    top_k=None
                )
                results['transformer']['translation_accuracy'] = transformer_acc
            except Exception as e:
                print(f"  ERROR: {e}")
                results['transformer']['translation_accuracy'] = None
    
    # 3. Comparison Summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nModel Parameters:")
    print(f"  Backpack:    {backpack_params:,}")
    print(f"  Transformer: {transformer_params:,}")
    print(f"  Difference:  {backpack_params - transformer_params:,} ({((backpack_params - transformer_params) / transformer_params * 100):.1f}% more)")
    
    if results['backpack'].get('translation_bleu') and results['transformer'].get('translation_bleu'):
        print(f"\nTranslation Quality (BLEU):")
        bp_bleu = results['backpack']['translation_bleu']['avg_bleu']
        tf_bleu = results['transformer']['translation_bleu']['avg_bleu']
        print(f"  Backpack:    {bp_bleu:.4f}")
        print(f"  Transformer: {tf_bleu:.4f}")
        print(f"  Difference:  {bp_bleu - tf_bleu:.4f} ({((bp_bleu - tf_bleu) / max(tf_bleu, 0.0001) * 100):.1f}%)")
        results['comparison']['bleu_diff'] = bp_bleu - tf_bleu
    
    if results['backpack'].get('translation_accuracy') and results['transformer'].get('translation_accuracy'):
        print(f"\nTranslation Accuracy:")
        bp_acc = results['backpack']['translation_accuracy']
        tf_acc = results['transformer']['translation_accuracy']
        
        print(f"  Exact Match Rate:")
        print(f"    Backpack:    {bp_acc['exact_match_rate']:.4f}")
        print(f"    Transformer: {tf_acc['exact_match_rate']:.4f}")
        
        print(f"  Word-level Accuracy:")
        print(f"    Backpack:    {bp_acc['avg_word_accuracy']:.4f}")
        print(f"    Transformer: {tf_acc['avg_word_accuracy']:.4f}")
        
        print(f"  Character-level Accuracy:")
        print(f"    Backpack:    {bp_acc['avg_char_accuracy']:.4f}")
        print(f"    Transformer: {tf_acc['avg_char_accuracy']:.4f}")
        
        results['comparison']['exact_match_diff'] = bp_acc['exact_match_rate'] - tf_acc['exact_match_rate']
        results['comparison']['word_acc_diff'] = bp_acc['avg_word_accuracy'] - tf_acc['avg_word_accuracy']
        results['comparison']['char_acc_diff'] = bp_acc['avg_char_accuracy'] - tf_acc['avg_char_accuracy']
    
    # Save comparison results
    output_file = 'out/model_comparison.json'
    os.makedirs('out', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Comparison results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare Backpack and Transformer models')
    parser.add_argument('--backpack_dir', type=str, default='out/backpack_full',
                       help='Directory containing Backpack model checkpoint')
    parser.add_argument('--transformer_dir', type=str, default='out/transformer_full',
                       help='Directory containing Transformer model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--translation_samples', type=int, default=500,
                       help='Number of test samples for translation evaluation')
    parser.add_argument('--data_dir', type=str, default='data/europarl',
                       help='Directory containing Europarl data')
    parser.add_argument('--language_pair', type=str, default='en-fr',
                       help='Language pair for translation evaluation')
    parser.add_argument('--skip_translation', action='store_true',
                       help='Skip translation evaluation')
    
    args = parser.parse_args()
    
    compare_models(
        backpack_dir=args.backpack_dir,
        transformer_dir=args.transformer_dir,
        device=args.device,
        translation_samples=args.translation_samples,
        data_dir=args.data_dir,
        language_pair=args.language_pair,
        skip_translation=args.skip_translation
    )


if __name__ == '__main__':
    main()
