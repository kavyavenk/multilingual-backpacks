"""
Complete evaluation suite for multilingual Backpack models
Runs MultiSimLex word similarity, cross-lingual evaluation, and sense analysis
"""

import os
import argparse
import torch
import json
from evaluate import (
    load_model,
    evaluate_multisimlex,
    evaluate_cross_lingual_multisimlex,
    analyze_sense_vectors,
    load_test_data,
    evaluate_translation_bleu,
    evaluate_translation_accuracy
)
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Run full evaluation suite')
    parser.add_argument('--out_dir', type=str, default='out/tiny',
                       help='Directory containing trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--skip_multisimlex', action='store_true',
                       help='Skip MultiSimLex evaluation (if dataset unavailable)')
    parser.add_argument('--skip_translation', action='store_true',
                       help='Skip translation evaluation (BLEU and accuracy)')
    parser.add_argument('--translation_samples', type=int, default=500,
                       help='Number of test samples for translation evaluation')
    parser.add_argument('--data_dir', type=str, default='data/europarl',
                       help='Directory containing Europarl data')
    parser.add_argument('--language_pair', type=str, default='en-fr',
                       help='Language pair for translation evaluation')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MULTILINGUAL BACKPACK EVALUATION SUITE")
    print("="*70)
    print(f"Model directory: {args.out_dir}")
    print(f"Device: {args.device}")
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.out_dir, args.device)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded successfully!")
    print(f"  Type: {type(model).__name__}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Embedding dim: {config.n_embd}")
    if hasattr(config, 'n_senses'):
        print(f"  Number of senses: {config.n_senses}")
    print(f"  Vocab size: {config.vocab_size if hasattr(config, 'vocab_size') else 'unknown'}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {config.tokenizer_name if hasattr(config, 'tokenizer_name') else 'xlm-roberta-base'}...")
    tokenizer_name = config.tokenizer_name if hasattr(config, 'tokenizer_name') else 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    results = {}
    
    # 1. Sense Vector Analysis (always run - doesn't require external datasets)
    print("\n" + "="*70)
    print("1. SENSE VECTOR ANALYSIS")
    print("="*70)
    print("Analyzing what each sense predicts for key multilingual words...\n")
    
    test_words = [
        'hello', 'bonjour',  # Greetings
        'world', 'monde',     # World
        'parliament', 'parlement',  # Parliament (Europarl-specific)
        'support', 'soutenir',  # Support
        'proposal', 'proposition',  # Proposal
    ]
    
    sense_analysis = analyze_sense_vectors(model, tokenizer, test_words, args.device, top_k=5)
    results['sense_analysis'] = sense_analysis
    
    # 2. MultiSimLex Evaluation (skip if not available)
    if not args.skip_multisimlex:
        print("\n" + "="*70)
        print("2. WORD SIMILARITY EVALUATION (MultiSimLex)")
        print("="*70)
        
        # English monolingual
        try:
            print("\n2a. English word similarity...")
            en_corr = evaluate_multisimlex(model, tokenizer, args.device, language='en')
            results['multisimlex_en'] = en_corr
        except Exception as e:
            print(f"  ERROR: {e}")
            results['multisimlex_en'] = None
        
        # French monolingual
        try:
            print("\n2b. French word similarity...")
            fr_corr = evaluate_multisimlex(model, tokenizer, args.device, language='fr')
            results['multisimlex_fr'] = fr_corr
        except Exception as e:
            print(f"  ERROR: {e}")
            results['multisimlex_fr'] = None
        
        # Cross-lingual English-French
        try:
            print("\n2c. Cross-lingual word similarity (EN-FR)...")
            cross_corr = evaluate_cross_lingual_multisimlex(model, tokenizer, args.device, 'en', 'fr')
            results['multisimlex_en_fr'] = cross_corr
        except Exception as e:
            print(f"  ERROR: {e}")
            results['multisimlex_en_fr'] = None
    else:
        print("\n2. WORD SIMILARITY EVALUATION (MultiSimLex) - SKIPPED")
    
    # 3. Translation Evaluation (BLEU and Accuracy)
    if not args.skip_translation:
        print("\n" + "="*70)
        print("3. TRANSLATION EVALUATION (BLEU & ACCURACY)")
        print("="*70)
        
        # Load test data
        print(f"\nLoading test data from {args.data_dir}...")
        test_pairs = load_test_data(
            data_dir=args.data_dir,
            language_pair=args.language_pair,
            max_samples=args.translation_samples,
            split='validation'  # Use validation set as test set
        )
        
        if test_pairs:
            # BLEU Score Evaluation
            try:
                print("\n3a. BLEU Score Evaluation...")
                bleu_results = evaluate_translation_bleu(
                    model, tokenizer, test_pairs, args.device,
                    max_samples=args.translation_samples,
                    max_new_tokens=100,
                    temperature=0.3,  # Lower temperature for better quality
                    top_k=10,  # Top-k sampling for better quality
                    greedy=True  # Use greedy decoding for best quality
                )
                if bleu_results:
                    results['translation_bleu'] = bleu_results
            except Exception as e:
                print(f"  ERROR in BLEU evaluation: {e}")
                results['translation_bleu'] = None
            
            # Translation Accuracy Evaluation
            try:
                print("\n3b. Translation Accuracy Evaluation...")
                accuracy_results = evaluate_translation_accuracy(
                    model, tokenizer, test_pairs, args.device,
                    max_samples=args.translation_samples,
                    max_new_tokens=100,
                    temperature=0.3,  # Lower temperature for better quality
                    top_k=10,  # Top-k sampling for better quality
                    greedy=True  # Use greedy decoding for best quality
                )
                if accuracy_results:
                    results['translation_accuracy'] = accuracy_results
            except Exception as e:
                print(f"  ERROR in accuracy evaluation: {e}")
                results['translation_accuracy'] = None
        else:
            print("  Could not load test data - skipping translation evaluation")
            results['translation_bleu'] = None
            results['translation_accuracy'] = None
    else:
        print("\n3. TRANSLATION EVALUATION (BLEU & ACCURACY) - SKIPPED")
    
    # 4. Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\nModel: {args.out_dir}")
    print(f"Parameters: {n_params:,}")
    
    if results.get('multisimlex_en') is not None:
        print(f"\nWord Similarity (Spearman correlation):")
        print(f"  English:        {results['multisimlex_en']['correlation']:.4f} (p={results['multisimlex_en']['p_value']:.4f})")
        
        if results.get('multisimlex_fr') is not None:
            print(f"  French:         {results['multisimlex_fr']['correlation']:.4f} (p={results['multisimlex_fr']['p_value']:.4f})")
        
        if results.get('multisimlex_en_fr') is not None:
            print(f"  Cross-lingual:  {results['multisimlex_en_fr']['correlation']:.4f} (p={results['multisimlex_en_fr']['p_value']:.4f})")
            
            print(f"\nInterpretation:")
            cross_corr = results['multisimlex_en_fr']['correlation']
            if cross_corr > 0.5:
                print(f"  ✓ EXCELLENT: Strong cross-lingual alignment (>{cross_corr:.2f})")
            elif cross_corr > 0.35:
                print(f"  ○ GOOD: Moderate cross-lingual alignment ({cross_corr:.2f})")
            elif cross_corr > 0.2:
                print(f"  ⚠ WEAK: Poor cross-lingual alignment ({cross_corr:.2f})")
            else:
                print(f"  ✗ FAILED: No meaningful cross-lingual alignment ({cross_corr:.2f})")
    
    print(f"\nSense vectors analyzed for {len(test_words)} words")
    print(f"  See detailed predictions in results file")
    
    # Translation evaluation summary
    if results.get('translation_bleu') is not None:
        print(f"\nTranslation Quality (BLEU):")
        bleu = results['translation_bleu']
        print(f"  Average BLEU: {bleu['avg_bleu']:.4f}")
        if bleu.get('sacrebleu'):
            print(f"  SacreBLEU: {bleu['sacrebleu']['score']:.4f}")
    
    if results.get('translation_accuracy') is not None:
        print(f"\nTranslation Accuracy:")
        acc = results['translation_accuracy']
        print(f"  Exact match rate: {acc['exact_match_rate']:.4f} ({acc['exact_matches']}/{acc['n_pairs']})")
        print(f"  Word-level accuracy: {acc['avg_word_accuracy']:.4f}")
        print(f"  Character-level accuracy: {acc['avg_char_accuracy']:.4f}")
    
    # Save results
    output_file = os.path.join(args.out_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == 'sense_analysis':
                # Skip saving large sense analysis to JSON (too verbose)
                json_results[key] = f"{len(value)} words analyzed"
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    main()
