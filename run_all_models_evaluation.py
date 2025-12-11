#!/usr/bin/env python3
"""
Run comprehensive evaluations on all three models:
1. Backpack (out/backpack_full)
2. Finetuned Backpack (out/finetuning_best_model_weights.pt)
3. Transformer (out/transformer_full)
"""

import os
import sys
import torch
import json
from evaluate import (
    load_model,
    evaluate_multisimlex,
    evaluate_cross_lingual_multisimlex,
    analyze_sense_vectors,
    load_test_data,
    evaluate_translation_bleu,
    evaluate_translation_accuracy,
    evaluate_sentence_similarity,
    evaluate_perplexity
)
from transformers import AutoTokenizer

def main():
    # Configuration
    BASE_DIR = os.getcwd()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model paths
    models = {
        'backpack': os.path.join(BASE_DIR, 'out/backpack_full'),
        'backpack_finetuned': os.path.join(BASE_DIR, 'out/finetuning_best_model_weights.pt'),
        'transformer': os.path.join(BASE_DIR, 'out/transformer_full'),
    }
    
    DATA_DIR = os.path.join(BASE_DIR, 'data/europarl')
    LANGUAGE_PAIR = 'en-fr'
    TRANSLATION_SAMPLES = 500
    
    print("="*70)
    print("COMPREHENSIVE EVALUATION - ALL MODELS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Translation samples: {TRANSLATION_SAMPLES}")
    print("="*70)
    
    all_results = {}
    
    # Check which models exist
    available_models = {}
    for name, path in models.items():
        if os.path.isfile(path) and path.endswith('.pt'):
            # Direct .pt file
            if os.path.exists(path):
                available_models[name] = path
                print(f"✓ Found {name}: {path}")
            else:
                print(f"⚠️  {name} not found: {path}")
        else:
            # Directory with ckpt.pt
            ckpt_path = os.path.join(path, 'ckpt.pt')
            if os.path.exists(ckpt_path):
                available_models[name] = path
                print(f"✓ Found {name}: {ckpt_path}")
            else:
                print(f"⚠️  {name} not found: {ckpt_path}")
    
    if not available_models:
        print("\n❌ No models found! Please check paths.")
        return
    
    print(f"\n📊 Evaluating {len(available_models)} model(s): {list(available_models.keys())}\n")
    
    # Load test data once
    print("Loading test data...")
    test_pairs = load_test_data(
        data_dir=DATA_DIR,
        language_pair=LANGUAGE_PAIR,
        max_samples=TRANSLATION_SAMPLES,
        split='validation'
    )
    
    if not test_pairs:
        print("❌ No test data loaded!")
        return
    
    print(f"✓ Loaded {len(test_pairs)} test pairs\n")
    
    # Evaluate each model
    for model_name, model_path in available_models.items():
        print("="*70)
        print(f"EVALUATING: {model_name.upper()}")
        print("="*70)
        
        try:
            # Load model
            print(f"\nLoading {model_name}...")
            model, config = load_model(model_path, device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Loaded ({n_params:,} parameters)")
            
            # Load tokenizer
            tokenizer_name = config.tokenizer_name if hasattr(config, 'tokenizer_name') else 'xlm-roberta-base'
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            results = {
                'model_name': model_name,
                'parameters': n_params,
                'model_path': model_path
            }
            
            # 1. Sense Vector Analysis (Backpack models only) - Skip for faster evaluation
            # if hasattr(model, 'get_sense_vectors') or hasattr(config, 'n_senses'):
            #     print(f"\n{'='*70}")
            #     print("1. SENSE VECTOR ANALYSIS")
            #     print(f"{'='*70}")
            #     test_words = ['hello', 'bonjour', 'world', 'monde', 'parliament', 'parlement']
            #     sense_analysis = analyze_sense_vectors(model, tokenizer, test_words, device, top_k=5)
            #     results['sense_analysis'] = sense_analysis
            
            # 2. Translation Evaluation
            print(f"\n{'='*70}")
            print("2. TRANSLATION EVALUATION")
            print(f"{'='*70}")
            
            # BLEU Score
            print(f"\n2a. BLEU Score Evaluation...")
            try:
                bleu_results = evaluate_translation_bleu(
                    model, tokenizer, test_pairs, device,
                    max_samples=TRANSLATION_SAMPLES,
                    max_new_tokens=100,
                    temperature=0.3,
                    top_k=10,
                    greedy=True
                )
                results['translation_bleu'] = bleu_results
                print(f"  ✓ Average BLEU: {bleu_results['avg_bleu']:.4f}")
                print(f"  ✓ Median BLEU: {bleu_results['median_bleu']:.4f}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results['translation_bleu'] = None
            
            # Translation Accuracy
            print(f"\n2b. Translation Accuracy Evaluation...")
            try:
                accuracy_results = evaluate_translation_accuracy(
                    model, tokenizer, test_pairs, device,
                    max_samples=TRANSLATION_SAMPLES,
                    max_new_tokens=100,
                    temperature=0.3,
                    top_k=10,
                    greedy=True
                )
                results['translation_accuracy'] = accuracy_results
                print(f"  ✓ Exact Match: {accuracy_results['exact_match_rate']:.4f}")
                print(f"  ✓ Word Accuracy: {accuracy_results['avg_word_accuracy']:.4f}")
                print(f"  ✓ Char Accuracy: {accuracy_results['avg_char_accuracy']:.4f}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results['translation_accuracy'] = None
            
            # Sentence Similarity
            print(f"\n2c. Sentence Similarity Evaluation...")
            try:
                sent_pairs = test_pairs[:min(100, len(test_pairs))]
                sent_similarities = evaluate_sentence_similarity(
                    model, tokenizer, sent_pairs, device, method='mean'
                )
                if sent_similarities:
                    similarities = [sim for _, _, sim in sent_similarities]
                    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                    results['sentence_similarity'] = {
                        'avg_similarity': avg_sim,
                        'n_pairs': len(sent_similarities),
                        'min_similarity': min(similarities) if similarities else 0.0,
                        'max_similarity': max(similarities) if similarities else 0.0,
                        'std_similarity': np.std(similarities) if similarities else 0.0
                    }
                    print(f"  ✓ Average Similarity: {avg_sim:.4f}")
                    print(f"  ✓ Range: {results['sentence_similarity']['min_similarity']:.4f} - {results['sentence_similarity']['max_similarity']:.4f}")
                else:
                    results['sentence_similarity'] = None
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results['sentence_similarity'] = None
            
            # 2d. Perplexity Evaluation
            print(f"\n2d. Perplexity Evaluation...")
            try:
                # Use full pairs (English <|lang_sep|> French) to match training format
                # CRITICAL: Model was trained on interleaved pairs, not just French text
                perplexity_data = test_pairs[:min(500, len(test_pairs))]
                perplexity_results = evaluate_perplexity(
                    model, tokenizer, perplexity_data, device,
                    max_samples=500,
                    batch_size=8,
                    max_length=512
                )
                results['perplexity'] = perplexity_results
                if perplexity_results:
                    print(f"  ✓ Perplexity: {perplexity_results['perplexity']:.2f}")
                    print(f"  ✓ Interpretation: {perplexity_results['interpretation']}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                results['perplexity'] = None
            
            # 3. Word Similarity (MultiSimLex fallback)
            print(f"\n{'='*70}")
            print("3. WORD SIMILARITY EVALUATION")
            print(f"{'='*70}")
            
            try:
                # English
                print("\n3a. English word similarity...")
                en_corr = evaluate_multisimlex(model, tokenizer, device, language='en', max_samples=50)
                results['multisimlex_en'] = en_corr
                if en_corr:
                    print(f"  ✓ Correlation: {en_corr.get('correlation', 'N/A'):.4f}")
                
                # French
                print("\n3b. French word similarity...")
                fr_corr = evaluate_multisimlex(model, tokenizer, device, language='fr', max_samples=50)
                results['multisimlex_fr'] = fr_corr
                if fr_corr:
                    print(f"  ✓ Correlation: {fr_corr.get('correlation', 'N/A'):.4f}")
                
                # Cross-lingual
                print("\n3c. Cross-lingual word similarity...")
                cross_corr = evaluate_cross_lingual_multisimlex(model, tokenizer, device, max_samples=50)
                results['multisimlex_cross'] = cross_corr
                if cross_corr:
                    print(f"  ✓ Correlation: {cross_corr.get('correlation', 'N/A'):.4f}")
            except Exception as e:
                print(f"  ⚠️  Word similarity evaluation failed: {e}")
                results['multisimlex_en'] = None
                results['multisimlex_fr'] = None
                results['multisimlex_cross'] = None
            
            # Save individual results
            all_results[model_name] = results
            
            # Save to file
            if os.path.isfile(model_path) and model_path.endswith('.pt'):
                # For .pt files, save in out/ directory
                output_file = os.path.join(BASE_DIR, 'out', f'{model_name}_evaluation_results.json')
            else:
                output_file = os.path.join(model_path, 'full_evaluation_results.json')
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")
            
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {'error': str(e)}
    
    # Create summary comparison
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY - ALL MODELS")
    print(f"{'='*70}")
    
    import pandas as pd
    
    summary_data = []
    for model_name, results in all_results.items():
        if 'error' in results:
            continue
        
        row = {'Model': model_name}
        
        # BLEU scores
        if 'translation_bleu' in results and results['translation_bleu']:
            bleu = results['translation_bleu']
            row['Avg BLEU'] = f"{bleu['avg_bleu']:.4f}"
            row['Median BLEU'] = f"{bleu['median_bleu']:.4f}"
        else:
            row['Avg BLEU'] = 'N/A'
            row['Median BLEU'] = 'N/A'
        
        # Accuracy scores
        if 'translation_accuracy' in results and results['translation_accuracy']:
            acc = results['translation_accuracy']
            row['Exact Match'] = f"{acc['exact_match_rate']:.4f}"
            row['Word Acc'] = f"{acc['avg_word_accuracy']:.4f}"
            row['Char Acc'] = f"{acc['avg_char_accuracy']:.4f}"
        else:
            row['Exact Match'] = 'N/A'
            row['Word Acc'] = 'N/A'
            row['Char Acc'] = 'N/A'
        
        # Sentence similarity
        if 'sentence_similarity' in results and results['sentence_similarity']:
            sim = results['sentence_similarity']
            row['Sent Sim'] = f"{sim['avg_similarity']:.4f}"
        else:
            row['Sent Sim'] = 'N/A'
        
        # Perplexity
        if 'perplexity' in results and results['perplexity']:
            ppl = results['perplexity']
            row['Perplexity'] = f"{ppl['perplexity']:.2f}"
        else:
            row['Perplexity'] = 'N/A'
        
        # Word similarity
        if 'multisimlex_cross' in results and results['multisimlex_cross']:
            corr = results['multisimlex_cross'].get('correlation', None)
            row['Word Sim'] = f"{corr:.4f}" if corr is not None else 'N/A'
        else:
            row['Word Sim'] = 'N/A'
        
        # Parameters
        if 'parameters' in results:
            row['Parameters'] = f"{results['parameters']:,}"
        
        summary_data.append(row)
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        print("\n")
        print(df_summary.to_string(index=False))
        
        # Save summary
        summary_file = os.path.join(BASE_DIR, 'out', 'all_models_evaluation_summary.json')
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Full results saved to: {summary_file}")
    else:
        print("\n⚠️  No results to display")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    import numpy as np
    main()
