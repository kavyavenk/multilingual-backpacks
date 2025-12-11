#!/usr/bin/env python3
"""
Compare translation generation (model.generate()) vs sense retrieval
on all three models: Backpack, Finetuned Backpack, and Transformer

Since training loss values were excellent, we want to test if actual
autoregressive generation works better than sense retrieval.
"""

import os
import torch
import json
from evaluate import (
    load_model,
    load_test_data,
    generate_translation,
    calculate_bleu_score
)
from transformers import AutoTokenizer

def compare_methods(model, tokenizer, test_pairs, device, model_name, max_samples=50):
    """
    Compare generation vs sense retrieval for a model.
    
    Returns:
        dict with results for both methods
    """
    print(f"\n{'='*70}")
    print(f"COMPARING METHODS FOR: {model_name.upper()}")
    print(f"{'='*70}")
    
    if max_samples and max_samples < len(test_pairs):
        test_pairs = test_pairs[:max_samples]
    
    results = {
        'model_name': model_name,
        'n_samples': len(test_pairs),
        'generation': {'bleu_scores': [], 'examples': []},
        'sense_retrieval': {'bleu_scores': [], 'examples': []}
    }
    
    print(f"\nTesting {len(test_pairs)} sentence pairs...")
    
    for i, (source_text, target_text) in enumerate(test_pairs):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_pairs)} pairs...")
        
        # Method 1: Generation (autoregressive)
        try:
            gen_translation = generate_translation(
                model, tokenizer, source_text, device,
                max_new_tokens=100,
                temperature=0.3,  # Lower temperature for more deterministic
                top_k=10,
                greedy=False,
                use_sense_retrieval=False  # Use actual generation
            )
            gen_bleu = calculate_bleu_score(target_text, gen_translation)
            if gen_bleu is not None:
                results['generation']['bleu_scores'].append(gen_bleu)
                if len(results['generation']['examples']) < 5:
                    results['generation']['examples'].append({
                        'source': source_text,
                        'target': target_text,
                        'generated': gen_translation,
                        'bleu': gen_bleu
                    })
        except Exception as e:
            print(f"    Generation error for pair {i+1}: {e}")
            continue
        
        # Method 2: Sense Retrieval
        try:
            retrieval_translation = generate_translation(
                model, tokenizer, source_text, device,
                use_sense_retrieval=True  # Use sense retrieval
            )
            retrieval_bleu = calculate_bleu_score(target_text, retrieval_translation)
            if retrieval_bleu is not None:
                results['sense_retrieval']['bleu_scores'].append(retrieval_bleu)
                if len(results['sense_retrieval']['examples']) < 5:
                    results['sense_retrieval']['examples'].append({
                        'source': source_text,
                        'target': target_text,
                        'generated': retrieval_translation,
                        'bleu': retrieval_bleu
                    })
        except Exception as e:
            print(f"    Retrieval error for pair {i+1}: {e}")
            continue
    
    # Calculate statistics
    if results['generation']['bleu_scores']:
        gen_scores = results['generation']['bleu_scores']
        results['generation']['avg_bleu'] = sum(gen_scores) / len(gen_scores)
        results['generation']['median_bleu'] = sorted(gen_scores)[len(gen_scores) // 2]
        results['generation']['max_bleu'] = max(gen_scores)
        results['generation']['min_bleu'] = min(gen_scores)
    
    if results['sense_retrieval']['bleu_scores']:
        ret_scores = results['sense_retrieval']['bleu_scores']
        results['sense_retrieval']['avg_bleu'] = sum(ret_scores) / len(ret_scores)
        results['sense_retrieval']['median_bleu'] = sorted(ret_scores)[len(ret_scores) // 2]
        results['sense_retrieval']['max_bleu'] = max(ret_scores)
        results['sense_retrieval']['min_bleu'] = min(ret_scores)
    
    return results


def main():
    BASE_DIR = os.getcwd()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("GENERATION vs SENSE RETRIEVAL COMPARISON")
    print("="*70)
    print(f"Device: {device}")
    print("="*70)
    
    # Model paths
    models = {
        'backpack': os.path.join(BASE_DIR, 'out/backpack_full'),
        'backpack_finetuned': os.path.join(BASE_DIR, 'out/finetuning_best_model_weights.pt'),
        'transformer': os.path.join(BASE_DIR, 'out/transformer_full'),
    }
    
    # Load test data
    DATA_DIR = os.path.join(BASE_DIR, 'data/europarl')
    print("\nLoading test data...")
    test_pairs = load_test_data(
        data_dir=DATA_DIR,
        language_pair='en-fr',
        max_samples=50,  # Start with 50 for quick comparison
        split='validation'
    )
    
    if not test_pairs:
        print("❌ No test data loaded!")
        return
    
    print(f"✓ Loaded {len(test_pairs)} test pairs\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    all_results = {}
    
    # Test each model
    for model_name, model_path in models.items():
        # Check if model exists
        if os.path.isfile(model_path) and model_path.endswith('.pt'):
            ckpt_path = model_path
        else:
            ckpt_path = os.path.join(model_path, 'ckpt.pt')
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️  Skipping {model_name}: {ckpt_path} not found")
            continue
        
        try:
            print(f"\n{'='*70}")
            print(f"Loading {model_name.upper()}...")
            print(f"{'='*70}")
            
            model, config = load_model(model_path, device)
            model.eval()
            
            # Compare methods
            results = compare_methods(model, tokenizer, test_pairs, device, model_name, max_samples=50)
            all_results[model_name] = results
            
            # Print summary
            print(f"\n{'─'*70}")
            print(f"RESULTS FOR {model_name.upper()}:")
            print(f"{'─'*70}")
            
            if 'generation' in results and 'avg_bleu' in results['generation']:
                gen = results['generation']
                print(f"\n📊 Generation (model.generate()):")
                print(f"   Average BLEU: {gen['avg_bleu']:.4f}")
                print(f"   Median BLEU:  {gen['median_bleu']:.4f}")
                print(f"   Range:        {gen['min_bleu']:.4f} - {gen['max_bleu']:.4f}")
                print(f"   Samples:      {len(gen['bleu_scores'])}")
            
            if 'sense_retrieval' in results and 'avg_bleu' in results['sense_retrieval']:
                ret = results['sense_retrieval']
                print(f"\n📊 Sense Retrieval:")
                print(f"   Average BLEU: {ret['avg_bleu']:.4f}")
                print(f"   Median BLEU:  {ret['median_bleu']:.4f}")
                print(f"   Range:        {ret['min_bleu']:.4f} - {ret['max_bleu']:.4f}")
                print(f"   Samples:      {len(ret['bleu_scores'])}")
            
            # Compare
            if ('generation' in results and 'avg_bleu' in results['generation'] and
                'sense_retrieval' in results and 'avg_bleu' in results['sense_retrieval']):
                gen_avg = results['generation']['avg_bleu']
                ret_avg = results['sense_retrieval']['avg_bleu']
                diff = gen_avg - ret_avg
                print(f"\n📈 Comparison:")
                print(f"   Generation vs Retrieval: {diff:+.4f}")
                if diff > 0:
                    print(f"   ✅ Generation is BETTER by {diff:.4f}")
                elif diff < 0:
                    print(f"   ✅ Sense Retrieval is BETTER by {abs(diff):.4f}")
                else:
                    print(f"   ⚖️  Both methods are EQUAL")
            
            # Show examples
            if results['generation']['examples']:
                print(f"\n📝 Generation Examples:")
                for ex in results['generation']['examples'][:3]:
                    print(f"\n   Source:  {ex['source'][:80]}...")
                    print(f"   Target:  {ex['target'][:80]}...")
                    print(f"   Generated: {ex['generated'][:80]}...")
                    print(f"   BLEU: {ex['bleu']:.4f}")
            
            if results['sense_retrieval']['examples']:
                print(f"\n📝 Sense Retrieval Examples:")
                for ex in results['sense_retrieval']['examples'][:3]:
                    print(f"\n   Source:  {ex['source'][:80]}...")
                    print(f"   Target:  {ex['target'][:80]}...")
                    print(f"   Generated: {ex['generated'][:80]}...")
                    print(f"   BLEU: {ex['bleu']:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_file = os.path.join(BASE_DIR, 'out', 'generation_vs_retrieval_comparison.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_file}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for model_name, results in all_results.items():
        if 'generation' in results and 'avg_bleu' in results['generation']:
            gen_avg = results['generation']['avg_bleu']
            ret_avg = results['sense_retrieval'].get('avg_bleu', 0)
            print(f"\n{model_name.upper()}:")
            print(f"  Generation:      {gen_avg:.4f}")
            print(f"  Sense Retrieval: {ret_avg:.4f}")
            print(f"  Difference:      {gen_avg - ret_avg:+.4f}")


if __name__ == '__main__':
    main()
