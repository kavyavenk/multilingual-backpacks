#!/usr/bin/env python3
"""
Evaluate stanfordnlp/backpack-gpt2 from HuggingFace using our evaluation pipeline.

This script loads the HuggingFace Backpack model and runs our standard evaluations.
"""

import os
import sys
import torch
from evaluate import (
    load_huggingface_model,
    load_test_data,
    evaluate_translation_bleu,
    evaluate_translation_accuracy,
    evaluate_sentence_similarity,
    analyze_sense_vectors,
    generate_translation
)
from transformers import AutoTokenizer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("EVALUATING HUGGINGFACE BACKPACK MODEL")
    print("="*70)
    print(f"Model: stanfordnlp/backpack-gpt2")
    print(f"Device: {device}")
    print("="*70)
    
    # Load HuggingFace model
    print("\nLoading HuggingFace model...")
    try:
        model, config = load_huggingface_model('stanfordnlp/backpack-gpt2', device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure you have transformers installed:")
        print("  pip install transformers")
        return
    
    # Load tokenizer (use GPT-2 tokenizer since backpack-gpt2 is based on GPT-2)
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('stanfordnlp/backpack-gpt2')
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"⚠️  Could not load model-specific tokenizer, using gpt2: {e}")
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Load test data
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'data/europarl')
    
    print("\nLoading test data...")
    test_pairs = load_test_data(
        data_dir=DATA_DIR,
        language_pair='en-fr',
        max_samples=100,  # Start with 100 for testing
        split='validation'
    )
    
    if not test_pairs:
        print("⚠️  No test data found. Skipping translation evaluation.")
        print("   You can still test generation manually.")
        
        # Test basic generation
        print("\n" + "="*70)
        print("TESTING BASIC GENERATION")
        print("="*70)
        test_sentences = [
            "Hello, how are you?",
            "The parliament is meeting today.",
            "I support this proposal."
        ]
        
        for sentence in test_sentences:
            print(f"\nSource: {sentence}")
            try:
                # Note: HuggingFace models may not support our sense retrieval
                # So we'll test generation
                generated = generate_translation(
                    model, tokenizer, sentence, device,
                    use_sense_retrieval=False,  # Use actual generation
                    max_new_tokens=50,
                    temperature=0.7,
                    top_k=50
                )
                print(f"Generated: {generated}")
            except Exception as e:
                print(f"Error: {e}")
        
        return
    
    print(f"✓ Loaded {len(test_pairs)} test pairs")
    
    # Run evaluations
    results = {}
    
    # 1. Translation BLEU (using generation)
    print("\n" + "="*70)
    print("1. TRANSLATION BLEU EVALUATION (Generation)")
    print("="*70)
    try:
        bleu_results = evaluate_translation_bleu(
            model, tokenizer, test_pairs, device,
            max_samples=100,
            max_new_tokens=100,
            temperature=0.3,
            top_k=10,
            greedy=False
        )
        results['translation_bleu'] = bleu_results
    except Exception as e:
        print(f"❌ Error in BLEU evaluation: {e}")
        import traceback
        traceback.print_exc()
        results['translation_bleu'] = None
    
    # 2. Translation Accuracy
    print("\n" + "="*70)
    print("2. TRANSLATION ACCURACY EVALUATION")
    print("="*70)
    try:
        acc_results = evaluate_translation_accuracy(
            model, tokenizer, test_pairs, device,
            max_samples=100,
            max_new_tokens=100,
            temperature=0.3,
            top_k=10,
            greedy=False
        )
        results['translation_accuracy'] = acc_results
    except Exception as e:
        print(f"❌ Error in accuracy evaluation: {e}")
        import traceback
        traceback.print_exc()
        results['translation_accuracy'] = None
    
    # 3. Sentence Similarity
    print("\n" + "="*70)
    print("3. SENTENCE SIMILARITY EVALUATION")
    print("="*70)
    try:
        sent_pairs = test_pairs[:min(50, len(test_pairs))]
        sent_similarities = evaluate_sentence_similarity(
            model, tokenizer, sent_pairs, device, method='mean'
        )
        if sent_similarities:
            similarities = [sim for _, _, sim in sent_similarities]
            results['sentence_similarity'] = {
                'avg_similarity': sum(similarities) / len(similarities),
                'n_pairs': len(sent_similarities),
                'min_similarity': min(similarities),
                'max_similarity': max(similarities)
            }
        else:
            results['sentence_similarity'] = None
    except Exception as e:
        print(f"❌ Error in sentence similarity: {e}")
        import traceback
        traceback.print_exc()
        results['sentence_similarity'] = None
    
    # 4. Sense Vector Analysis (may not work with HuggingFace model)
    print("\n" + "="*70)
    print("4. SENSE VECTOR ANALYSIS")
    print("="*70)
    print("Note: HuggingFace models may have different architecture.")
    print("      Sense analysis may not work if the model doesn't expose sense vectors.")
    try:
        test_words = ['hello', 'world', 'parliament', 'proposal', 'support']
        sense_results = analyze_sense_vectors(
            model, tokenizer, test_words, device, top_k=5, verbose=True
        )
        results['sense_analysis'] = sense_results
    except Exception as e:
        print(f"⚠️  Sense analysis not available: {e}")
        print("   This is expected if the HuggingFace model doesn't expose sense vectors.")
        results['sense_analysis'] = None
    
    # Save results
    import json
    output_file = os.path.join(BASE_DIR, 'out', 'huggingface_backpack_evaluation.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    if results.get('translation_bleu'):
        bleu = results['translation_bleu']
        print(f"\n📊 BLEU Score: {bleu.get('avg_bleu', 'N/A'):.4f}")
    
    if results.get('translation_accuracy'):
        acc = results['translation_accuracy']
        print(f"📊 Word Accuracy: {acc.get('avg_word_accuracy', 0)*100:.2f}%")
        print(f"📊 Char Accuracy: {acc.get('avg_char_accuracy', 0)*100:.2f}%")
    
    if results.get('sentence_similarity'):
        sim = results['sentence_similarity']
        print(f"📊 Sentence Similarity: {sim.get('avg_similarity', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
