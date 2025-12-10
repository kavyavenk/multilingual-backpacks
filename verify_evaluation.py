#!/usr/bin/env python3
"""
Comprehensive verification script for evaluation system.
Run this before reporting results in your paper.
"""

import argparse
import json
import sys
from pathlib import Path

def check_sense_labels():
    """Verify sense labels are defined and correct."""
    print("\n" + "="*70)
    print("1. CHECKING SENSE LABELS")
    print("="*70)
    
    try:
        from evaluate import SENSE_LABELS
        
        if len(SENSE_LABELS) != 16:
            print(f"❌ ERROR: Expected 16 sense labels, got {len(SENSE_LABELS)}")
            return False
        
        print(f"✅ Found {len(SENSE_LABELS)} sense labels")
        for i in range(16):
            if i not in SENSE_LABELS:
                print(f"❌ ERROR: Missing label for sense {i}")
                return False
            print(f"   Sense {i:2d}: {SENSE_LABELS[i]}")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_translation_generation(out_dir):
    """Verify translation generation works."""
    print("\n" + "="*70)
    print("2. CHECKING TRANSLATION GENERATION")
    print("="*70)
    
    try:
        from evaluate import generate_translation, load_model
        from transformers import AutoTokenizer
        
        model, _ = load_model(out_dir, 'cpu')
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        
        test_pairs = [
            ('hello', 'bonjour'),
            ('world', 'monde'),
            ('parliament', 'parlement'),
        ]
        
        all_passed = True
        for en, expected_fr in test_pairs:
            result = generate_translation(model, tokenizer, en, 'cpu', use_sense_retrieval=True)
            result_lower = result.lower().strip()
            expected_lower = expected_fr.lower().strip()
            
            if result_lower == expected_lower:
                print(f"✅ {en:15s} → {result:15s} (correct)")
            else:
                print(f"⚠️  {en:15s} → {result:15s} (expected: {expected_fr})")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_evaluation_results(out_dir):
    """Verify evaluation results file exists and is complete."""
    print("\n" + "="*70)
    print("3. CHECKING EVALUATION RESULTS")
    print("="*70)
    
    results_file = Path(out_dir) / 'evaluation_results.json'
    
    if not results_file.exists():
        print(f"❌ ERROR: Results file not found: {results_file}")
        return False
    
    try:
        with open(results_file) as f:
            results = json.load(f)
        
        expected_keys = ['translation_bleu', 'translation_accuracy']
        missing_keys = [k for k in expected_keys if k not in results]
        
        if missing_keys:
            print(f"❌ ERROR: Missing keys in results: {missing_keys}")
            return False
        
        print("✅ Evaluation results file exists and contains expected keys")
        
        # Print key metrics
        if 'translation_accuracy' in results:
            acc = results['translation_accuracy']
            print(f"   Word-level accuracy: {acc.get('avg_word_accuracy', 0):.2%}")
            print(f"   Character-level accuracy: {acc.get('avg_char_accuracy', 0):.2%}")
        
        if 'translation_bleu' in results:
            bleu = results['translation_bleu']
            print(f"   Average BLEU: {bleu.get('avg_bleu', 0):.4f}")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_model_comparison():
    """Verify model comparison file exists."""
    print("\n" + "="*70)
    print("4. CHECKING MODEL COMPARISON")
    print("="*70)
    
    comparison_file = Path('out/model_comparison.json')
    
    if not comparison_file.exists():
        print(f"⚠️  WARNING: Comparison file not found: {comparison_file}")
        print("   Run compare_models.py to generate this file")
        return False
    
    try:
        with open(comparison_file) as f:
            comparison = json.load(f)
        
        if 'backpack' not in comparison or 'transformer' not in comparison:
            print("❌ ERROR: Comparison file missing backpack or transformer results")
            return False
        
        print("✅ Model comparison file exists")
        
        # Print comparison
        bp_bleu = comparison['backpack']['translation_bleu']['avg_bleu']
        tf_bleu = comparison['transformer']['translation_bleu']['avg_bleu']
        diff = comparison.get('bleu_diff', bp_bleu - tf_bleu)
        
        print(f"   Backpack BLEU: {bp_bleu:.4f}")
        print(f"   Transformer BLEU: {tf_bleu:.4f}")
        print(f"   Advantage: {diff:.4f} ({diff/tf_bleu*100:.1f}% better)")
        
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def check_language_filtering():
    """Verify language filtering works."""
    print("\n" + "="*70)
    print("5. CHECKING LANGUAGE FILTERING")
    print("="*70)
    
    try:
        from evaluate import _is_english_or_french
        
        # Test known non-English/French words
        non_en_fr_words = ['Prishtinë', 'scherm', 'gol', 'pomembno', 'Hoy', 'fejl']
        for word in non_en_fr_words:
            if _is_english_or_french(word):
                print(f"❌ ERROR: {word} should be filtered but isn't")
                return False
            print(f"✅ {word:15s} correctly filtered")
        
        # Test English/French words
        en_fr_words = ['hello', 'bonjour', 'world', 'monde', 'parliament', 'parlement']
        for word in en_fr_words:
            if not _is_english_or_french(word):
                print(f"❌ ERROR: {word} should pass but doesn't")
                return False
        
        print("✅ Language filtering working correctly")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify evaluation system')
    parser.add_argument('--out_dir', type=str, default='out/backpack_full',
                       help='Model output directory')
    args = parser.parse_args()
    
    print("="*70)
    print("EVALUATION SYSTEM VERIFICATION")
    print("="*70)
    
    checks = [
        ("Sense Labels", check_sense_labels),
        ("Translation Generation", lambda: check_translation_generation(args.out_dir)),
        ("Evaluation Results", lambda: check_evaluation_results(args.out_dir)),
        ("Model Comparison", check_model_comparison),
        ("Language Filtering", check_language_filtering),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("="*70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - READY FOR PAPER REPORTING")
    else:
        print("❌ SOME CHECKS FAILED - REVIEW ERRORS ABOVE")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
