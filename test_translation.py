"""
Test translation generation to debug BLEU score issues
"""

import torch
from evaluate import load_model, generate_translation
from transformers import AutoTokenizer

def test_translations():
    device = 'cpu'
    
    print("Loading Backpack model...")
    model, config = load_model('out/backpack_full', device)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    # Test cases
    test_cases = [
        ("Hello, how are you?", "Bonjour, comment allez-vous?"),
        ("The parliament is meeting today.", "Le parlement se réunit aujourd'hui."),
        ("I support this proposal.", "Je soutiens cette proposition."),
    ]
    
    print("\n" + "="*70)
    print("TESTING TRANSLATION GENERATION")
    print("="*70)
    
    for source, reference in test_cases:
        print(f"\nSource: {source}")
        print(f"Reference: {reference}")
        
        # Test with different settings
        for name, params in [
            ("Greedy", {"greedy": True, "temperature": 0.0, "top_k": 1}),
            ("Low temp", {"temperature": 0.3, "top_k": 10}),
            ("Medium temp", {"temperature": 0.7, "top_k": 50}),
        ]:
            try:
                generated = generate_translation(
                    model, tokenizer, source, device,
                    max_new_tokens=50,
                    **params
                )
                print(f"  {name}: {generated[:100]}")
            except Exception as e:
                print(f"  {name}: ERROR - {e}")

if __name__ == '__main__':
    test_translations()
