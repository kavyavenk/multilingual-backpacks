"""
Sense Vector Analysis Experiment
Analyze what each sense vector represents/predicts
Based on nanoBackpackLM experiments
"""

import os
import argparse
import torch
import torch.nn.functional as F
from model import BackpackLM
from transformers import AutoTokenizer


class SenseVectorExperiment:
    """Experiment class for analyzing sense vectors"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def sense_projection(self, word, top_k=5):
        """
        Project each sense vector through the LM head to see what it predicts.
        
        Returns:
            list: List of top-k predictions for each sense
        """
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 0:
            return []
        
        token_id = torch.tensor([tokens[0]], device=self.device).unsqueeze(0)
        sense_vectors = self.model.get_sense_vectors(token_id)  # (1, 1, n_senses, n_embd)
        sense_vectors = sense_vectors.squeeze(0).squeeze(0)  # (n_senses, n_embd)
        
        sense_predictions = []
        with torch.no_grad():
            for sense_idx in range(sense_vectors.shape[0]):
                sense_vec = sense_vectors[sense_idx].unsqueeze(0)  # (1, n_embd)
                logits = self.model.lm_head(sense_vec)  # (1, vocab_size)
                
                # Get top-k predictions
                top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
                top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices[0]]
                sense_predictions.append(top_tokens)
        
        return sense_predictions
    
    def analyze_multilingual_senses(self, word_pairs):
        """
        Analyze sense vectors for translation pairs to see if senses align.
        
        Args:
            word_pairs: List of (word_en, word_fr) tuples
        """
        results = {}
        
        for word_en, word_fr in word_pairs:
            en_senses = self.sense_projection(word_en)
            fr_senses = self.sense_projection(word_fr)
            
            results[(word_en, word_fr)] = {
                'en': en_senses,
                'fr': fr_senses,
            }
        
        return results


def load_model(out_dir, device):
    """Load trained model"""
    import pickle
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
    parser = argparse.ArgumentParser(description='Analyze sense vectors')
    parser.add_argument('--out_dir', type=str, required=True, help='Model output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--tokenizer_name', type=str, default='xlm-roberta-base', help='Tokenizer name')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load model
    print(f"Loading model from {args.out_dir}...")
    model, config = load_model(args.out_dir, device)
    
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Initialize experiment
    ex = SenseVectorExperiment(model, tokenizer, device)
    
    # Test words
    print("\n=== English Word Sense Analysis ===")
    english_words = ['hello', 'world', 'language', 'model', 'learning']
    for word in english_words:
        print(f"\n# {word}")
        predictions = ex.sense_projection(word)
        for sense_idx, preds in enumerate(predictions):
            print(f"Sense {sense_idx}: {preds}")
    
    print("\n=== French Word Sense Analysis ===")
    french_words = ['bonjour', 'monde', 'langue', 'modèle', 'apprentissage']
    for word in french_words:
        print(f"\n# {word}")
        predictions = ex.sense_projection(word)
        for sense_idx, preds in enumerate(predictions):
            print(f"Sense {sense_idx}: {preds}")
    
    # Multilingual analysis
    print("\n=== Multilingual Sense Comparison ===")
    translation_pairs = [
        ('hello', 'bonjour'),
        ('world', 'monde'),
        ('language', 'langue'),
        ('model', 'modèle'),
    ]
    
    multilingual_results = ex.analyze_multilingual_senses(translation_pairs)
    for (word_en, word_fr), results in multilingual_results.items():
        print(f"\n{word_en} <-> {word_fr}:")
        print(f"  English senses:")
        for sense_idx, preds in enumerate(results['en']):
            print(f"    Sense {sense_idx}: {preds}")
        print(f"  French senses:")
        for sense_idx, preds in enumerate(results['fr']):
            print(f"    Sense {sense_idx}: {preds}")


if __name__ == '__main__':
    main()

