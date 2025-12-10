"""
Evaluation script for Backpack Language Models
Evaluates word-level and sentence-level representations
"""

import os
import pickle
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from model import BackpackLM, StandardTransformerLM
from configurator import ModelConfig


def load_model(out_dir, device):
    """Load trained model (Backpack or StandardTransformer)"""
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        error_msg = f"\n{'='*60}\n"
        error_msg += f"ERROR: Checkpoint not found: {ckpt_path}\n"
        error_msg += f"{'='*60}\n"
        error_msg += "\nYou need to train a model first before evaluating.\n\n"
        error_msg += "To train a model, run:\n"
        error_msg += f"  python train.py --config train_europarl_tiny --out_dir {out_dir} --data_dir europarl\n\n"
        error_msg += "After training completes, you can then run evaluation:\n"
        error_msg += f"  python evaluate.py --out_dir {out_dir} --multisimlex --languages en fr\n"
        error_msg += f"{'='*60}\n"
        raise FileNotFoundError(error_msg)
    
    with torch.serialization.safe_globals([ModelConfig]):
        checkpoint = torch.load(ckpt_path, map_location=device)

    config = checkpoint['config']
    
    # Determine model type based on config or checkpoint
    is_transformer = hasattr(config, 'n_senses') and config.n_senses == 1
    if not is_transformer:
        # Check if we're working with a transformer baseline by looking at model class name
        # or by checking if sense_embeddings exists in state dict
        state_dict_keys = checkpoint['model'].keys()
        is_transformer = 'sense_embeddings' not in state_dict_keys
    
    if is_transformer:
        model = StandardTransformerLM(config)
    else:
        model = BackpackLM(config)
    
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, config


def get_word_representations(model, tokenizer, words, device):
    """
    Extract word representations (sense vectors for Backpack, token embeddings for Transformer).
    
    Returns:
        dict: {word: vectors} where vectors is (n_senses, n_embd) for Backpack or (1, n_embd) for Transformer
    """
    representations = {}
    is_backpack = isinstance(model, BackpackLM)
    
    for word in words:
        # Tokenize word
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 0:
            continue
        
        token_id = torch.tensor([tokens[0]], device=device).unsqueeze(0)
        
        if is_backpack:
            # Get sense vectors for Backpack model
            sense_vectors = model.get_sense_vectors(token_id)  # (1, 1, n_senses, n_embd)
            sense_vectors = sense_vectors.squeeze(0).squeeze(0)  # (n_senses, n_embd)
            representations[word] = sense_vectors.cpu().detach().numpy()
        else:
            # For StandardTransformer, get token embedding
            with torch.no_grad():
                token_emb = model.token_embeddings(token_id)  # (1, 1, n_embd)
                token_emb = token_emb.squeeze(0).squeeze(0)  # (1, n_embd)
                # Expand to match Backpack format (1, n_embd) -> treat as single "sense"
                representations[word] = token_emb.cpu().detach().numpy().reshape(1, -1)
    
    return representations


def get_sentence_representation(model, tokenizer, sentence, device, method='mean'):
    """
    Get sentence-level representation from Backpack model.
    
    Args:
        method: 'mean' (average across sequence), 'last' (last token), 'cls' (first token)
    """
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    if len(tokens) > model.config.block_size:
        tokens = tokens[:model.config.block_size]
    token_ids = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        # Forward through model to get hidden states
        # We'll need to modify the forward pass or extract intermediate states
        # For now, we'll use a hook or modify the model to return hidden states
        # Simplified approach: use the output before the LM head
        
        # Get sense embeddings and combine them
        B, T = token_ids.size()
        sense_embs = model.sense_embeddings(token_ids)  # (B, T, n_senses * n_embd)
        sense_embs = sense_embs.view(B, T, model.n_senses, model.config.n_embd)
        
        # Get position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = model.pos_embeddings(pos)  # (T, n_embd)
        
        # Predict sense weights
        context = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, T, n_embd)
        sense_weights = model.sense_predictor(context)  # (B, T, n_senses)
        sense_weights = torch.nn.functional.softmax(sense_weights, dim=-1)  # (B, T, n_senses)
        
        # Weighted sum of sense vectors
        x = torch.einsum('btsd,bts->btd', sense_embs, sense_weights)  # (B, T, n_embd)
        x = x + pos_emb.unsqueeze(0)
        x = model.drop(x)
        
        # Apply transformer blocks
        x = model.blocks(x)
        hidden_states = model.ln_f(x)  # (B, T, n_embd)
        
        if method == 'mean':
            sentence_repr = hidden_states.mean(dim=1)  # (B, n_embd)
        elif method == 'last':
            sentence_repr = hidden_states[:, -1, :]  # (B, n_embd)
        elif method == 'cls':
            sentence_repr = hidden_states[:, 0, :]  # (B, n_embd)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return sentence_repr.squeeze(0).cpu().detach().numpy()


def evaluate_word_similarity(model, tokenizer, word_pairs, device):
    """
    Evaluate cosine similarity between word representations.
    Useful for testing multilingual alignment.
    """
    similarities = []
    
    for word1, word2 in word_pairs:
        repr1 = get_word_representations(model, tokenizer, [word1], device)[word1]
        repr2 = get_word_representations(model, tokenizer, [word2], device)[word2]
        
        # Average sense vectors for each word
        repr1_mean = repr1.mean(axis=0)
        repr2_mean = repr2.mean(axis=0)
        
        # Cosine similarity
        cos_sim = np.dot(repr1_mean, repr2_mean) / (np.linalg.norm(repr1_mean) * np.linalg.norm(repr2_mean))
        similarities.append((word1, word2, cos_sim))
    
    return similarities


def evaluate_sentence_similarity(model, tokenizer, sentence_pairs, device, method='mean'):
    """
    Evaluate cosine similarity between sentence representations.
    """
    similarities = []
    
    for sent1, sent2 in sentence_pairs:
        repr1 = get_sentence_representation(model, tokenizer, sent1, device, method)
        repr2 = get_sentence_representation(model, tokenizer, sent2, device, method)
        
        # Cosine similarity
        cos_sim = np.dot(repr1, repr2) / (np.linalg.norm(repr1) * np.linalg.norm(repr2))
        similarities.append((sent1, sent2, cos_sim))
    
    return similarities


def analyze_sense_vectors(model, tokenizer, words, device, top_k=5):
    """
    Analyze what each sense vector predicts.
    Similar to the sense_vector.py experiment in nanoBackpackLM.
    """
    from transformers import AutoTokenizer
    
    results = {}
    
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 0:
            continue
        
        token_id = torch.tensor([tokens[0]], device=device).unsqueeze(0)
        sense_vectors = model.get_sense_vectors(token_id)  # (1, 1, n_senses, n_embd)
        sense_vectors = sense_vectors.squeeze(0).squeeze(0)  # (n_senses, n_embd)
        
        # Project each sense vector through the LM head to get predictions
        sense_predictions = []
        for sense_idx in range(sense_vectors.shape[0]):
            sense_vec = sense_vectors[sense_idx].unsqueeze(0)  # (1, n_embd)
            logits = model.lm_head(sense_vec)  # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices[0]]
            sense_predictions.append(top_tokens)
        
        results[word] = sense_predictions
    
    return results


# Expected performance benchmarks for MultiSimLex
# Based on published results from Vulic et al. (2020) and other baselines
MULTISIMLEX_BENCHMARKS = {
    'en': {
        'excellent': 0.70,  # Strong multilingual models (XLM-R, mBERT)
        'good': 0.60,      # Good multilingual models
        'baseline': 0.45,  # Basic word embeddings (Word2Vec, GloVe)
        'poor': 0.30       # Random or very weak models
    },
    'fr': {
        'excellent': 0.65,
        'good': 0.55,
        'baseline': 0.40,
        'poor': 0.25
    },
    'cross_lingual': {
        'excellent': 0.60,  # Strong cross-lingual alignment
        'good': 0.50,
        'baseline': 0.35,
        'poor': 0.20
    }
}


def evaluate_multisimlex(model, tokenizer, device, language='en', max_samples=None):
    """
    Evaluate on MultiSimLex-999 word similarity benchmark.
    
    Args:
        model: Trained Backpack or StandardTransformer model
        tokenizer: Tokenizer instance
        device: Device to run on
        language: Language code ('en', 'fr', etc.)
        max_samples: Maximum number of word pairs to evaluate (None = all)
    
    Returns:
        dict: Results with correlation, p-value, and benchmark comparison
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Install with: pip install datasets")
        return None
    
    print(f"\n{'='*60}")
    print(f"MultiSimLex Evaluation - {language.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load MultiSimLex dataset
        dataset = load_dataset("Helsinki-NLP/multisimlex", language)
    except Exception as e:
        print(f"Error loading MultiSimLex dataset for {language}: {e}")
        print("Trying alternative dataset name...")
        try:
            dataset = load_dataset("multisimlex", language)
        except Exception as e2:
            print(f"Could not load MultiSimLex: {e2}")
            return None
    
    model_similarities = []
    human_ratings = []
    skipped = 0
    
    # Limit dataset size if max_samples specified
    test_data = dataset['test']
    if max_samples is not None and max_samples < len(test_data):
        test_data = test_data.select(range(min(max_samples, len(test_data))))
        print(f"Using subset: {len(test_data)} word pairs (out of {len(dataset['test'])} total)")
    else:
        print(f"Processing {len(test_data)} word pairs...")
    
    for item in test_data:
        word1 = item['word1']
        word2 = item['word2']
        human_score = item['score']  # 0-10 scale
        
        try:
            # Get word representations
            reprs = get_word_representations(model, tokenizer, [word1, word2], device)
            
            if word1 not in reprs or word2 not in reprs:
                skipped += 1
                continue
            
            repr1 = reprs[word1]
            repr2 = reprs[word2]
            
            # Average across senses (or use single embedding for Transformer)
            repr1_mean = repr1.mean(axis=0)
            repr2_mean = repr2.mean(axis=0)
            
            # Compute cosine similarity
            cosine_sim = np.dot(repr1_mean, repr2_mean) / (np.linalg.norm(repr1_mean) * np.linalg.norm(repr2_mean))
            
            model_similarities.append(cosine_sim)
            human_ratings.append(human_score)
            
        except Exception as e:
            skipped += 1
            continue
    
    if len(model_similarities) == 0:
        print("Error: No valid word pairs processed")
        return None
    
    # Compute Spearman correlation
    correlation, p_value = spearmanr(model_similarities, human_ratings)
    
    # Compare with benchmarks
    benchmarks = MULTISIMLEX_BENCHMARKS.get(language, MULTISIMLEX_BENCHMARKS['en'])
    if correlation >= benchmarks['excellent']:
        benchmark_level = "EXCELLENT"
    elif correlation >= benchmarks['good']:
        benchmark_level = "GOOD"
    elif correlation >= benchmarks['baseline']:
        benchmark_level = "BASELINE"
    else:
        benchmark_level = "NEEDS IMPROVEMENT"
    
    print(f"\nResults:")
    print(f"  Spearman correlation: {correlation:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Number of pairs: {len(human_ratings)}")
    print(f"  Skipped pairs: {skipped}")
    print(f"\nBenchmark Comparison:")
    print(f"  Performance level: {benchmark_level}")
    print(f"  Excellent threshold: {benchmarks['excellent']:.2f}")
    print(f"  Good threshold: {benchmarks['good']:.2f}")
    print(f"  Baseline threshold: {benchmarks['baseline']:.2f}")
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'n_pairs': len(human_ratings),
        'skipped': skipped,
        'benchmark_level': benchmark_level,
        'language': language
    }


def evaluate_cross_lingual_multisimlex(model, tokenizer, device, lang1='en', lang2='fr', max_samples=None):
    """
    Evaluate cross-lingual word similarity on MultiSimLex.
    Tests if translation pairs have high similarity.
    
    Args:
        model: Trained Backpack or StandardTransformer model
        tokenizer: Tokenizer instance
        device: Device to run on
        lang1: First language code
        lang2: Second language code
        max_samples: Maximum number of word pairs to evaluate (None = all)
    
    Returns:
        dict: Results with correlation, p-value, and benchmark comparison
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Install with: pip install datasets")
        return None
    
    print(f"\n{'='*60}")
    print(f"Cross-lingual MultiSimLex Evaluation - {lang1.upper()}-{lang2.upper()}")
    print(f"{'='*60}")
    
    try:
        # Try loading cross-lingual dataset
        dataset = load_dataset("Helsinki-NLP/multisimlex", f"{lang1}-{lang2}")
    except Exception as e:
        print(f"Error loading cross-lingual MultiSimLex: {e}")
        print("Note: Cross-lingual evaluation may require aligned word pairs.")
        return None
    
    model_similarities = []
    human_ratings = []
    skipped = 0
    
    # Limit dataset size if max_samples specified
    test_data = dataset['test']
    if max_samples is not None and max_samples < len(test_data):
        test_data = test_data.select(range(min(max_samples, len(test_data))))
        print(f"Using subset: {len(test_data)} word pairs (out of {len(dataset['test'])} total)")
    else:
        print(f"Processing {len(test_data)} cross-lingual word pairs...")
    
    for item in test_data:
        # MultiSimLex cross-lingual format may vary
        # Try different possible key names
        word1_key = f'word_{lang1}' if f'word_{lang1}' in item else 'word1'
        word2_key = f'word_{lang2}' if f'word_{lang2}' in item else 'word2'
        
        word1 = item.get(word1_key, item.get('word1', ''))
        word2 = item.get(word2_key, item.get('word2', ''))
        human_score = item.get('score', item.get('similarity', 0))
        
        if not word1 or not word2:
            skipped += 1
            continue
        
        try:
            # Get representations for words in different languages
            reprs = get_word_representations(model, tokenizer, [word1, word2], device)
            
            if word1 not in reprs or word2 not in reprs:
                skipped += 1
                continue
            
            repr1 = reprs[word1]
            repr2 = reprs[word2]
            
            # Average across senses
            repr1_mean = repr1.mean(axis=0)
            repr2_mean = repr2.mean(axis=0)
            
            # Compute cosine similarity
            cosine_sim = np.dot(repr1_mean, repr2_mean) / (np.linalg.norm(repr1_mean) * np.linalg.norm(repr2_mean))
            
            model_similarities.append(cosine_sim)
            human_ratings.append(human_score)
            
        except Exception as e:
            skipped += 1
            continue
    
    if len(model_similarities) == 0:
        print("Error: No valid word pairs processed")
        return None
    
    # Compute Spearman correlation
    correlation, p_value = spearmanr(model_similarities, human_ratings)
    
    # Compare with benchmarks
    benchmarks = MULTISIMLEX_BENCHMARKS['cross_lingual']
    if correlation >= benchmarks['excellent']:
        benchmark_level = "EXCELLENT"
    elif correlation >= benchmarks['good']:
        benchmark_level = "GOOD"
    elif correlation >= benchmarks['baseline']:
        benchmark_level = "BASELINE"
    else:
        benchmark_level = "NEEDS IMPROVEMENT"
    
    print(f"\nResults:")
    print(f"  Spearman correlation: {correlation:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Number of pairs: {len(human_ratings)}")
    print(f"  Skipped pairs: {skipped}")
    print(f"\nBenchmark Comparison:")
    print(f"  Performance level: {benchmark_level}")
    print(f"  Excellent threshold: {benchmarks['excellent']:.2f}")
    print(f"  Good threshold: {benchmarks['good']:.2f}")
    print(f"  Baseline threshold: {benchmarks['baseline']:.2f}")
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'n_pairs': len(human_ratings),
        'skipped': skipped,
        'benchmark_level': benchmark_level,
        'languages': f"{lang1}-{lang2}"
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Backpack Language Model')
    parser.add_argument('--out_dir', type=str, required=True, help='Model output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--tokenizer_name', type=str, default='xlm-roberta-base', help='Tokenizer name')
    parser.add_argument('--multisimlex', action='store_true', help='Run MultiSimLex evaluation')
    parser.add_argument('--languages', nargs='+', default=['en', 'fr'], help='Languages for MultiSimLex evaluation')
    parser.add_argument('--cross_lingual', action='store_true', help='Run cross-lingual MultiSimLex evaluation')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load model
    print(f"Loading model from {args.out_dir}...")
    model, config = load_model(args.out_dir, device)
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    print("\n=== Word-level Evaluation ===")
    
    # Test words in both languages
    test_words_en = ['hello', 'world', 'language', 'model', 'learning']
    test_words_fr = ['bonjour', 'monde', 'langue', 'modèle', 'apprentissage']
    
    print("\nEnglish word sense vectors:")
    en_reprs = get_word_representations(model, tokenizer, test_words_en, device)
    for word, repr in en_reprs.items():
        print(f"  {word}: shape {repr.shape}")
    
    print("\nFrench word sense vectors:")
    fr_reprs = get_word_representations(model, tokenizer, test_words_fr, device)
    for word, repr in fr_reprs.items():
        print(f"  {word}: shape {repr.shape}")
    
    # Analyze sense vectors
    print("\n=== Sense Vector Analysis ===")
    print("\nAnalyzing English words:")
    en_senses = analyze_sense_vectors(model, tokenizer, test_words_en[:3], device)
    for word, predictions in en_senses.items():
        print(f"\n{word}:")
        for sense_idx, preds in enumerate(predictions):
            print(f"  Sense {sense_idx}: {preds}")
    
    print("\nAnalyzing French words:")
    fr_senses = analyze_sense_vectors(model, tokenizer, test_words_fr[:3], device)
    for word, predictions in fr_senses.items():
        print(f"\n{word}:")
        for sense_idx, preds in enumerate(predictions):
            print(f"  Sense {sense_idx}: {preds}")
    
    # Evaluate cross-lingual similarity
    print("\n=== Cross-lingual Word Similarity ===")
    translation_pairs = [
        ('hello', 'bonjour'),
        ('world', 'monde'),
        ('language', 'langue'),
        ('model', 'modèle'),
        ('learning', 'apprentissage'),
    ]
    
    similarities = evaluate_word_similarity(model, tokenizer, translation_pairs, device)
    print("\nTranslation pair similarities:")
    for word1, word2, sim in similarities:
        print(f"  {word1} <-> {word2}: {sim:.4f}")
    
    # Sentence-level evaluation
    print("\n=== Sentence-level Evaluation ===")
    test_sentences_en = [
        "Hello, how are you?",
        "The language model is learning.",
        "This is a test sentence.",
    ]
    test_sentences_fr = [
        "Bonjour, comment allez-vous?",
        "Le modèle de langue apprend.",
        "Ceci est une phrase de test.",
    ]
    
    print("\nEnglish sentence representations:")
    for sent in test_sentences_en:
        repr = get_sentence_representation(model, tokenizer, sent, device, method='mean')
        print(f"  {sent}: shape {repr.shape}")
    
    print("\nFrench sentence representations:")
    for sent in test_sentences_fr:
        repr = get_sentence_representation(model, tokenizer, sent, device, method='mean')
        print(f"  {sent}: shape {repr.shape}")
    
    # Cross-lingual sentence similarity
    print("\n=== Cross-lingual Sentence Similarity ===")
    sentence_pairs = list(zip(test_sentences_en, test_sentences_fr))
    sent_similarities = evaluate_sentence_similarity(model, tokenizer, sentence_pairs, device)
    print("\nTranslation pair sentence similarities:")
    for sent1, sent2, sim in sent_similarities:
        print(f"  EN: {sent1[:50]}...")
        print(f"  FR: {sent2[:50]}...")
        print(f"  Similarity: {sim:.4f}\n")
    
    # MultiSimLex evaluation
    if args.multisimlex:
        print("\n" + "="*60)
        print("MultiSimLex Benchmark Evaluation")
        print("="*60)
        
        results = {}
        for lang in args.languages:
            result = evaluate_multisimlex(model, tokenizer, device, language=lang)
            if result:
                results[lang] = result
        
        # Cross-lingual evaluation
        if args.cross_lingual and len(args.languages) >= 2:
            lang1, lang2 = args.languages[0], args.languages[1]
            cross_result = evaluate_cross_lingual_multisimlex(model, tokenizer, device, lang1, lang2)
            if cross_result:
                results[f'{lang1}-{lang2}'] = cross_result
        
        # Summary
        if results:
            print("\n" + "="*60)
            print("MultiSimLex Summary")
            print("="*60)
            for key, result in results.items():
                print(f"{key.upper()}: {result['correlation']:.4f} ({result['benchmark_level']})")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

