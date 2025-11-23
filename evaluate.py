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
from model import BackpackLM
from configurator import ModelConfig


def load_model(out_dir, device):
    """Load trained model"""
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


def get_word_representations(model, tokenizer, words, device):
    """
    Extract word representations (sense vectors) for given words.
    
    Returns:
        dict: {word: sense_vectors} where sense_vectors is (n_senses, n_embd)
    """
    representations = {}
    
    for word in words:
        # Tokenize word
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 0:
            continue
        
        # Get sense vectors for the first token (or average if multiple tokens)
        token_id = torch.tensor([tokens[0]], device=device).unsqueeze(0)
        sense_vectors = model.get_sense_vectors(token_id)  # (1, 1, n_senses, n_embd)
        sense_vectors = sense_vectors.squeeze(0).squeeze(0)  # (n_senses, n_embd)
        
        representations[word] = sense_vectors.cpu().detach().numpy()
    
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate Backpack Language Model')
    parser.add_argument('--out_dir', type=str, required=True, help='Model output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--tokenizer_name', type=str, default='xlm-roberta-base', help='Tokenizer name')
    
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
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()

