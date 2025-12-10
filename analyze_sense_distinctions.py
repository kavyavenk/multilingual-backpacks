#!/usr/bin/env python3
"""
Advanced sense analysis that distinguishes senses by:
1. Embedding-space similarity (relatedness)
2. Contextual activation patterns
3. Syntactic patterns (next wordpiece, verb objects, nmod nouns)
4. Proper noun associations
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from evaluate import load_model, _is_english_or_french
from transformers import AutoTokenizer


def get_sense_embedding_similarities(model, tokenizer, sense_vectors, vocab_sample_size=100000, device='cpu'):
    """
    Find words that are semantically similar to each sense vector in embedding space.
    Only returns English/French tokens.
    """
    # Sample vocabulary tokens (not all, too expensive)
    vocab_size = model.config.vocab_size
    sample_indices = torch.randperm(vocab_size)[:vocab_sample_size].to(device)
    
    # Get embeddings for sampled vocabulary
    sample_embeddings = model.token_embedding(sample_indices)  # (sample_size, n_embd)
    
    n_senses = sense_vectors.shape[0]
    sense_similar_words = {}
    
    for sense_idx in range(n_senses):
        sense_vec = sense_vectors[sense_idx].unsqueeze(0)  # (1, n_embd)
        
        # Compute cosine similarity
        similarities = F.cosine_similarity(
            sense_vec, sample_embeddings, dim=1
        )  # (sample_size,)
        
        # Get all similarities and filter to English/French
        all_pairs = [(sim.item(), idx.item()) for idx, sim in zip(sample_indices, similarities)]
        all_pairs.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity
        
        similar_words = []
        checked_count = 0
        for sim, token_id in all_pairs:
            checked_count += 1
            token_str = tokenizer.decode([token_id])
            # Only keep English/French tokens
            if _is_english_or_french(token_str):
                similar_words.append((token_str, sim))
                if len(similar_words) >= 20:  # Stop once we have enough
                    break
            # Stop if we've checked too many without finding enough
            if checked_count > 5000 and len(similar_words) < 3:
                break
        
        sense_similar_words[sense_idx] = similar_words
    
    return sense_similar_words


def analyze_contextual_activation(model, tokenizer, test_sentences, device='cpu'):
    """
    Analyze when each sense is activated in different contexts.
    """
    sense_activations = defaultdict(list)  # sense_idx -> list of (sentence, word, weight)
    
    for sentence in test_sentences:
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        if len(tokens) > model.config.block_size:
            tokens = tokens[:model.config.block_size]
        
        token_ids = torch.tensor([tokens], device=device)
        
        with torch.no_grad():
            B, T = token_ids.size()
            
            # Get position embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = model.pos_embeddings(pos)
            
            # Predict sense weights
            context = pos_emb.unsqueeze(0).expand(B, -1, -1)
            sense_weights = model.sense_predictor(context)
            sense_weights = F.softmax(sense_weights, dim=-1)  # (B, T, n_senses)
            
            # For each token position, find which sense is most activated
            for t in range(T):
                token_str = tokenizer.decode([tokens[t]])
                weights = sense_weights[0, t, :].cpu().numpy()  # (n_senses,)
                
                # Record top activated senses
                top_senses = np.argsort(weights)[::-1][:3]  # Top 3
                for sense_idx in top_senses:
                    sense_activations[sense_idx].append((
                        sentence,
                        token_str,
                        float(weights[sense_idx])
                    ))
    
    return sense_activations


def analyze_syntactic_patterns(model, tokenizer, words, device='cpu', context_window=3):
    """
    Analyze syntactic patterns: what comes before/after, verb objects, etc.
    """
    # Create test contexts
    test_contexts = []
    
    # Verb + object patterns
    verb_object_patterns = [
        ("support", "the proposal"),
        ("soutenir", "la proposition"),
        ("discuss", "the issue"),
        ("discuter", "le problème"),
        ("present", "the report"),
        ("présenter", "le rapport"),
    ]
    
    # Noun + modifier patterns
    noun_modifier_patterns = [
        ("the important", "proposal"),
        ("la importante", "proposition"),
        ("the European", "parliament"),
        ("le européen", "parlement"),
    ]
    
    # Proper noun patterns
    proper_noun_patterns = [
        ("European", "Parliament"),
        ("Parlement", "européen"),
        ("United", "States"),
        ("États", "Unis"),
    ]
    
    all_patterns = verb_object_patterns + noun_modifier_patterns + proper_noun_patterns
    
    sense_contexts = defaultdict(list)  # sense_idx -> list of contexts
    
    for word in words:
        # Test word in different positions
        for before, after in all_patterns:
            # Word after context
            context_before = f"{before} {word}"
            # Word before context  
            context_after = f"{word} {after}"
            
            for context, position in [(context_before, 'after'), (context_after, 'before')]:
                tokens = tokenizer.encode(context, add_special_tokens=True)
                if len(tokens) > model.config.block_size:
                    continue
                
                token_ids = torch.tensor([tokens], device=device)
                
                with torch.no_grad():
                    B, T = token_ids.size()
                    word_pos = None
                    
                    # Find word position
                    for i in range(T):
                        decoded = tokenizer.decode([tokens[i]])
                        if word.lower() in decoded.lower() or decoded.lower() in word.lower():
                            word_pos = i
                            break
                    
                    if word_pos is None:
                        continue
                    
                    # Get sense weights
                    pos = torch.arange(0, T, dtype=torch.long, device=device)
                    pos_emb = model.pos_embeddings(pos)
                    context_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
                    sense_weights = model.sense_predictor(context_emb)
                    sense_weights = F.softmax(sense_weights, dim=-1)
                    
                    # Get top activated sense
                    weights = sense_weights[0, word_pos, :].cpu().numpy()
                    top_sense = np.argmax(weights)
                    
                    sense_contexts[top_sense].append({
                        'word': word,
                        'context': context,
                        'position': position,
                        'pattern_type': 'verb_object' if (before, after) in verb_object_patterns else
                                       'noun_modifier' if (before, after) in noun_modifier_patterns else
                                       'proper_noun',
                        'weight': float(weights[top_sense])
                    })
    
    return sense_contexts


def analyze_sense_distinctions(model, tokenizer, words, device='cpu'):
    """
    Comprehensive analysis to distinguish senses.
    """
    print("="*70)
    print("ADVANCED SENSE DISTINCTION ANALYSIS")
    print("="*70)
    
    n_senses = model.n_senses
    
    # 1. Get sense vectors for a sample word
    sample_word = words[0] if words else 'hello'
    tokens = tokenizer.encode(sample_word, add_special_tokens=False)
    if len(tokens) == 0:
        print(f"Could not tokenize '{sample_word}'")
        return
    
    token_id = torch.tensor([tokens[0]], device=device).unsqueeze(0)
    sense_vectors = model.get_sense_vectors(token_id).squeeze(0).squeeze(0)  # (n_senses, n_embd)
    
    print(f"\n1. EMBEDDING-SPACE SIMILARITY (Relatedness)")
    print("-" * 70)
    print("Finding words semantically similar to each sense in embedding space...")
    
    similar_words = get_sense_embedding_similarities(model, tokenizer, sense_vectors, vocab_sample_size=2000, device=device)
    
    for sense_idx in range(n_senses):
        print(f"\nSense {sense_idx:2d} - Semantically Similar Words:")
        for i, (word, sim) in enumerate(similar_words[sense_idx][:10], 1):
            print(f"  {i:2d}. {word:20s} (cosine sim: {sim:.3f})")
    
    # 2. Contextual activation patterns
    print(f"\n\n2. CONTEXTUAL ACTIVATION PATTERNS")
    print("-" * 70)
    print("Analyzing when each sense is activated in different contexts...")
    
    test_sentences = [
        "The parliament discussed the proposal",
        "Le parlement a discuté la proposition",
        "We support the European Union",
        "Nous soutenons l'Union européenne",
        "The important debate continues",
        "Le débat important continue",
        "Hello world, how are you?",
        "Bonjour monde, comment allez-vous?",
    ]
    
    context_activations = analyze_contextual_activation(model, tokenizer, test_sentences, device)
    
    for sense_idx in range(n_senses):
        if sense_idx in context_activations:
            activations = sorted(context_activations[sense_idx], key=lambda x: x[2], reverse=True)[:5]
            print(f"\nSense {sense_idx:2d} - Most Activated In:")
            for sentence, word, weight in activations:
                print(f"  '{sentence[:50]}...' (word: '{word}', weight: {weight:.3f})")
    
    # 3. Syntactic patterns
    print(f"\n\n3. SYNTACTIC PATTERNS")
    print("-" * 70)
    print("Analyzing syntactic relationships (verb objects, noun modifiers, proper nouns)...")
    
    syntactic_patterns = analyze_syntactic_patterns(model, tokenizer, words[:5], device)
    
    pattern_types = ['verb_object', 'noun_modifier', 'proper_noun']
    pattern_names = {
        'verb_object': 'Verb Objects',
        'noun_modifier': 'Noun Modifiers (nmod)',
        'proper_noun': 'Proper Noun Associations'
    }
    
    for sense_idx in range(n_senses):
        if sense_idx in syntactic_patterns:
            patterns = syntactic_patterns[sense_idx]
            print(f"\nSense {sense_idx:2d} - Syntactic Patterns:")
            
            for pattern_type in pattern_types:
                type_patterns = [p for p in patterns if p['pattern_type'] == pattern_type]
                if type_patterns:
                    print(f"  {pattern_names[pattern_type]}:")
                    for p in type_patterns[:3]:
                        print(f"    - '{p['context']}' ({p['position']}, weight: {p['weight']:.3f})")
    
    # 4. Sense vector similarity matrix
    print(f"\n\n4. SENSE SIMILARITY MATRIX")
    print("-" * 70)
    print("How similar are senses to each other? (cosine similarity)")
    
    similarity_matrix = np.zeros((n_senses, n_senses))
    for i in range(n_senses):
        for j in range(n_senses):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = F.cosine_similarity(
                    sense_vectors[i].unsqueeze(0),
                    sense_vectors[j].unsqueeze(0),
                    dim=1
                ).item()
                similarity_matrix[i, j] = sim
    
    print("\n     ", end="")
    for j in range(min(n_senses, 16)):
        print(f" S{j:2d}", end="")
    print()
    
    for i in range(min(n_senses, 16)):
        print(f" S{i:2d} ", end="")
        for j in range(min(n_senses, 16)):
            print(f"{similarity_matrix[i, j]:5.2f}", end="")
        print()
    
    # 5. Summary: What distinguishes each sense
    print(f"\n\n5. SENSE DISTINCTIONS SUMMARY")
    print("-" * 70)
    
    for sense_idx in range(n_senses):
        print(f"\nSense {sense_idx:2d}:")
        
        # Most similar words
        if sense_idx in similar_words:
            top_words = [w for w, _ in similar_words[sense_idx][:5]]
            print(f"  Related words: {', '.join(top_words)}")
        
        # Most common contexts
        if sense_idx in context_activations:
            contexts = Counter([s[:30] for s, _, _ in context_activations[sense_idx][:10]])
            print(f"  Common contexts: {', '.join(list(contexts.keys())[:3])}")
        
        # Syntactic preferences
        if sense_idx in syntactic_patterns:
            patterns = syntactic_patterns[sense_idx]
            pattern_counts = Counter([p['pattern_type'] for p in patterns])
            if pattern_counts:
                top_pattern = pattern_counts.most_common(1)[0][0]
                print(f"  Syntactic role: {pattern_names.get(top_pattern, top_pattern)}")
        
        # Similarity to other senses
        similar_senses = []
        for j in range(n_senses):
            if i != j and similarity_matrix[sense_idx, j] > 0.7:
                similar_senses.append(f"S{j}")
        if similar_senses:
            print(f"  Similar to: {', '.join(similar_senses)} (may be redundant)")
        else:
            print(f"  Distinct from other senses (good diversity)")


def main():
    parser = argparse.ArgumentParser(description='Analyze sense distinctions')
    parser.add_argument('--out_dir', type=str, default='out/backpack_full',
                       help='Directory containing trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.out_dir}...")
    model, config = load_model(args.out_dir, args.device)
    
    # Load tokenizer
    tokenizer_name = config.tokenizer_name if hasattr(config, 'tokenizer_name') else 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if not hasattr(config, 'n_senses'):
        print("Error: Model doesn't have n_senses attribute. Is this a Backpack model?")
        return
    
    print(f"Model has {config.n_senses} senses\n")
    
    # Test words
    test_words = [
        'hello', 'bonjour', 'world', 'monde',
        'parliament', 'parlement', 'support', 'soutenir',
        'proposal', 'proposition', 'European', 'européen'
    ]
    
    analyze_sense_distinctions(model, tokenizer, test_words, args.device)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
