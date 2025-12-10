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
    
    # Load checkpoint - handle both old and new PyTorch versions
    try:
        # Try with safe_globals (PyTorch 2.6+)
        try:
            with torch.serialization.safe_globals([ModelConfig]):
                checkpoint = torch.load(ckpt_path, map_location=device)
        except AttributeError:
            # Fallback to weights_only=False for older PyTorch or if safe_globals not available
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        # Final fallback
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    
    # Determine model type based on checkpoint state dict keys
    state_dict_keys = list(checkpoint['model'].keys())
    
    # Check for BackpackLM-specific keys
    # BackpackLM has: token_embedding (singular), sense_layer, sense_predictor
    # StandardTransformerLM has: token_embeddings (plural), no sense_layer or sense_predictor
    has_backpack_keys = (
        any('sense_layer' in k for k in state_dict_keys) or
        any('sense_predictor' in k for k in state_dict_keys) or
        'token_embedding.weight' in state_dict_keys  # BackpackLM uses singular
    )
    has_transformer_keys = 'token_embeddings.weight' in state_dict_keys  # StandardTransformerLM uses plural
    
    # Also check for explicit model_type in checkpoint or config
    model_type = None
    if hasattr(config, 'model_type'):
        model_type = config.model_type
    elif 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    
    # Determine model type
    if model_type:
        # Use explicit model_type if available
        if model_type == 'backpack':
            is_transformer = False
        elif model_type == 'transformer':
            is_transformer = True
        else:
            # Fallback to key-based detection
            is_transformer = not has_backpack_keys
    else:
        # Infer from state dict keys
        if has_backpack_keys:
            is_transformer = False
        elif has_transformer_keys:
            is_transformer = True
        else:
            # Fallback: check n_senses in config
            if hasattr(config, 'n_senses'):
                is_transformer = config.n_senses == 1
            else:
                # Default to BackpackLM if uncertain (safer)
                print("Warning: Could not determine model type, defaulting to BackpackLM")
                is_transformer = False
    
    if is_transformer:
        print("Loading StandardTransformerLM model...")
        model = StandardTransformerLM(config)
    else:
        print("Loading BackpackLM model...")
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
        # Use get_sense_vectors which handles the current architecture
        sense_embs = model.get_sense_vectors(token_ids)  # (B, T, n_senses, n_embd)
        
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
            print(f"  Warning: Could not tokenize '{word}', skipping...")
            continue
        
        try:
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
            print(f"\n{word}:")
            for sense_idx, preds in enumerate(sense_predictions):
                print(f"  Sense {sense_idx}: {preds}")
        except Exception as e:
            print(f"  Error analyzing '{word}': {e}")
            continue
    
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


def load_test_data(data_dir='data/europarl', language_pair='en-fr', max_samples=1000, split='validation'):
    """
    Load test/validation parallel sentences from Europarl dataset.
    
    Args:
        data_dir: Directory containing Europarl data
        language_pair: Language pair (e.g., 'en-fr')
        max_samples: Maximum number of sentence pairs to load
        split: 'validation' or 'test' for validation/test set
    
    Returns:
        List of (source_text, target_text) tuples
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Install with: pip install datasets")
        return None
    
    lang1, lang2 = language_pair.split('-')
    
    # Map 'val' to 'validation' for compatibility
    if split == 'val':
        split = 'validation'
    
    try:
        # Try loading from HuggingFace datasets
        try:
            dataset = load_dataset("europarl_bilingual", language_pair, split=split)
            print(f"Loaded Europarl {language_pair} {split} set from HuggingFace")
        except:
            # Try with 'test' split if 'validation' doesn't exist
            if split == 'validation':
                try:
                    dataset = load_dataset("europarl_bilingual", language_pair, split='test')
                    print(f"Loaded Europarl {language_pair} test set from HuggingFace (using test as validation)")
                except:
                    # Try train split and take a subset
                    dataset = load_dataset("europarl_bilingual", language_pair, split='train')
                    # Take last 10% as validation
                    val_size = len(dataset) // 10
                    dataset = dataset.select(range(len(dataset) - val_size, len(dataset)))
                    print(f"Using last {len(dataset)} samples from train set as validation")
            else:
                raise
    except:
        try:
            # Try OPUS dataset
            try:
                dataset = load_dataset("opus100", language_pair, split=split)
                print(f"Using OPUS100 {language_pair} {split} set")
            except:
                if split == 'validation':
                    dataset = load_dataset("opus100", language_pair, split='test')
                    print(f"Using OPUS100 {language_pair} test set (as validation)")
                else:
                    raise
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Try loading from segregated files if available
            try:
                from data.europarl.read_segregated import SegregatedDataReader
                reader = SegregatedDataReader(data_dir, language_pair)
                pairs = reader.get_parallel_pairs(limit=max_samples)
                if pairs:
                    print(f"Loaded {len(pairs)} pairs from segregated files")
                    return [(pair[1], pair[2]) for pair in pairs]  # (lang1_text, lang2_text)
                else:
                    raise FileNotFoundError("No pairs found in segregated files")
            except Exception as e2:
                print(f"Could not load test data: {e2}")
                # Last resort: try to load from validation binary file if it exists
                try:
                    import pickle
                    import numpy as np
                    val_file = os.path.join(data_dir, 'val.bin')
                    if os.path.exists(val_file):
                        print(f"Attempting to load from {val_file}...")
                        # This would require the data loading logic from train.py
                        print("Note: Binary file loading requires data preparation. Please prepare test data first.")
                        return None
                    else:
                        return None
                except:
                    return None
    
    # Limit dataset size
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    # Extract parallel sentences
    parallel_pairs = []
    for item in dataset:
        if 'translation' in item:
            text1 = item['translation'].get(lang1, '')
            text2 = item['translation'].get(lang2, '')
        else:
            text1 = item.get(lang1, item.get('en', ''))
            text2 = item.get(lang2, item.get('fr', ''))
        
        if text1 and text2:
            parallel_pairs.append((text1, text2))
    
    print(f"Loaded {len(parallel_pairs)} parallel sentence pairs")
    return parallel_pairs


def generate_translation(model, tokenizer, source_text, device, max_new_tokens=100, temperature=1.0, top_k=None):
    """
    Generate translation from source text using the model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        source_text: Source sentence to translate
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
    
    Returns:
        Generated translation text
    """
    # Tokenize source text
    source_ids = tokenizer.encode(source_text, add_special_tokens=True)
    source_ids = torch.tensor([source_ids], dtype=torch.long, device=device)
    
    # Generate translation
    with torch.no_grad():
        generated_ids = model.generate(
            source_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    
    # Remove source text from generated text (if it was included)
    if generated_text.startswith(source_text):
        generated_text = generated_text[len(source_text):].strip()
    
    return generated_text


def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score between reference and candidate translations.
    
    Args:
        reference: Reference translation (string)
        candidate: Candidate translation (string)
    
    Returns:
        BLEU score (float)
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        # Download required NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    except ImportError:
        print("Error: nltk not installed. Install with: pip install nltk")
        return None
    
    # Tokenize sentences
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    
    # Use smoothing function to handle cases where n-grams don't match
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU score
    try:
        bleu_score = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            smoothing_function=smoothing
        )
        return bleu_score
    except:
        return 0.0


def calculate_sacrebleu(references, candidates):
    """
    Calculate BLEU score using sacrebleu (more standard).
    
    Args:
        references: List of reference translations
        candidates: List of candidate translations
    
    Returns:
        BLEU score dict with score and other metrics
    """
    try:
        import sacrebleu
    except ImportError:
        print("Warning: sacrebleu not installed. Using NLTK BLEU instead.")
        print("For better results, install sacrebleu: pip install sacrebleu")
        return None
    
    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(candidates, [references])
    
    return {
        'score': bleu.score / 100.0,  # Convert to 0-1 scale
        'precisions': [p / 100.0 for p in bleu.precisions],
        'bp': bleu.bp,
        'ratio': bleu.ratio,
        'sys_len': bleu.sys_len,
        'ref_len': bleu.ref_len
    }


def evaluate_translation_bleu(model, tokenizer, test_pairs, device, max_samples=None, 
                              max_new_tokens=100, temperature=1.0, top_k=None):
    """
    Evaluate translation quality using BLEU scores.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        test_pairs: List of (source_text, target_text) tuples
        device: Device to run on
        max_samples: Maximum number of pairs to evaluate
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
    
    Returns:
        dict: Results with BLEU scores and statistics
    """
    print(f"\n{'='*60}")
    print("TRANSLATION BLEU SCORE EVALUATION")
    print(f"{'='*60}")
    
    if max_samples and max_samples < len(test_pairs):
        test_pairs = test_pairs[:max_samples]
        print(f"Evaluating {max_samples} sentence pairs (out of {len(test_pairs)} total)")
    else:
        print(f"Evaluating {len(test_pairs)} sentence pairs...")
    
    references = []
    candidates = []
    bleu_scores = []
    
    print("Generating translations...")
    for i, (source_text, target_text) in enumerate(test_pairs):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_pairs)} pairs...")
        
        # Generate translation
        try:
            generated_text = generate_translation(
                model, tokenizer, source_text, device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            
            references.append(target_text)
            candidates.append(generated_text)
            
            # Calculate individual BLEU score
            bleu_score = calculate_bleu_score(target_text, generated_text)
            if bleu_score is not None:
                bleu_scores.append(bleu_score)
        except Exception as e:
            print(f"  Error translating pair {i+1}: {e}")
            continue
    
    if len(bleu_scores) == 0:
        print("Error: No valid translations generated")
        return None
    
    # Calculate corpus-level BLEU using sacrebleu if available
    sacrebleu_result = calculate_sacrebleu(references, candidates)
    
    # Calculate statistics
    avg_bleu = np.mean(bleu_scores)
    std_bleu = np.std(bleu_scores)
    median_bleu = np.median(bleu_scores)
    
    print(f"\nResults:")
    print(f"  Number of pairs evaluated: {len(bleu_scores)}")
    print(f"  Average BLEU score: {avg_bleu:.4f}")
    print(f"  Median BLEU score: {median_bleu:.4f}")
    print(f"  Std deviation: {std_bleu:.4f}")
    print(f"  Min BLEU: {np.min(bleu_scores):.4f}")
    print(f"  Max BLEU: {np.max(bleu_scores):.4f}")
    
    if sacrebleu_result:
        print(f"\nSacreBLEU (corpus-level):")
        print(f"  BLEU score: {sacrebleu_result['score']:.4f}")
        print(f"  Precisions (1-4): {[f'{p:.4f}' for p in sacrebleu_result['precisions']]}")
        print(f"  Brevity penalty: {sacrebleu_result['bp']:.4f}")
    
    return {
        'n_pairs': len(bleu_scores),
        'avg_bleu': float(avg_bleu),
        'median_bleu': float(median_bleu),
        'std_bleu': float(std_bleu),
        'min_bleu': float(np.min(bleu_scores)),
        'max_bleu': float(np.max(bleu_scores)),
        'sacrebleu': sacrebleu_result,
        'individual_scores': bleu_scores[:10]  # Store first 10 for reference
    }


def evaluate_translation_accuracy(model, tokenizer, test_pairs, device, max_samples=None,
                                  max_new_tokens=100, temperature=1.0, top_k=None):
    """
    Evaluate translation accuracy using exact match and word-level metrics.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        test_pairs: List of (source_text, target_text) tuples
        device: Device to run on
        max_samples: Maximum number of pairs to evaluate
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
    
    Returns:
        dict: Results with accuracy metrics
    """
    print(f"\n{'='*60}")
    print("TRANSLATION ACCURACY EVALUATION")
    print(f"{'='*60}")
    
    if max_samples and max_samples < len(test_pairs):
        test_pairs = test_pairs[:max_samples]
        print(f"Evaluating {max_samples} sentence pairs (out of {len(test_pairs)} total)")
    else:
        print(f"Evaluating {len(test_pairs)} sentence pairs...")
    
    exact_matches = 0
    word_accuracies = []
    char_accuracies = []
    
    print("Generating translations...")
    for i, (source_text, target_text) in enumerate(test_pairs):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_pairs)} pairs...")
        
        try:
            # Generate translation
            generated_text = generate_translation(
                model, tokenizer, source_text, device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            
            # Exact match
            if generated_text.lower().strip() == target_text.lower().strip():
                exact_matches += 1
            
            # Word-level accuracy
            ref_words = set(target_text.lower().split())
            gen_words = set(generated_text.lower().split())
            if len(ref_words) > 0:
                word_overlap = len(ref_words & gen_words) / len(ref_words)
                word_accuracies.append(word_overlap)
            
            # Character-level accuracy (Levenshtein-like)
            ref_chars = len(target_text.replace(' ', ''))
            gen_chars = len(generated_text.replace(' ', ''))
            if ref_chars > 0:
                char_accuracy = min(gen_chars / ref_chars, 1.0) if gen_chars <= ref_chars else ref_chars / gen_chars
                char_accuracies.append(char_accuracy)
        except Exception as e:
            print(f"  Error translating pair {i+1}: {e}")
            continue
    
    n_evaluated = len(word_accuracies)
    if n_evaluated == 0:
        print("Error: No valid translations generated")
        return None
    
    exact_match_rate = exact_matches / n_evaluated
    avg_word_accuracy = np.mean(word_accuracies) if word_accuracies else 0.0
    avg_char_accuracy = np.mean(char_accuracies) if char_accuracies else 0.0
    
    print(f"\nResults:")
    print(f"  Number of pairs evaluated: {n_evaluated}")
    print(f"  Exact match rate: {exact_match_rate:.4f} ({exact_matches}/{n_evaluated})")
    print(f"  Average word-level accuracy: {avg_word_accuracy:.4f}")
    print(f"  Average character-level accuracy: {avg_char_accuracy:.4f}")
    
    return {
        'n_pairs': n_evaluated,
        'exact_match_rate': float(exact_match_rate),
        'exact_matches': exact_matches,
        'avg_word_accuracy': float(avg_word_accuracy),
        'avg_char_accuracy': float(avg_char_accuracy),
        'word_accuracies': word_accuracies[:10]  # Store first 10 for reference
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

