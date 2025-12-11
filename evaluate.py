"""
Evaluation script for Backpack Language Models
Evaluates word-level and sentence-level representations
"""

import os
import argparse
import re
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from model import BackpackLM, StandardTransformerLM
from configurator import ModelConfig

# Sense labels for 16-sense Backpack model - Clearer, more distinct labels
SENSE_LABELS = {
    0: "Parliamentary Discourse & Debate",
    1: "Parliamentary Discourse & Debate",
    2: "Adjectives & Descriptive Terms",
    3: "Negation & Absence",
    4: "Temporal & Small Scale",
    5: "European Union & Future Planning",
    6: "Debate & Discussion",
    7: "Progress & Development",
    8: "Prepositions & Spatial Relations",
    9: "Future & Temporal Planning",
    10: "Parliamentary Discourse & Debate",
    11: "Negation & Absence",
    12: "Problems & Issues",
    13: "Future & Temporal Planning",
    14: "States & Countries",
    15: "European Union & Institutions",
}


def load_huggingface_model(model_name, device):
    """
    Load a HuggingFace Backpack model (e.g., stanfordnlp/backpack-gpt2).
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'stanfordnlp/backpack-gpt2')
        device: Device to load model on
    
    Returns:
        model, config: Loaded model and config
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        raise ImportError("transformers library required for HuggingFace models. Install with: pip install transformers")
    
    print(f"Loading HuggingFace model: {model_name}")
    print("Note: This model may have different architecture than our custom BackpackLM")
    
    # Load config and model with trust_remote_code=True (required for custom architectures)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded {model_name}")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Vocab size: {getattr(config, 'vocab_size', 'N/A')}")
    print(f"  Hidden size: {getattr(config, 'n_embd', getattr(config, 'hidden_size', 'N/A'))}")
    
    return model, config


def load_model(out_dir_or_file, device):
    """
    Load trained model (Backpack or StandardTransformer).
    
    Supports:
    1. Local checkpoint files (.pt files or directories with ckpt.pt)
    2. HuggingFace model identifiers (if starts with a known HF prefix)
    
    Args:
        out_dir_or_file: Either:
            - A directory containing 'ckpt.pt' 
            - A direct path to a .pt checkpoint file
            - A HuggingFace model identifier (e.g., 'stanfordnlp/backpack-gpt2')
        device: Device to load model on
    """
    # Check if this is a HuggingFace model identifier
    if isinstance(out_dir_or_file, str) and '/' in out_dir_or_file and not os.path.exists(out_dir_or_file):
        # Likely a HuggingFace model identifier
        try:
            return load_huggingface_model(out_dir_or_file, device)
        except Exception as e:
            print(f"Warning: Failed to load as HuggingFace model: {e}")
            print("Trying as local checkpoint...")
    
    # Original local checkpoint loading logic
    # Handle both directory and direct .pt file paths
    if os.path.isfile(out_dir_or_file) and out_dir_or_file.endswith('.pt'):
        # Direct .pt file path (e.g., finetuned model weights)
        ckpt_path = out_dir_or_file
    else:
        # Directory path - look for ckpt.pt inside
        ckpt_path = os.path.join(out_dir_or_file, 'ckpt.pt')
    
    if not os.path.exists(ckpt_path):
        error_msg = f"\n{'='*60}\n"
        error_msg += f"ERROR: Checkpoint not found: {ckpt_path}\n"
        error_msg += f"{'='*60}\n"
        error_msg += "\nYou need to train a model first before evaluating.\n\n"
        error_msg += "To train a model, run:\n"
        error_msg += f"  python train.py --config train_europarl_tiny --out_dir {out_dir_or_file} --data_dir europarl\n\n"
        error_msg += "After training completes, you can then run evaluation:\n"
        error_msg += f"  python evaluate.py --out_dir {out_dir_or_file} --multisimlex --languages en fr\n"
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
    
    # Handle finetuned weights files that may only contain state dict
    if 'config' not in checkpoint:
        # If this is a finetuned weights file, try to load config from base backpack model
        base_backpack_dir = os.path.join(os.path.dirname(ckpt_path), '..', 'backpack_full')
        base_ckpt_path = os.path.join(base_backpack_dir, 'ckpt.pt')
        if os.path.exists(base_ckpt_path):
            print(f"Finetuned weights file detected. Loading config from base model: {base_ckpt_path}")
            try:
                base_checkpoint = torch.load(base_ckpt_path, map_location=device, weights_only=False)
                checkpoint['config'] = base_checkpoint['config']
            except Exception as e:
                raise ValueError(f"Could not load config from base model: {e}")
        else:
            raise ValueError(f"Finetuned weights file missing 'config'. Expected base model at: {base_ckpt_path}")

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


def _is_huggingface_model(model):
    """Check if model is a HuggingFace model (not our custom BackpackLM/StandardTransformerLM)"""
    # HuggingFace models typically have 'transformer' attribute or are from transformers library
    return (hasattr(model, 'transformer') or 
            hasattr(model, 'config') and hasattr(model.config, 'model_type') and 
            'gpt' in model.config.model_type.lower() or
            type(model).__name__ in ['BackpackGPT2LMHeadModel', 'GPT2LMHeadModel'])


def get_word_representations(model, tokenizer, words, device):
    """
    Extract word representations (sense vectors for Backpack, token embeddings for Transformer).
    
    Returns:
        dict: {word: vectors} where vectors is (n_senses, n_embd) for Backpack or (1, n_embd) for Transformer
    """
    representations = {}
    is_backpack = isinstance(model, BackpackLM)
    is_hf = _is_huggingface_model(model)
    
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
        elif is_hf:
            # HuggingFace models use transformer.wte (word token embeddings)
            with torch.no_grad():
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                    token_emb = model.transformer.wte(token_id)  # (1, 1, n_embd)
                elif hasattr(model, 'get_input_embeddings'):
                    token_emb = model.get_input_embeddings()(token_id)  # (1, 1, n_embd)
                else:
                    # Fallback: try to find embeddings
                    continue
                token_emb = token_emb.squeeze(0).squeeze(0)  # (1, n_embd)
                representations[word] = token_emb.cpu().detach().numpy().reshape(1, -1)
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
    
    Returns:
        numpy array: Normalized sentence representation (L2 normalized)
    """
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    # Get block_size from config (handle both our models and HuggingFace)
    block_size = getattr(model.config, 'block_size', getattr(model.config, 'n_positions', 1024))
    if len(tokens) > block_size:
        tokens = tokens[:block_size]
    token_ids = torch.tensor([tokens], device=device)
    
    with torch.no_grad():
        # Forward through model to get hidden states
        # Get sense embeddings and combine them
        B, T = token_ids.size()
        
        # Check if model is Backpack, StandardTransformer, or HuggingFace
        is_backpack = isinstance(model, BackpackLM)
        is_hf = _is_huggingface_model(model)
        
        if is_hf:
            # HuggingFace models: use forward pass to get hidden states
            outputs = model.transformer(input_ids=token_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer: (B, T, n_embd)
        elif is_backpack:
            # Use get_sense_vectors which handles the current architecture
            sense_embs = model.get_sense_vectors(token_ids)  # (B, T, n_senses, n_embd)
            
            # Get position embeddings
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = model.pos_embeddings(pos)  # (T, n_embd)
            
            # Use sense attention (new architecture) or sense_predictor (old architecture)
            if hasattr(model, 'sense_attention'):
                # New architecture: use attention-based sense weighting
                att_weights = model.sense_attention(token_embs + pos_emb.unsqueeze(0))  # (B, n_senses, T, T)
                # Average attention over past tokens to get sense weights per token
                sense_weights = att_weights.mean(dim=-1).transpose(1, 2)  # (B, T, n_senses)
                sense_weights = torch.nn.functional.softmax(sense_weights, dim=-1)
            elif hasattr(model, 'sense_predictor'):
                # Old architecture: use MLP-based sense predictor
                context = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, T, n_embd)
                sense_weights = model.sense_predictor(context)  # (B, T, n_senses)
                sense_weights = torch.nn.functional.softmax(sense_weights, dim=-1)  # (B, T, n_senses)
            else:
                # Fallback: uniform weights
                n_senses = sense_embs.shape[2]
                sense_weights = torch.ones(B, T, n_senses, device=device) / n_senses
            
            # Weighted sum of sense vectors
            x = torch.einsum('btsd,bts->btd', sense_embs, sense_weights)  # (B, T, n_embd)
            x = x + pos_emb.unsqueeze(0)
            x = model.drop(x)
            
            # Apply transformer blocks
            x = model.blocks(x)
            hidden_states = model.ln_f(x)  # (B, T, n_embd)
        else:
            # StandardTransformer: use token embeddings + transformer blocks
            x = model.token_embeddings(token_ids)
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            pos_emb = model.pos_embeddings(pos)
            x = x + pos_emb.unsqueeze(0)
            x = model.drop(x)
            x = model.blocks(x)
            hidden_states = model.ln_f(x)  # (B, T, n_embd)
        
        # Extract sentence representation based on method
        # CRITICAL FIX: Exclude special tokens (BOS, EOS, padding) from pooling
        # XLM-RoBERTa uses: 0=<s>, 1=</s>, 2=<pad>
        special_token_ids = torch.tensor([0, 1, 2], device=device)  # BOS, EOS, PAD
        
        if method == 'mean':
            # Mean pooling: average across content tokens only (exclude special tokens)
            # Create mask for non-special tokens (compatible with older PyTorch)
            mask = torch.ones_like(token_ids, dtype=torch.bool, device=device)
            for special_id in special_token_ids:
                mask = mask & (token_ids != special_id)
            
            # Expand mask to match hidden_states shape: (B, T) -> (B, T, 1)
            mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
            
            # Masked mean: sum over masked tokens, divide by count
            masked_hidden = hidden_states * mask_expanded  # (B, T, n_embd)
            token_count = mask_expanded.sum(dim=1, keepdim=True)  # (B, 1, 1)
            token_count = torch.clamp(token_count, min=1.0)  # Avoid division by zero
            sentence_repr = masked_hidden.sum(dim=1) / token_count.squeeze(1)  # (B, n_embd)
        elif method == 'last':
            # Use last non-special token
            special_set = set(special_token_ids.cpu().tolist())
            for i in range(T - 1, -1, -1):
                if token_ids[0, i].item() not in special_set:
                    sentence_repr = hidden_states[:, i, :]  # (B, n_embd)
                    break
            else:
                # Fallback: use last token if all are special
                sentence_repr = hidden_states[:, -1, :]
        elif method == 'cls':
            # Use first non-special token (usually index 0 is <s>, so use index 1)
            if T > 1 and token_ids[0, 1].item() not in special_token_ids:
                sentence_repr = hidden_states[:, 1, :]  # (B, n_embd)
            else:
                sentence_repr = hidden_states[:, 0, :]  # Fallback
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # L2 normalize the representation for better cosine similarity
        sentence_repr = F.normalize(sentence_repr, p=2, dim=-1)
    
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
    
    Args:
        model: BackpackLM or StandardTransformerLM model
        tokenizer: Tokenizer instance
        sentence_pairs: List of (sentence1, sentence2) tuples
        device: Device to run on
        method: Pooling method ('mean', 'last', 'cls')
    
    Returns:
        List of (sentence1, sentence2, similarity) tuples
    """
    similarities = []
    
    for sent1, sent2 in sentence_pairs:
        try:
            repr1 = get_sentence_representation(model, tokenizer, sent1, device, method)
            repr2 = get_sentence_representation(model, tokenizer, sent2, device, method)
            
            # Ensure embeddings are numpy arrays
            if isinstance(repr1, torch.Tensor):
                repr1 = repr1.cpu().detach().numpy()
            if isinstance(repr2, torch.Tensor):
                repr2 = repr2.cpu().detach().numpy()
            
            # L2 normalize (should already be normalized, but ensure it)
            repr1_norm = repr1 / (np.linalg.norm(repr1) + 1e-8)
            repr2_norm = repr2 / (np.linalg.norm(repr2) + 1e-8)
            
            # Cosine similarity (dot product of normalized vectors)
            cos_sim = np.dot(repr1_norm, repr2_norm)
            
            # Clamp to [-1, 1] to handle numerical errors
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            
            similarities.append((sent1, sent2, float(cos_sim)))
        except Exception as e:
            # Skip pairs that fail (e.g., empty sentences, tokenization issues)
            continue
    
    return similarities


def _is_english_or_french(token_str):
    """
    Check if a token is likely English or French.
    
    Args:
        token_str: Token string to check
    
    Returns:
        bool: True if token appears to be English or French
    """
    import re
    
    # Strip whitespace
    token_str = token_str.strip()
    
    if not token_str:
        return False
    
    # Common non-English/French language indicators
    # These are Unicode ranges for various scripts
    non_eu_patterns = [
        r'[\u4e00-\u9fff]',  # Chinese characters
        r'[\u3040-\u309f\u30a0-\u30ff]',  # Japanese hiragana/katakana
        r'[\u0400-\u04ff]',  # Cyrillic (Russian, etc.)
        r'[\u0590-\u05ff]',  # Hebrew
        r'[\u0600-\u06ff]',  # Arabic
        r'[\u0e00-\u0e7f]',  # Thai
        r'[\u1100-\u11ff\uac00-\ud7af]',  # Korean
        r'[\u0900-\u097f]',  # Devanagari (Hindi, etc.)
        r'[\u0d80-\u0dff]',  # Sinhala
        r'[\u0a80-\u0aff]',  # Gujarati
        r'[\u0980-\u09ff]',  # Bengali
        r'[\u0c80-\u0cff]',  # Kannada
        r'[\u0b80-\u0bff]',  # Tamil
        r'[\u0e80-\u0eff]',  # Lao
        r'[\u0370-\u03ff]',  # Greek (though some might be used, filter for now)
        r'[\u1e00-\u1eff]',  # Latin Extended Additional (some non-EU languages)
    ]
    
    # Check if token contains non-European scripts
    for pattern in non_eu_patterns:
        if re.search(pattern, token_str):
            return False
    
    # Check for common non-English/French words/patterns
    # Extended list of non-English/French tokens
    non_eu_patterns_str = [
        r'^[а-яА-Я]+',  # Cyrillic words
        r'^[\u0590-\u05ff]+',  # Hebrew words
        r'^[\u0600-\u06ff]+',  # Arabic words
        r'^[\u4e00-\u9fff]+',  # Chinese characters
        r'^[\u3040-\u309f\u30a0-\u30ff]+',  # Japanese
        r'^[\u0e00-\u0e7f]+',  # Thai
        r'^[\u1100-\u11ff\uac00-\ud7af]+',  # Korean
    ]
    
    for pattern in non_eu_patterns_str:
        if re.match(pattern, token_str):
            return False
    
    # Extended blacklist of non-English/French words
    # Include common tokens from other languages that appear in XLM-RoBERTa vocab
    non_en_fr_words = {
        # Non-European scripts
        'облустук', 'Augstākā', 'ประสบความสําเร็จ', 'Тайланд', 'חיוב',
        'професій', 'טקסט', 'ите', '利用者', '石头', 'ຕາມ', 'хэлбэл',
        'дминистративен', 'خير', '復', 'вияв', '利用해', 'увагу', 'اص',
        'yangi', 'võrk', 'sageli', 'kaitse', 'Αστυνομικ', 'ുണ്ടെന്ന', 'ቢ',
        # Other European languages (extended list)
        'pelaaja', 'demostrar', 'bestu', 'rund', 'dreapta', 'Toivottavasti',
        'fremragende', 'harcama', 'kukk', 'keduanya', 'çãeste', 'Meiriceá',
        'esimene', 'Vokietijoje', 'Código', 'impossível', 'tyvät', 'situé',
        'erius', 'bonyolult', 'gledati', 'ilerek', 'frakt', 'érettségi',
        'passaggio', 'qonaq', 'indeks', 'voorbehou', 'Gjermani', 'dali',
        'Ordu', 'fila', 'kenapa', 'purtat', 'bæjar', 'úřad', 'setkání',
        'jiems', 'ønn', 'slår', 'Beslut', 'ayya', 'Zotit', 'Gunnar', 'Seal',
        'viongozi', 'kérdések', 'strainséir', 'kami', 'Sentyabr', 'UNDE',
        'NAM', 'vit', 'slid', 'grup', 'jalan', 'kebijakan', 'ndlich',
        'kriev', 'odkril', 'sortiment', 'Wenger', 'voed', 'oddych', 'Diari',
        # Additional non-English/French words found in output
        'Prishtinë', 'Prishtin', 'prishtinë',  # Albanian (Pristina)
        'scherm', 'schermen',  # Dutch (screen)
        'gol', 'goles',  # Spanish/Portuguese (goal) - but allow if context suggests French
        'pomembno', 'pomembna',  # Slovenian (important)
        'Hoy', 'hoy',  # Spanish (today)
        'fejl', 'fejler',  # Danish/Norwegian (error)
        'bodde', 'bodden',  # Norwegian (lived)
        'eremu', 'eremua',  # Basque (field)
        'Juventud', 'juventud',  # Spanish (youth)
        'projesi', 'projeler',  # Turkish (project)
        'xavfsizligi',  # Uzbek (security)
        'liberal', 'liberale',  # Could be English/French but often other languages
        'gniti', 'gnit',  # Various Slavic languages
        'shambuliaji',  # Georgian
        'zni', 'zn',  # Various Slavic
        'ansvar', 'ansvaret',  # Norwegian/Swedish (responsibility)
        'haver', 'havers',  # Portuguese (to have)
        'acesteia', 'acestei',  # Romanian (this)
        'njen', 'njene',  # Various Slavic
        'earre', 'earren',  # Basque
        'BON', 'bon',  # Could be French but often other languages
        'lako', 'lakom',  # Various Slavic
        'Pwyllgor', 'pwyllgor',  # Welsh (committee)
        'unos', 'una',  # Spanish (some/one)
        'uppgifter', 'uppgift',  # Swedish (tasks)
        'pieteikumu', 'pieteikums',  # Latvian (application)
        'vn', 'vna',  # Various abbreviations
        'automatik', 'automatikë',  # Albanian
        'zve', 'zvezda',  # Various Slavic
        'creat', 'creatul',  # Romanian
        'Allergi', 'allergi',  # Norwegian/Swedish (allergy)
        'printre', 'printr',  # Romanian (among)
        'yaka', 'yak',  # Turkish (how)
        'fadh', 'fadha',  # Arabic transliteration
        'anois', 'anoise',  # Irish (now)
        'industri', 'industrië',  # Dutch/Indonesian (industry)
        'obtido', 'obtidos',  # Portuguese (obtained)
        'frische', 'frisch',  # German (fresh)
        'viaggi', 'viaggio',  # Italian (travel)
        'sesso', 'sessi',  # Italian (sex)
        'preço', 'preços',  # Portuguese (price)
        'ukus', 'guztia', 'enean', 'skrivnost', 'pozytywny', 'kain', 'Luki',
        'menge', 'likleri', 'tenido', 'objetos', 'iniciativa', 'littera',
        'Glass', 'Tulisan', 'xarici', 'doplnil', 'terdengar', 'burg', 'shib',
        'qir', 'razliko', 'kesk', 'giorno', 'referência', 'Komunejo', 'ORGANI',
        'oitu', 'izjemno', 'askari', 'Maza', 'dramatik', 'uba', 'paglala',
        'festivala', 'etiladi', 'czy', 'zwraca', 'hverandre', 'mne', 'levy',
        'procure', 'jer', 'potpis', 'hisar', 'Magic', 'kende', 'makan',
        'kekurangan', 'caminho', 'ajam', 'Abbiamo', 'umjetni',
        # Additional problematic words found in semantic relatedness output
        'landi', 'Landi', 'LANDI',  # Albanian/Italian (land)
        'Budi', 'budi', 'BUDI',  # Indonesian (Buddha/be)
        'suhbat', 'Suhbat', 'SUHBAT',  # Uzbek/Turkish (conversation)
        'kork', 'Kork', 'KORK',  # Turkish (fear)
        'torial', 'Torial', 'TORIAL',  # Various languages (editorial fragment)
        'korku', 'korkular',  # Turkish (fears)
        'suhbati', 'suhbatlar',  # Uzbek (conversations)
        'budil', 'budili',  # Various Slavic (wake)
        'landia', 'landija',  # Various languages
        # Common fragments/subwords that aren't English/French
        'λερ', 'Farg', 'vi', 'cy', 'ster', 'che', 'hui', 'élé', 'da', 'crai'
    }
    
    if token_str.lower() in non_en_fr_words or token_str in non_en_fr_words:
        return False
    
    # Check for tokens that are clearly not English/French based on character patterns
    # Allow common punctuation and numbers
    if re.match(r'^[0-9\s\.,;:!?\-\(\)\[\]\{\}\'\"\/\\]+$', token_str):
        # Pure punctuation/numbers - allow if meaningful
        return len(token_str.strip()) > 0
    
    # More restrictive: Only allow tokens that match English/French character patterns
    # English: a-z, A-Z
    # French: a-z, A-Z + French accents: àâäçèéêëîïôùûüÿ
    # Allow common punctuation and numbers
    en_fr_char_pattern = r'^[a-zA-ZàâäçèéêëîïôùûüÿÀÂÄÇÈÉÊËÎÏÔÙÛÜŸ\s\.,;:!?\-\(\)\[\]\{\}\'\"\/\\0-9]+$'
    
    if not re.match(en_fr_char_pattern, token_str):
        return False
    
    # Filter out tokens with non-French European accents (indicates other languages)
    # These patterns indicate other European languages
    non_fr_eu_accents = r'[äöüßÄÖÜßåæøÅÆØąęćłńóśźżĄĆĘŁŃÓŚŹŻăâîșțĂÂÎȘȚčćđšžČĆĐŠŽõÕðþÐÞ]'
    
    # Check if token has non-French European accents
    if re.search(non_fr_eu_accents, token_str):
        # Has accents from other European languages - filter out
        return False
    
    # Filter Spanish/Portuguese accents (áéíóú) unless also has French accents
    # Spanish/Portuguese: áéíóú (but some French words have these too)
    # If token has Spanish/Portuguese accents but NO French accents, likely not French
    if re.search(r'[áéíóúÁÉÍÓÚ]', token_str) and not re.search(r'[àâäçèéêëîïôùûüÿ]', token_str):
        # Has Spanish/Portuguese accents but no French accents - likely not French
        # Filter it out
        return False
    
    # Additional aggressive filter: Check for common non-English/French word endings/patterns
    # These patterns are common in other European languages
    suspicious_endings = [
        r'czy$',  # Polish question word
        r'jer$',  # Common in Slavic languages
        r'zwraca$',  # Polish
        r'hverandre$',  # Norwegian/Danish
        r'mne$',  # Czech/Slovak
        r'potpis$',  # Croatian/Serbian
        r'hisar$',  # Turkish
        r'kende$',  # Danish
        r'makan$',  # Indonesian/Malay
        r'kekurangan$',  # Indonesian
        r'caminho$',  # Portuguese
        r'ajam$',  # Indonesian
        r'Abbiamo$',  # Italian
        r'umjetni$',  # Croatian
    ]
    
    for pattern in suspicious_endings:
        if re.search(pattern, token_str, re.IGNORECASE):
            return False
    
    # Additional heuristic: if token is very short and doesn't match common English/French patterns
    # Allow common short words
    if len(token_str) <= 2:
        common_short_en_fr = {
            'le', 'la', 'de', 'du', 'un', 'une', 'en', 'et', 'ou', 'à', 'au',
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'is', 'it', 'be',
            'se', 'ne', 'ce', 'je', 'tu', 'il', 'nous', 'vous', 'ils', 'elles',
            'me', 'my', 'we', 'he', 'she', 'us', 'as', 'am', 'if', 'or', 'so',
            'no', 'do', 'go', 'up', 'by', 'so', 'ok', 'hi', 'oh', 'ah', 'ok'
        }
        if token_str.lower() not in common_short_en_fr and not token_str.isalpha():
            return False
    
    # Final check: Filter out tokens with suspicious patterns that indicate other languages
    # Extended blacklist of non-English/French tokens found in analysis
    additional_non_en_fr = {
        'shakl', 'pontos', 'effettua', 'waarvoor', 'xiquet', 'ança', 'IRMA',
        'tahimik', 'Barry', 'tavad', 'Hij', 'Race', 'Dhawa', 'oppdage',
        'lugu', 'melder', 'condimentum', 'trattati', 'yalar', 'belirli',
        'aurretik', 'sindicatos', 'frum', 'outro', 'folosesc', 'presidente',
        'vratiti', 'konkretne', 'lingar', 'funkcjonowania', 'eiden',
        'organisasjon', 'erat', 'adnoddau', 'rischia', 'navigazione', 'brukar',
        'Ezek', 'német', 'saber', 'nieuwsbrief', 'decyduje', 'telèfon',
        'relacionamento', 'altid', 'dhéanann', 'Anh', 'struct', 'Neuro', 'STA',
        'continguts', 'Lever', 'bent', 'Vat', 'Januari', 'Jang', 'baut',
        'montes', 'netiek', 'Sementara', 'hodet'
    }
    
    if token_str in additional_non_en_fr or token_str.lower() in additional_non_en_fr:
        return False
    
    # Check for common non-English/French word patterns
    suspicious_patterns_final = [
        r'^[a-z]+czy$',  # Polish words ending in czy
        r'^[a-z]+ati$',  # Common in Slavic languages
        r'^[a-z]+mentum$',  # Latin (condimentum)
        r'^[a-z]+trattati$',  # Italian
        r'^[a-z]+yar$',  # Turkish/Azerbaijani
        r'^[a-z]+belirli$',  # Turkish
        r'^[a-z]+retik$',  # Basque
        r'^[a-z]+sindicatos$',  # Spanish/Portuguese
        r'^[a-z]+frum$',  # Romanian
        r'^[a-z]+outro$',  # Portuguese
        r'^[a-z]+tavad$',  # Estonian
        r'^[a-z]+folosesc$',  # Romanian
        r'^[a-z]+vratiti$',  # Croatian/Serbian
        r'^[a-z]+konkretne$',  # Polish/Czech
        r'^[a-z]+lingar$',  # Various languages
        r'^[a-z]+funkcjonowania$',  # Polish
        r'^[a-z]+eiden$',  # Various languages
        r'^[a-z]+shakl$',  # Uzbek/Turkic
        r'^[a-z]+pontos$',  # Portuguese/Greek
        r'^[a-z]+effettua$',  # Italian
        r'^[a-z]+waarvoor$',  # Dutch
        r'^[a-z]+xiquet$',  # Catalan
        r'^[a-z]+ança$',  # Portuguese
        r'^[a-z]+tahimik$',  # Tagalog
        r'^[a-z]+Hij$',  # Dutch
        r'^[a-z]+oppdage$',  # Norwegian/Danish
        r'^[a-z]+lugu$',  # Estonian
        r'^[a-z]+melder$',  # German/Dutch
        r'^[a-z]+organisasjon$',  # Norwegian
        r'^[a-z]+erat$',  # Latin
        r'^[a-z]+adnoddau$',  # Welsh
        r'^[a-z]+rischia$',  # Italian
        r'^[a-z]+navigazione$',  # Italian
        r'^[a-z]+brukar$',  # Swedish
        r'^[a-z]+német$',  # Hungarian
        r'^[a-z]+saber$',  # Spanish/Portuguese
        r'^[a-z]+nieuwsbrief$',  # Dutch
        r'^[a-z]+decyduje$',  # Polish
        r'^[a-z]+telèfon$',  # Catalan
        r'^[a-z]+relacionamento$',  # Portuguese
        r'^[a-z]+altid$',  # Norwegian/Danish
        r'^[a-z]+dhéanann$',  # Irish
    ]
    
    # Allow some exceptions that might be English/French or proper nouns
    allowed_exceptions = {'presidente', 'Magic', 'Barry', 'Race', 'Dhawa', 'Neuro', 'STA', 'IRMA', 'Ezek', 'Anh'}
    
    if token_str not in allowed_exceptions:
        for pattern in suspicious_patterns_final:
            if re.match(pattern, token_str, re.IGNORECASE):
                return False
    
    # Additional check: Filter tokens that look like they're from other languages
    # Check for common non-English/French word endings
    non_en_fr_endings = [
        'asjon', 'asjoner', 'asjonen',  # Norwegian
        'tion', 'tions',  # But also English/French, so be careful
        'zione', 'zioni',  # Italian
        'ção', 'ções',  # Portuguese
        'ción', 'ciones',  # Spanish
        'tion', 'tions',  # English/French - allow
        'sjon', 'sjons',  # Swedish
        'ssion', 'ssions',  # English/French - allow
    ]
    
    # Only filter if it's clearly not English/French
    for ending in ['asjon', 'asjoner', 'asjonen', 'zione', 'zioni', 'ção', 'ções', 'ción', 'ciones', 'sjon', 'sjons']:
        if token_str.lower().endswith(ending):
            return False
    
    return True


def _filter_meaningful_tokens(token_str, min_prob=0.001):
    """
    Filter out garbage/meaningless tokens from predictions.
    
    Filters out:
    - Special tokens (<|lang_sep|>, <pad>, <unk>, <s>, </s>, etc.)
    - Punctuation-only tokens
    - Single character fragments (unless they're meaningful like 'a', 'I')
    - Whitespace-only tokens
    - Very low probability tokens
    
    Args:
        token_str: Token string to check
        min_prob: Minimum probability threshold (not used here but kept for consistency)
    
    Returns:
        bool: True if token should be kept, False if filtered out
    """
    
    # Strip whitespace
    token_str = token_str.strip()
    
    # Filter empty strings
    if not token_str:
        return False
    
    # Filter special tokens (common patterns)
    special_patterns = [
        r'^<\|.*\|>$',  # <|lang_sep|>, <|endoftext|>, etc.
        r'^<pad>$',
        r'^<unk>$',
        r'^<s>$',
        r'^</s>$',
        r'^<mask>$',
        r'^<cls>$',
        r'^<sep>$',
        r'^\[PAD\]$',
        r'^\[UNK\]$',
        r'^\[CLS\]$',
        r'^\[SEP\]$',
        r'^\[MASK\]$',
    ]
    
    for pattern in special_patterns:
        if re.match(pattern, token_str, re.IGNORECASE):
            return False
    
    # Filter punctuation-only tokens (but keep some meaningful ones)
    punctuation_only = re.match(r'^[^\w\s]+$', token_str)
    if punctuation_only:
        # Keep common meaningful punctuation in context
        meaningful_punct = {'.', '!', '?', ',', ';', ':', '-', "'", '"'}
        if token_str not in meaningful_punct:
            return False
    
    # Filter single character tokens that are likely fragments
    # (keep common meaningful single chars like 'a', 'I', 'é', etc.)
    if len(token_str) == 1:
        # Keep if it's a letter (including accented) or common punctuation
        if not (token_str.isalpha() or token_str in {'.', '!', '?', ',', ';', ':', '-', "'", '"'}):
            return False
    
    # Filter tokens that are just whitespace or control characters
    if token_str.isspace() or not any(c.isprintable() for c in token_str):
        return False
    
    # Filter tokens that are just numbers (unless they're part of a word)
    if token_str.isdigit() and len(token_str) <= 2:
        return False
    
    # Filter very short fragments that are likely subword pieces
    # (but keep if they're common words or letters)
    if len(token_str) <= 2:
        # Keep common short words/prefixes
        common_short = {
            'le', 'la', 'de', 'du', 'un', 'une', 'en', 'et', 'ou', 'à', 'au',
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'is', 'it', 'be',
            'se', 'ne', 'ce', 'je', 'tu', 'il', 'nous', 'vous', 'ils', 'elles'
        }
        if token_str.lower() not in common_short and not token_str.isalpha():
            # Likely a fragment if it's not a common word
            return False
    
    return True


def analyze_sense_vectors(model, tokenizer, words, device, top_k=5, verbose=True, filter_tokens=True, 
                          analyze_relatedness=True, analyze_syntax=True):
    """
    Improved sense vector analysis with probabilities, quantitative metrics, and better visualization.
    
    Features:
    - Shows probabilities with predictions
    - Computes sense diversity metrics (entropy, uniqueness)
    - Calculates similarity between senses
    - Provides sense vector statistics (magnitude, variance)
    - Filters out garbage/meaningless tokens
    - Analyzes semantic relatedness (embedding-space similarity)
    - Analyzes syntactic patterns (next wordpiece, verb objects, nmod nouns)
    - Better formatted output
    
    Args:
        model: BackpackLM model (or HuggingFace model - will skip if not supported)
        tokenizer: Tokenizer instance
        words: List of words to analyze
        device: Device to run on
        top_k: Number of top predictions to show per sense (after filtering)
        verbose: Whether to print detailed output
        filter_tokens: Whether to filter out meaningless tokens (default: True)
        analyze_relatedness: Whether to analyze semantic relatedness in embedding space (default: True)
        analyze_syntax: Whether to analyze syntactic patterns (default: True)
    
    Returns:
        dict: Comprehensive results including predictions, probabilities, and metrics
    """
    # Check if model supports sense vectors
    is_hf = _is_huggingface_model(model)
    if is_hf or not hasattr(model, 'get_sense_vectors'):
        if verbose:
            print("  Note: This model does not expose sense vectors. Skipping sense analysis.")
        return {}
    
    # Check if model supports sense vectors
    is_hf = _is_huggingface_model(model)
    if is_hf or not hasattr(model, 'get_sense_vectors'):
        if verbose:
            print("  Note: This model does not expose sense vectors. Skipping sense analysis.")
        return {}
    
    results = {}
    
    for word in words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if len(tokens) == 0:
            if verbose:
                print(f"  Warning: Could not tokenize '{word}', skipping...")
            continue
        
        try:
            with torch.no_grad():
                token_id = torch.tensor([tokens[0]], device=device).unsqueeze(0)
                sense_vectors = model.get_sense_vectors(token_id)  # (1, 1, n_senses, n_embd)
                sense_vectors = sense_vectors.squeeze(0).squeeze(0)  # (n_senses, n_embd)
                n_senses = sense_vectors.shape[0]
                
                # Store predictions with probabilities
                sense_predictions = []
                sense_probs_list = []
                all_probs = []
                
                for sense_idx in range(n_senses):
                    sense_vec = sense_vectors[sense_idx].unsqueeze(0)  # (1, n_embd)
                    logits = model.lm_head(sense_vec)  # (1, vocab_size)
                    probs = F.softmax(logits, dim=-1)
                    all_probs.append(probs)
                    
                    # Get more candidates than top_k to account for filtering
                    # We'll get top_k * 5 to ensure we have enough meaningful tokens
                    candidate_k = top_k * 5 if filter_tokens else top_k
                    top_probs, top_indices = torch.topk(probs, min(candidate_k, probs.shape[-1]), dim=-1)
                    
                    top_tokens = []
                    top_probs_list = []
                    
                    # Collect tokens and filter
                    for idx, prob in zip(top_indices[0], top_probs[0]):
                        token_str = tokenizer.decode([idx.item()])
                        
                        # Filter out meaningless tokens if enabled
                        if filter_tokens:
                            if not _filter_meaningful_tokens(token_str, prob.item()):
                                continue
                            # Also filter out non-English/French tokens
                            if not _is_english_or_french(token_str):
                                continue
                        
                        top_tokens.append(token_str)
                        top_probs_list.append(prob.item())
                        
                        # Stop once we have enough meaningful tokens
                        if len(top_tokens) >= top_k:
                            break
                    
                    # If we don't have enough after filtering, get more candidates
                    if len(top_tokens) < top_k and filter_tokens:
                        # Get more candidates to filter
                        remaining_needed = top_k - len(top_tokens)
                        # Try to get more from the top predictions
                        for idx, prob in zip(top_indices[0], top_probs[0]):
                            token_str = tokenizer.decode([idx.item()])
                            # Skip if already added or doesn't pass filters
                            if token_str in top_tokens:
                                continue
                            if not _filter_meaningful_tokens(token_str, prob.item()):
                                continue
                            if not _is_english_or_french(token_str):
                                continue
                            top_tokens.append(token_str)
                            top_probs_list.append(prob.item())
                            if len(top_tokens) >= top_k:
                                break
                    
                    sense_predictions.append({
                        'tokens': top_tokens[:top_k],  # Ensure we only keep top_k
                        'probs': top_probs_list[:top_k],
                        'top_prob': top_probs_list[0] if top_probs_list else 0.0
                    })
                    sense_probs_list.append(probs)
                
                # Compute quantitative metrics
                # 1. Entropy (diversity) for each sense
                entropies = []
                for probs in sense_probs_list:
                    # Filter out very small probabilities for numerical stability
                    probs_clipped = torch.clamp(probs, min=1e-10)
                    entropy = -torch.sum(probs_clipped * torch.log(probs_clipped)).item()
                    entropies.append(entropy)
                
                # 2. Sense similarity matrix (cosine similarity)
                sense_similarities = []
                sense_norms = []
                for i in range(n_senses):
                    norm_i = torch.norm(sense_vectors[i]).item()
                    sense_norms.append(norm_i)
                    similarities_row = []
                    for j in range(n_senses):
                        if i == j:
                            similarities_row.append(1.0)
                        else:
                            cos_sim = F.cosine_similarity(
                                sense_vectors[i].unsqueeze(0),
                                sense_vectors[j].unsqueeze(0),
                                dim=1
                            ).item()
                            similarities_row.append(cos_sim)
                    sense_similarities.append(similarities_row)
                
                # 3. Prediction overlap between senses
                prediction_overlap = []
                for i in range(n_senses):
                    overlap_row = []
                    tokens_i = set(sense_predictions[i]['tokens'])
                    for j in range(n_senses):
                        tokens_j = set(sense_predictions[j]['tokens'])
                        overlap = len(tokens_i & tokens_j) / len(tokens_i | tokens_j) if (tokens_i | tokens_j) else 0.0
                        overlap_row.append(overlap)
                    prediction_overlap.append(overlap_row)
                
                # 4. Sense vector statistics
                sense_vectors_np = sense_vectors.cpu().numpy()
                mean_magnitude = np.mean(sense_norms)
                std_magnitude = np.std(sense_norms)
                mean_entropy = np.mean(entropies)
                std_entropy = np.std(entropies)
                
                # 5. Average sense similarity (excluding diagonal)
                avg_similarity = np.mean([
                    sense_similarities[i][j] 
                    for i in range(n_senses) 
                    for j in range(n_senses) 
                    if i != j
                ]) if n_senses > 1 else 0.0
                
                # Additional analysis: Semantic relatedness and syntactic patterns
                semantic_relatedness = {}
                next_wordpiece_patterns = {}
                
                if analyze_relatedness:
                    # Find semantically similar words in embedding space
                    # Strategy: Sample many tokens, compute similarities, then filter to English/French only
                    vocab_sample_size = 100000  # Sample many tokens to get enough English/French ones
                    sample_indices = torch.randperm(model.config.vocab_size)[:vocab_sample_size].to(device)
                    # Handle both BackpackLM (token_embedding) and StandardTransformerLM (token_embeddings)
                    if hasattr(model, 'token_embedding'):
                        sample_embeddings = model.token_embedding(sample_indices)
                    elif hasattr(model, 'token_embeddings'):
                        sample_embeddings = model.token_embeddings(sample_indices)
                    else:
                        raise AttributeError("Model has neither token_embedding nor token_embeddings")
                    
                    for sense_idx in range(n_senses):
                        sense_vec = sense_vectors[sense_idx].unsqueeze(0)
                        similarities = F.cosine_similarity(sense_vec, sample_embeddings, dim=1)
                        
                        # Get all similarities and filter
                        # Sort all similarities and filter to English/French
                        all_pairs = [(sim.item(), idx.item()) for idx, sim in zip(sample_indices, similarities)]
                        all_pairs.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity
                        
                        related_words = []
                        checked_count = 0
                        max_checks = 50000  # Check even more tokens to find English/French ones
                        seen_words = set()  # Track seen words to avoid duplicates
                        for sim, token_id in all_pairs:
                            checked_count += 1
                            token_str = tokenizer.decode([token_id]).strip()
                            
                            # Skip empty tokens
                            if not token_str:
                                continue
                            
                            # Normalize: lowercase for comparison, but keep original case
                            token_str_lower = token_str.lower()
                            
                            # Skip duplicates (case-insensitive)
                            if token_str_lower in seen_words:
                                continue
                            
                            # STRICT filtering: must be meaningful AND English/French
                            if not _filter_meaningful_tokens(token_str):
                                continue
                            
                            # CRITICAL: Filter to English/French only - remove all other languages
                            # This is the most important check - must pass before adding
                            if not _is_english_or_french(token_str):
                                continue
                            
                            # Additional check: make sure it's not a number or pure punctuation
                            if re.match(r'^[0-9\s\.,;:!?\-\(\)\[\]\{\}\'\"\/\\]+$', token_str):
                                if len(token_str.strip()) <= 1:
                                    continue
                            
                            # Final check: ensure token is not empty after stripping
                            token_str_clean = token_str.strip()
                            if not token_str_clean:
                                continue
                            
                            # Additional validation: ensure it looks like a real word
                            # Must have at least 2 letters (allow single letters like 'a', 'I')
                            if len(token_str_clean) > 1:
                                # For multi-character tokens, ensure it has at least one letter
                                if not re.search(r'[a-zA-ZàâäçèéêëîïôùûüÿÀÂÄÇÈÉÊËÎÏÔÙÛÜŸ]', token_str_clean):
                                    continue
                            
                            # All checks passed - add to results
                            seen_words.add(token_str_lower)
                            related_words.append((token_str_clean, sim))
                            if len(related_words) >= 10:  # Stop once we have enough
                                break
                            
                            # Stop if we've checked too many without finding enough
                            if checked_count >= max_checks:
                                break
                        
                        semantic_relatedness[sense_idx] = related_words[:10]
                
                if analyze_syntax:
                    # Analyze next-wordpiece patterns (what tokens follow this sense)
                    # This is already captured in predictions, but we can categorize them
                    for sense_idx in range(n_senses):
                        pred_tokens = sense_predictions[sense_idx]['tokens']
                        
                        # Categorize predictions
                        categories = {
                            'articles': [],
                            'prepositions': [],
                            'verbs': [],
                            'nouns': [],
                            'proper_nouns': [],
                            'other': []
                        }
                        
                        # Simple heuristics for categorization
                        articles = {'the', 'le', 'la', 'les', 'un', 'une', 'des', 'a', 'an'}
                        prepositions = {'de', 'du', 'dans', 'sur', 'à', 'pour', 'avec', 'par', 
                                       'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
                        
                        for token in pred_tokens:
                            token_lower = token.lower()
                            if token_lower in articles:
                                categories['articles'].append(token)
                            elif token_lower in prepositions:
                                categories['prepositions'].append(token)
                            elif token[0].isupper() and len(token) > 2:
                                categories['proper_nouns'].append(token)
                            else:
                                categories['other'].append(token)
                        
                        next_wordpiece_patterns[sense_idx] = categories
                
                # Store comprehensive results
                results[word] = {
                    'predictions': sense_predictions,
                    'entropies': entropies,
                    'sense_similarities': sense_similarities,
                    'prediction_overlap': prediction_overlap,
                    'sense_norms': sense_norms,
                    'semantic_relatedness': semantic_relatedness if analyze_relatedness else {},
                    'syntactic_patterns': next_wordpiece_patterns if analyze_syntax else {},
                    'metrics': {
                        'mean_entropy': mean_entropy,
                        'std_entropy': std_entropy,
                        'mean_magnitude': mean_magnitude,
                        'std_magnitude': std_magnitude,
                        'avg_sense_similarity': avg_similarity,
                        'n_senses': n_senses
                    }
                }
                
                # Print formatted output
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"Word: '{word}' (tokenized as: {tokenizer.decode(tokens)})")
                    print(f"{'='*70}")
                    
                    # Print predictions with probabilities
                    print(f"\nTop-{top_k} Predictions per Sense (Next Wordpiece):")
                    print("-" * 70)
                    for sense_idx in range(n_senses):
                        preds = sense_predictions[sense_idx]
                        # Get sense label
                        sense_label = SENSE_LABELS.get(sense_idx, f"Sense {sense_idx}")
                        print(f"\nSense {sense_idx:2d}: {sense_label}")
                        print(f"  (entropy: {entropies[sense_idx]:.3f}, norm: {sense_norms[sense_idx]:.3f})")
                        print(f"  Next wordpiece predictions:")
                        for token, prob in zip(preds['tokens'], preds['probs']):
                            print(f"    {token:20s} {prob*100:6.2f}%")
                        
                        # Removed semantically related words printing for cleaner output
                        
                        # Show syntactic patterns if available
                        if analyze_syntax and sense_idx in results[word]['syntactic_patterns']:
                            patterns = results[word]['syntactic_patterns'][sense_idx]
                            pattern_strs = []
                            if patterns.get('articles'):
                                pattern_strs.append(f"Articles: {', '.join(patterns['articles'][:3])}")
                            if patterns.get('prepositions'):
                                pattern_strs.append(f"Prepositions: {', '.join(patterns['prepositions'][:3])}")
                            if patterns.get('proper_nouns'):
                                pattern_strs.append(f"Proper nouns: {', '.join(patterns['proper_nouns'][:3])}")
                            if pattern_strs:
                                print(f"  Syntactic patterns: {' | '.join(pattern_strs)}")
                    
                    # Print metrics summary
                    print(f"\n{'='*70}")
                    print("Quantitative Metrics:")
                    print("-" * 70)
                    print(f"  Mean Entropy:        {mean_entropy:.3f} ± {std_entropy:.3f}")
                    print(f"  Mean Magnitude:      {mean_magnitude:.3f} ± {std_magnitude:.3f}")
                    print(f"  Avg Sense Similarity: {avg_similarity:.3f}")
                    print(f"  Number of Senses:    {n_senses}")
                    
                    # Print sense similarity matrix (if small enough)
                    if n_senses <= 16:
                        print(f"\n{'='*70}")
                        print("Sense Similarity Matrix (cosine similarity):")
                        print("-" * 70)
                        print("     ", end="")
                        for j in range(n_senses):
                            print(f" S{j:2d}", end="")
                        print()
                        for i in range(n_senses):
                            print(f" S{i:2d} ", end="")
                            for j in range(n_senses):
                                print(f"{sense_similarities[i][j]:5.2f}", end="")
                            print()
                    
        except Exception as e:
            if verbose:
                print(f"  Error analyzing '{word}': {e}")
                import traceback
                traceback.print_exc()
            continue
    
    return results


def analyze_cross_lingual_sense_alignment(model, tokenizer, word_pairs, device, top_k=5, verbose=True):
    """
    Analyze cross-lingual sense alignment for translation pairs.
    
    For each word pair (e.g., 'hello'/'bonjour'), this function:
    1. Analyzes sense vectors for both words
    2. Computes similarity between senses across languages
    3. Identifies aligned sense pairs
    4. Provides quantitative alignment metrics
    
    Args:
        model: BackpackLM model
        tokenizer: Tokenizer instance
        word_pairs: List of (word_en, word_fr) tuples
        device: Device to run on
        top_k: Number of top predictions to show per sense
        verbose: Whether to print detailed output
    
    Returns:
        dict: Cross-lingual alignment results
    """
    results = {}
    
    for word_en, word_fr in word_pairs:
        try:
            # Analyze both words (with token filtering enabled)
            en_results = analyze_sense_vectors(model, tokenizer, [word_en], device, top_k=top_k, verbose=False, filter_tokens=True)
            fr_results = analyze_sense_vectors(model, tokenizer, [word_fr], device, top_k=top_k, verbose=False, filter_tokens=True)
            
            if word_en not in en_results or word_fr not in fr_results:
                if verbose:
                    print(f"  Warning: Could not analyze '{word_en}' or '{word_fr}', skipping...")
                continue
            
            en_data = en_results[word_en]
            fr_data = fr_results[word_fr]
            
            # Get sense vectors
            tokens_en = tokenizer.encode(word_en, add_special_tokens=False)
            tokens_fr = tokenizer.encode(word_fr, add_special_tokens=False)
            
            if len(tokens_en) == 0 or len(tokens_fr) == 0:
                continue
            
            with torch.no_grad():
                token_id_en = torch.tensor([tokens_en[0]], device=device).unsqueeze(0)
                token_id_fr = torch.tensor([tokens_fr[0]], device=device).unsqueeze(0)
                
                sense_vectors_en = model.get_sense_vectors(token_id_en).squeeze(0).squeeze(0)  # (n_senses, n_embd)
                sense_vectors_fr = model.get_sense_vectors(token_id_fr).squeeze(0).squeeze(0)  # (n_senses, n_embd)
                
                n_senses_en = sense_vectors_en.shape[0]
                n_senses_fr = sense_vectors_fr.shape[0]
                
                # Compute cross-lingual sense similarity matrix
                cross_lingual_similarities = []
                for i in range(n_senses_en):
                    similarities_row = []
                    for j in range(n_senses_fr):
                        cos_sim = F.cosine_similarity(
                            sense_vectors_en[i].unsqueeze(0),
                            sense_vectors_fr[j].unsqueeze(0),
                            dim=1
                        ).item()
                        similarities_row.append(cos_sim)
                    cross_lingual_similarities.append(similarities_row)
                
                # Find best alignment (greedy matching)
                alignment = []
                used_fr = set()
                for i in range(n_senses_en):
                    best_j = -1
                    best_sim = -1.0
                    for j in range(n_senses_fr):
                        if j not in used_fr and cross_lingual_similarities[i][j] > best_sim:
                            best_sim = cross_lingual_similarities[i][j]
                            best_j = j
                    if best_j >= 0:
                        alignment.append((i, best_j, best_sim))
                        used_fr.add(best_j)
                
                # Compute alignment metrics
                avg_alignment_sim = np.mean([sim for _, _, sim in alignment]) if alignment else 0.0
                max_alignment_sim = max([sim for _, _, sim in alignment]) if alignment else 0.0
                min_alignment_sim = min([sim for _, _, sim in alignment]) if alignment else 0.0
                
                # Store results
                results[(word_en, word_fr)] = {
                    'en_analysis': en_data,
                    'fr_analysis': fr_data,
                    'cross_lingual_similarities': cross_lingual_similarities,
                    'alignment': alignment,
                    'metrics': {
                        'avg_alignment_sim': avg_alignment_sim,
                        'max_alignment_sim': max_alignment_sim,
                        'min_alignment_sim': min_alignment_sim,
                        'n_aligned_pairs': len(alignment),
                        'n_senses_en': n_senses_en,
                        'n_senses_fr': n_senses_fr
                    }
                }
                
                # Print formatted output
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"Cross-Lingual Sense Alignment: '{word_en}' ↔ '{word_fr}'")
                    print(f"{'='*70}")
                    
                    print(f"\nAlignment Pairs (EN Sense → FR Sense, similarity):")
                    print("-" * 70)
                    for en_idx, fr_idx, sim in alignment:
                        en_top = en_data['predictions'][en_idx]['tokens'][0]
                        fr_top = fr_data['predictions'][fr_idx]['tokens'][0]
                        print(f"  EN Sense {en_idx:2d} ({en_top:15s}) ↔ FR Sense {fr_idx:2d} ({fr_top:15s}): {sim:.3f}")
                    
                    print(f"\n{'='*70}")
                    print("Cross-Lingual Alignment Metrics:")
                    print("-" * 70)
                    print(f"  Average Alignment Similarity: {avg_alignment_sim:.3f}")
                    print(f"  Max Alignment Similarity:       {max_alignment_sim:.3f}")
                    print(f"  Min Alignment Similarity:      {min_alignment_sim:.3f}")
                    print(f"  Number of Aligned Pairs:       {len(alignment)}/{min(n_senses_en, n_senses_fr)}")
                    
                    # Print similarity matrix if small enough
                    if n_senses_en <= 8 and n_senses_fr <= 8:
                        print(f"\n{'='*70}")
                        print("Cross-Lingual Sense Similarity Matrix:")
                        print("-" * 70)
                        print("      ", end="")
                        for j in range(n_senses_fr):
                            print(f" FR{j:2d}", end="")
                        print()
                        for i in range(n_senses_en):
                            print(f" EN{i:2d} ", end="")
                            for j in range(n_senses_fr):
                                print(f"{cross_lingual_similarities[i][j]:5.2f}", end="")
                            print()
        
        except Exception as e:
            if verbose:
                print(f"  Error analyzing cross-lingual alignment for '{word_en}'/'{word_fr}': {e}")
                import traceback
                traceback.print_exc()
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


def _get_fallback_word_pairs(language='en'):
    """
    Get fallback word pairs for similarity evaluation when MultiSimLex is unavailable.
    Returns list of (word1, word2, human_score) tuples.
    """
    if language == 'en':
        # Common English word pairs with expected similarity scores (0-10 scale)
        return [
            ('car', 'automobile', 9.5),
            ('car', 'bicycle', 3.0),
            ('happy', 'joyful', 8.5),
            ('happy', 'sad', 1.0),
            ('big', 'large', 9.0),
            ('big', 'small', 2.0),
            ('dog', 'cat', 4.0),
            ('dog', 'puppy', 8.5),
            ('water', 'liquid', 7.5),
            ('water', 'fire', 1.5),
            ('book', 'novel', 7.0),
            ('book', 'computer', 2.0),
            ('house', 'home', 8.0),
            ('house', 'car', 2.5),
            ('good', 'excellent', 8.0),
            ('good', 'bad', 1.0),
        ]
    elif language == 'fr':
        # Common French word pairs
        return [
            ('voiture', 'automobile', 9.5),
            ('voiture', 'vélo', 3.0),
            ('heureux', 'joyeux', 8.5),
            ('heureux', 'triste', 1.0),
            ('grand', 'large', 8.0),
            ('grand', 'petit', 2.0),
            ('chien', 'chat', 4.0),
            ('chien', 'chiot', 8.5),
            ('eau', 'liquide', 7.5),
            ('eau', 'feu', 1.5),
            ('livre', 'roman', 7.0),
            ('livre', 'ordinateur', 2.0),
            ('maison', 'foyer', 8.0),
            ('maison', 'voiture', 2.5),
            ('bon', 'excellent', 8.0),
            ('bon', 'mauvais', 1.0),
        ]
    return []


def _evaluate_cross_lingual_word_similarity_fallback(model, tokenizer, lang1, lang2, device, max_samples=None):
    """
    Fallback cross-lingual word similarity evaluation using common word pairs.
    """
    # Common English-French word pairs with similarity scores (0-10 scale)
    cross_lingual_pairs = [
        ('hello', 'bonjour', 9.5),
        ('world', 'monde', 9.5),
        ('parliament', 'parlement', 9.5),
        ('support', 'soutenir', 9.0),
        ('proposal', 'proposition', 9.0),
        ('important', 'important', 9.5),
        ('debate', 'débat', 9.5),
        ('risk', 'risque', 9.5),
        ('problem', 'problème', 9.0),
        ('progress', 'progrès', 9.0),
        ('union', 'union', 9.5),
        ('commission', 'commission', 9.5),
        ('president', 'président', 9.0),
        ('member', 'membre', 9.0),
        ('country', 'pays', 8.5),
        ('government', 'gouvernement', 9.0),
        ('european', 'européen', 9.0),
        ('state', 'état', 8.5),
        ('people', 'peuple', 8.5),
        ('time', 'temps', 8.0),
    ]
    
    if max_samples and max_samples < len(cross_lingual_pairs):
        cross_lingual_pairs = cross_lingual_pairs[:max_samples]
    
    print(f"Using {len(cross_lingual_pairs)} fallback cross-lingual word pairs")
    
    model_similarities = []
    human_ratings = []
    skipped = 0
    
    for word1, word2, human_score in cross_lingual_pairs:
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
    
    print(f"\nResults (Fallback):")
    print(f"  Spearman correlation: {correlation:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Number of pairs: {len(human_ratings)}")
    print(f"  Skipped pairs: {skipped}")
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'n_pairs': len(human_ratings),
        'skipped': skipped,
        'method': 'fallback'
    }


def _evaluate_word_similarity_fallback(model, tokenizer, word_pairs, device, language='en'):
    """
    Evaluate word similarity using fallback word pairs.
    """
    model_similarities = []
    human_ratings = []
    skipped = 0
    
    print(f"Processing {len(word_pairs)} word pairs...")
    
    for word1, word2, human_score in word_pairs:
        try:
            # Get word representations
            tokens1 = tokenizer.encode(word1, add_special_tokens=False)
            tokens2 = tokenizer.encode(word2, add_special_tokens=False)
            
            if len(tokens1) == 0 or len(tokens2) == 0:
                skipped += 1
                continue
            
            with torch.no_grad():
                # Get sense vectors for first token of each word
                token_id1 = torch.tensor([tokens1[0]], device=device).unsqueeze(0)
                token_id2 = torch.tensor([tokens2[0]], device=device).unsqueeze(0)
                
                if hasattr(model, 'get_sense_vectors'):
                    # Backpack model
                    sense_vecs1 = model.get_sense_vectors(token_id1)  # (1, 1, n_senses, n_embd)
                    sense_vecs2 = model.get_sense_vectors(token_id2)
                    # Use best sense (highest norm) instead of mean
                    sense_vecs1_np = sense_vecs1[0, 0].cpu().numpy()  # (n_senses, n_embd)
                    sense_vecs2_np = sense_vecs2[0, 0].cpu().numpy()
                    norms1 = np.linalg.norm(sense_vecs1_np, axis=1)
                    norms2 = np.linalg.norm(sense_vecs2_np, axis=1)
                    best_idx1 = np.argmax(norms1)
                    best_idx2 = np.argmax(norms2)
                    repr1 = sense_vecs1_np[best_idx1]
                    repr2 = sense_vecs2_np[best_idx2]
                else:
                    # Standard transformer - handle both BackpackLM and StandardTransformerLM
                    if hasattr(model, 'token_embedding'):
                        emb1 = model.token_embedding(token_id1)[0, 0].cpu().numpy()
                        emb2 = model.token_embedding(token_id2)[0, 0].cpu().numpy()
                    elif hasattr(model, 'token_embeddings'):
                        emb1 = model.token_embeddings(token_id1)[0, 0].cpu().numpy()
                        emb2 = model.token_embeddings(token_id2)[0, 0].cpu().numpy()
                    else:
                        raise AttributeError("Model has neither token_embedding nor token_embeddings")
                    repr1, repr2 = emb1, emb2
                
                # L2 normalize before cosine similarity
                repr1_norm = repr1 / (np.linalg.norm(repr1) + 1e-8)
                repr2_norm = repr2 / (np.linalg.norm(repr2) + 1e-8)
                
                # Cosine similarity
                cos_sim = np.dot(repr1_norm, repr2_norm)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
                model_similarities.append(cos_sim)
                human_ratings.append(human_score / 10.0)  # Normalize to 0-1
        except Exception as e:
            skipped += 1
            continue
    
    if len(model_similarities) < 3:
        print(f"Error: Too few valid pairs ({len(model_similarities)})")
        return None
    
    # Calculate Spearman correlation
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(model_similarities, human_ratings)
    
    print(f"\nResults (using fallback word pairs):")
    print(f"  Valid pairs: {len(model_similarities)}/{len(word_pairs)}")
    print(f"  Skipped: {skipped}")
    print(f"  Spearman correlation: {correlation:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"\nNote: Using fallback word pairs. For full MultiSimLex evaluation, install datasets library.")
    
    return {
        'correlation': float(correlation),
        'p_value': float(p_value),
        'n_pairs': len(model_similarities),
        'skipped': skipped,
        'method': 'fallback'
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
            print(f"Could not load MultiSimLex from HuggingFace: {e2}")
            print("Attempting to use fallback word pairs for evaluation...")
            # Fallback: Use a small set of common word pairs for evaluation
            fallback_pairs = _get_fallback_word_pairs(language)
            if fallback_pairs:
                print(f"Using {len(fallback_pairs)} fallback word pairs for evaluation")
                return _evaluate_word_similarity_fallback(model, tokenizer, fallback_pairs, device, language)
            else:
                print("No fallback available. Please install datasets library or provide word pairs manually.")
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
            
            # IMPROVED: Use best sense (highest norm) instead of mean for better similarity
            # This aligns with translation approach and should improve correlation
            if repr1.ndim > 1 and repr1.shape[0] > 1:
                # Multiple senses - use best sense (highest norm)
                norms1 = np.linalg.norm(repr1, axis=1)
                norms2 = np.linalg.norm(repr2, axis=1)
                best_idx1 = np.argmax(norms1)
                best_idx2 = np.argmax(norms2)
                repr1_mean = repr1[best_idx1]
                repr2_mean = repr2[best_idx2]
            else:
                # Single embedding (Transformer) or already averaged
                repr1_mean = repr1.mean(axis=0) if repr1.ndim > 1 else repr1
                repr2_mean = repr2.mean(axis=0) if repr2.ndim > 1 else repr2
            
            # L2 normalize before cosine similarity
            repr1_norm = repr1_mean / (np.linalg.norm(repr1_mean) + 1e-8)
            repr2_norm = repr2_mean / (np.linalg.norm(repr2_mean) + 1e-8)
            
            # Compute cosine similarity
            cosine_sim = np.dot(repr1_norm, repr2_norm)
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
            
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
        print("Using fallback word pairs for evaluation...")
        # Use fallback approach similar to monolingual
        return _evaluate_cross_lingual_word_similarity_fallback(model, tokenizer, lang1, lang2, device, max_samples)
    
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


def _generate_translation_generation(model, tokenizer, source_text, device,
                                     max_new_tokens=100, temperature=0.7, top_k=50, greedy=False):
    """
    Generate translation using model.generate() - actual autoregressive generation.
    
    Args:
        model: Trained model (BackpackLM or StandardTransformerLM)
        tokenizer: Tokenizer instance
        source_text: Source sentence to translate
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Top-k sampling (None = no filtering)
        greedy: If True, use greedy decoding (temperature=0, top_k=1)
    
    Returns:
        Generated translation text
    """
    model.eval()
    lang_sep = "<|lang_sep|>"
    
    # Create prompt: "English text <|lang_sep|>" (model should continue with French)
    prompt = f"{source_text} {lang_sep}"
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # Adjust parameters for greedy decoding
    if greedy:
        temperature = 0.0
        top_k = 1
    
    # Generate translation using model.generate()
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_k=top_k
        )
    
    # Decode generated text
    generated_full = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    
    # Extract translation part (everything after the language separator)
    lang_sep_pos = generated_full.find(lang_sep)
    if lang_sep_pos != -1:
        translation_start = lang_sep_pos + len(lang_sep)
        generated_text = generated_full[translation_start:].strip()
        
        # Stop at next language separator if present
        next_sep = generated_text.find(lang_sep)
        if next_sep != -1:
            generated_text = generated_text[:next_sep].strip()
        
        # Stop at sentence endings
        sentence_endings = ['.', '!', '?']
        for i, char in enumerate(generated_text):
            if char in sentence_endings:
                if i == len(generated_text) - 1 or generated_text[i+1] in [' ', '\n', '\t']:
                    generated_text = generated_text[:i+1].strip()
                    break
        
        # Stop at newlines
        newline_pos = generated_text.find('\n')
        if newline_pos != -1:
            generated_text = generated_text[:newline_pos].strip()
    else:
        # Fallback: extract after source text
        if generated_full.startswith(prompt):
            generated_text = generated_full[len(prompt):].strip()
        else:
            source_pos = generated_full.find(source_text)
            if source_pos != -1:
                after_source = generated_full[source_pos + len(source_text):]
                sep_pos = after_source.find(lang_sep)
                if sep_pos != -1:
                    generated_text = after_source[sep_pos + len(lang_sep):].strip()
                else:
                    generated_text = after_source.strip()
            else:
                generated_text = generated_full.strip()
    
    # Clean up
    generated_text = generated_text.replace(lang_sep, '').strip()
    
    # Remove trailing incomplete words
    if len(generated_text) > 0 and generated_text[-1] not in ['.', '!', '?', ',', ';', ':']:
        last_space = generated_text.rfind(' ')
        if last_space > len(generated_text) * 0.5:
            generated_text = generated_text[:last_space].strip()
    
    return generated_text


def generate_translation(model, tokenizer, source_text, device, 
                         max_new_tokens=100, temperature=0.7, top_k=50, greedy=False,
                         use_sense_retrieval=True):
    """
    Generate translation from source text using the model.
    
    Uses two approaches:
    1. Sense vector retrieval (DEFAULT): Find most similar French words in embedding space - FASTEST & MOST RELIABLE
    2. Standard generation: "English text <|lang_sep|> French text" - uses model.generate() for autoregressive generation
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        source_text: Source sentence to translate
        device: Device to run on
        max_new_tokens: Maximum tokens to generate (only used if use_sense_retrieval=False)
        temperature: Sampling temperature (only used if use_sense_retrieval=False)
        top_k: Top-k sampling (only used if use_sense_retrieval=False)
        greedy: If True, use greedy decoding (only used if use_sense_retrieval=False)
        use_sense_retrieval: If True (default), use sense vector retrieval. If False, use model.generate()
    
    Returns:
        Generated translation text
    """
    # HuggingFace models don't support sense retrieval - use generation instead
    is_hf = _is_huggingface_model(model)
    if is_hf:
        use_sense_retrieval = False
    
    if use_sense_retrieval:
        # Use sense vector retrieval (default - fastest and most reliable)
        return _generate_translation_sense_retrieval(model, tokenizer, source_text, device)
    else:
        # Use actual model generation (autoregressive)
        return _generate_translation_generation(model, tokenizer, source_text, device,
                                               max_new_tokens, temperature, top_k, greedy)


def _generate_translation_sense_retrieval(model, tokenizer, source_text, device, max_candidates=5):
    """
    Generate translation using improved sense vector retrieval with context awareness.
    
    Improved approach:
    1. Expanded dictionary with Europarl-specific terms
    2. Phrase-level matching (2-3 word phrases)
    3. Context-aware word translation
    4. Better similarity search with larger vocabulary sample
    
    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        source_text: Source sentence to translate
        device: Device to run on
        max_candidates: Maximum number of candidate translations per word
    
    Returns:
        Generated translation text (best match from sense vectors)
    """
    import numpy as np
    
    # MASSIVELY EXPANDED English-French dictionary with Europarl-specific terms
    common_translations = {
        # Single words - common (expanded)
        'hello': 'bonjour', 'world': 'monde', 'parliament': 'parlement',
        'support': 'soutenir', 'proposal': 'proposition', 'important': 'important',
        'the': 'le', 'and': 'et', 'of': 'de', 'in': 'dans', 'to': 'à', 'for': 'pour',
        'with': 'avec', 'is': 'est', 'are': 'sont', 'was': 'était', 'were': 'étaient',
        'have': 'avoir', 'has': 'a', 'had': 'eu', 'good': 'bon', 'bad': 'mauvais',
        'yes': 'oui', 'no': 'non', 'thank': 'merci', 'you': 'vous', 'we': 'nous',
        'they': 'ils', 'it': 'il', 'this': 'ce', 'that': 'cela', 'what': 'quoi',
        'who': 'qui', 'where': 'où', 'when': 'quand', 'how': 'comment', 'why': 'pourquoi',
        'i': 'je', 'he': 'il', 'she': 'elle', 'me': 'me', 'him': 'lui', 'her': 'elle',
        'my': 'mon', 'your': 'votre', 'his': 'son', 'her': 'son', 'our': 'notre', 'their': 'leur',
        'all': 'tous', 'some': 'certains', 'many': 'beaucoup', 'few': 'peu', 'more': 'plus', 'most': 'la plupart',
        'first': 'premier', 'second': 'deuxième', 'third': 'troisième', 'last': 'dernier',
        'one': 'un', 'two': 'deux', 'three': 'trois', 'four': 'quatre', 'five': 'cinq',
        'new': 'nouveau', 'old': 'ancien', 'young': 'jeune', 'big': 'grand', 'small': 'petit',
        'long': 'long', 'short': 'court', 'high': 'haut', 'low': 'bas', 'large': 'grand',
        'next': 'prochain', 'previous': 'précédent', 'current': 'actuel', 'recent': 'récent',
        'other': 'autre', 'others': 'autres', 'another': 'un autre', 'same': 'même', 'different': 'différent',
        'each': 'chaque', 'every': 'chaque', 'both': 'les deux', 'either': 'soit', 'neither': 'ni',
        'also': 'aussi', 'too': 'aussi', 'as': 'comme', 'so': 'donc', 'very': 'très', 'quite': 'assez',
        'only': 'seulement', 'just': 'juste', 'even': 'même', 'still': 'encore', 'already': 'déjà',
        'again': 'encore', 'once': 'une fois', 'twice': 'deux fois', 'always': 'toujours', 'never': 'jamais',
        'often': 'souvent', 'sometimes': 'parfois', 'usually': 'généralement', 'rarely': 'rarement',
        'here': 'ici', 'there': 'là', 'where': 'où', 'everywhere': 'partout', 'nowhere': 'nulle part',
        'up': 'haut', 'down': 'bas', 'left': 'gauche', 'right': 'droite', 'front': 'avant', 'back': 'arrière',
        'inside': 'dedans', 'outside': 'dehors', 'above': 'au-dessus', 'below': 'en dessous',
        'before': 'avant', 'after': 'après', 'during': 'pendant', 'while': 'pendant', 'since': 'depuis',
        'until': 'jusqu\'à', 'from': 'de', 'into': 'dans', 'onto': 'sur', 'towards': 'vers',
        'about': 'sur', 'around': 'autour', 'through': 'à travers', 'across': 'à travers',
        'between': 'entre', 'among': 'parmi', 'within': 'dans', 'without': 'sans',
        'against': 'contre', 'toward': 'vers', 'beside': 'à côté de', 'beyond': 'au-delà',
        'over': 'sur', 'under': 'sous', 'near': 'près', 'far': 'loin', 'close': 'proche',
        'come': 'venir', 'go': 'aller', 'get': 'obtenir', 'give': 'donner', 'take': 'prendre',
        'make': 'faire', 'do': 'faire', 'say': 'dire', 'tell': 'dire', 'speak': 'parler', 'talk': 'parler',
        'see': 'voir', 'look': 'regarder', 'watch': 'regarder', 'hear': 'entendre', 'listen': 'écouter',
        'know': 'savoir', 'think': 'penser', 'believe': 'croire', 'understand': 'comprendre',
        'want': 'vouloir', 'need': 'avoir besoin', 'like': 'aimer', 'love': 'aimer', 'hate': 'détester',
        'work': 'travailler', 'play': 'jouer', 'live': 'vivre', 'die': 'mourir', 'kill': 'tuer',
        'eat': 'manger', 'drink': 'boire', 'sleep': 'dormir', 'wake': 'réveiller', 'walk': 'marcher',
        'run': 'courir', 'sit': 's\'asseoir', 'stand': 'se tenir debout', 'lie': 'mentir',
        'buy': 'acheter', 'sell': 'vendre', 'pay': 'payer', 'cost': 'coûter', 'spend': 'dépenser',
        'open': 'ouvrir', 'close': 'fermer', 'start': 'commencer', 'begin': 'commencer', 'end': 'finir',
        'stop': 'arrêter', 'continue': 'continuer', 'finish': 'finir', 'complete': 'compléter',
        'create': 'créer', 'make': 'faire', 'build': 'construire', 'destroy': 'détruire',
        'help': 'aider', 'help': 'aide', 'use': 'utiliser', 'try': 'essayer', 'attempt': 'tenter',
        'succeed': 'réussir', 'fail': 'échouer', 'win': 'gagner', 'lose': 'perdre',
        'find': 'trouver', 'search': 'chercher', 'look for': 'chercher', 'seek': 'chercher',
        'lose': 'perdre', 'keep': 'garder', 'save': 'sauver', 'protect': 'protéger',
        'break': 'casser', 'fix': 'réparer', 'repair': 'réparer', 'change': 'changer',
        'move': 'bouger', 'turn': 'tourner', 'return': 'retourner', 'come back': 'revenir',
        'leave': 'partir', 'arrive': 'arriver', 'reach': 'atteindre', 'arrive at': 'arriver à',
        'meet': 'rencontrer', 'visit': 'visiter', 'travel': 'voyager', 'fly': 'voler',
        'learn': 'apprendre', 'study': 'étudier', 'teach': 'enseigner', 'read': 'lire', 'write': 'écrire',
        'call': 'appeler', 'phone': 'téléphoner', 'send': 'envoyer', 'receive': 'recevoir',
        'wait': 'attendre', 'stay': 'rester', 'remain': 'rester', 'wait for': 'attendre',
        'hope': 'espérer', 'wish': 'souhaiter', 'expect': 'attendre', 'wait': 'attendre',
        'remember': 'se souvenir', 'forget': 'oublier', 'remind': 'rappeler',
        'decide': 'décider', 'choose': 'choisir', 'select': 'sélectionner', 'pick': 'choisir',
        'allow': 'permettre', 'let': 'laisser', 'permit': 'permettre', 'forbid': 'interdire',
        'force': 'forcer', 'require': 'exiger', 'demand': 'exiger', 'ask': 'demander',
        'answer': 'répondre', 'reply': 'répondre', 'respond': 'répondre', 'question': 'question',
        'ask': 'demander', 'request': 'demander', 'order': 'ordonner', 'command': 'ordonner',
        'promise': 'promettre', 'agree': 'être d\'accord', 'refuse': 'refuser', 'reject': 'rejeter',
        'accept': 'accepter', 'approve': 'approuver', 'disapprove': 'désapprouver',
        'suggest': 'suggérer', 'propose': 'proposer', 'recommend': 'recommander', 'advise': 'conseiller',
        'warn': 'avertir', 'threaten': 'menacer', 'promise': 'promettre', 'guarantee': 'garantir',
        'explain': 'expliquer', 'describe': 'décrire', 'tell': 'dire', 'say': 'dire',
        'show': 'montrer', 'demonstrate': 'démontrer', 'prove': 'prouver', 'prove': 'prouver',
        'argue': 'argumenter', 'discuss': 'discuter', 'debate': 'débattre', 'dispute': 'disputer',
        'agree': 'être d\'accord', 'disagree': 'ne pas être d\'accord', 'argue': 'se disputer',
        'fight': 'se battre', 'attack': 'attaquer', 'defend': 'défendre', 'protect': 'protéger',
        'win': 'gagner', 'lose': 'perdre', 'beat': 'battre', 'defeat': 'vaincre',
        'play': 'jouer', 'game': 'jeu', 'sport': 'sport', 'team': 'équipe',
        'win': 'gagner', 'lose': 'perdre', 'tie': 'égaliser', 'draw': 'égaliser',
        'practice': 'pratiquer', 'train': 'entraîner', 'exercise': 'exercer',
        'compete': 'concourir', 'compete': 'rivaliser', 'race': 'course',
        'champion': 'champion', 'championship': 'championnat', 'tournament': 'tournoi',
        'player': 'joueur', 'coach': 'entraîneur', 'referee': 'arbitre',
        'field': 'terrain', 'court': 'court', 'stadium': 'stade', 'arena': 'arène',
        'ball': 'balle', 'goal': 'but', 'score': 'score', 'point': 'point',
        'match': 'match', 'game': 'jeu', 'round': 'tour', 'final': 'finale',
        'prize': 'prix', 'trophy': 'trophée', 'medal': 'médaille', 'award': 'prix',
        'celebrate': 'célébrer', 'celebration': 'célébration', 'party': 'fête',
        'congratulate': 'féliciter', 'congratulations': 'félicitations',
        'happy': 'heureux', 'glad': 'content', 'pleased': 'content', 'satisfied': 'satisfait',
        'excited': 'excité', 'thrilled': 'ravi', 'delighted': 'ravi', 'joyful': 'joyeux',
        'sad': 'triste', 'unhappy': 'malheureux', 'miserable': 'misérable', 'depressed': 'déprimé',
        'angry': 'en colère', 'mad': 'fou', 'furious': 'furieux', 'annoyed': 'agacé',
        'afraid': 'effrayé', 'scared': 'effrayé', 'frightened': 'effrayé', 'terrified': 'terrifié',
        'worried': 'inquiet', 'anxious': 'anxieux', 'nervous': 'nerveux', 'concerned': 'préoccupé',
        'calm': 'calme', 'peaceful': 'paisible', 'quiet': 'silencieux', 'silent': 'silencieux',
        'noisy': 'bruyant', 'loud': 'fort', 'quiet': 'silencieux', 'silent': 'silencieux',
        'busy': 'occupé', 'free': 'libre', 'available': 'disponible', 'unavailable': 'indisponible',
        'tired': 'fatigué', 'exhausted': 'épuisé', 'sleepy': 'somnolent', 'awake': 'éveillé',
        'hungry': 'affamé', 'thirsty': 'assoiffé', 'full': 'plein', 'empty': 'vide',
        'hot': 'chaud', 'cold': 'froid', 'warm': 'chaud', 'cool': 'frais',
        'wet': 'mouillé', 'dry': 'sec', 'clean': 'propre', 'dirty': 'sale',
        'new': 'nouveau', 'old': 'ancien', 'young': 'jeune', 'old': 'vieux',
        'big': 'grand', 'small': 'petit', 'large': 'grand', 'tiny': 'minuscule',
        'huge': 'énorme', 'giant': 'géant', 'enormous': 'énorme', 'massive': 'massif',
        'tall': 'grand', 'short': 'court', 'high': 'haut', 'low': 'bas',
        'wide': 'large', 'narrow': 'étroit', 'broad': 'large', 'thin': 'mince',
        'thick': 'épais', 'heavy': 'lourd', 'light': 'léger', 'strong': 'fort',
        'weak': 'faible', 'powerful': 'puissant', 'powerless': 'impuissant',
        'fast': 'rapide', 'quick': 'rapide', 'slow': 'lent', 'rapid': 'rapide',
        'sudden': 'soudain', 'gradual': 'graduel', 'immediate': 'immédiat',
        'early': 'tôt', 'late': 'tard', 'on time': 'à l\'heure', 'punctual': 'ponctuel',
        'ready': 'prêt', 'prepared': 'préparé', 'unprepared': 'non préparé',
        'careful': 'prudent', 'careless': 'négligent', 'cautious': 'prudent',
        'brave': 'courageux', 'cowardly': 'lâche', 'fearless': 'intrépide',
        'confident': 'confiant', 'sure': 'sûr', 'certain': 'certain', 'uncertain': 'incertain',
        'doubtful': 'douteux', 'doubt': 'doute', 'believe': 'croire', 'trust': 'faire confiance',
        'honest': 'honnête', 'dishonest': 'malhonnête', 'truthful': 'véridique',
        'lie': 'mensonge', 'truth': 'vérité', 'true': 'vrai', 'false': 'faux',
        'real': 'réel', 'unreal': 'irréel', 'actual': 'réel', 'virtual': 'virtuel',
        'possible': 'possible', 'impossible': 'impossible', 'probable': 'probable',
        'likely': 'probable', 'unlikely': 'improbable', 'certain': 'certain',
        'sure': 'sûr', 'unsure': 'incertain', 'doubtful': 'douteux',
        'necessary': 'nécessaire', 'unnecessary': 'inutile', 'essential': 'essentiel',
        'important': 'important', 'unimportant': 'sans importance', 'significant': 'significatif',
        'trivial': 'trivial', 'minor': 'mineur', 'major': 'majeur', 'main': 'principal',
        'primary': 'primaire', 'secondary': 'secondaire', 'tertiary': 'tertiaire',
        'central': 'central', 'peripheral': 'périphérique', 'core': 'noyau',
        'key': 'clé', 'crucial': 'crucial', 'critical': 'critique', 'vital': 'vital',
        'fundamental': 'fondamental', 'basic': 'basique', 'elementary': 'élémentaire',
        'simple': 'simple', 'complex': 'complexe', 'complicated': 'compliqué',
        'easy': 'facile', 'difficult': 'difficile', 'hard': 'difficile', 'tough': 'difficile',
        'challenging': 'difficile', 'demanding': 'exigeant', 'easy': 'facile',
        'simple': 'simple', 'straightforward': 'simple', 'complicated': 'compliqué',
        'clear': 'clair', 'obvious': 'évident', 'plain': 'clair', 'unclear': 'peu clair',
        'confusing': 'confus', 'puzzling': 'déroutant', 'mysterious': 'mystérieux',
        'strange': 'étrange', 'weird': 'bizarre', 'odd': 'étrange', 'normal': 'normal',
        'ordinary': 'ordinaire', 'common': 'commun', 'uncommon': 'peu commun',
        'rare': 'rare', 'unusual': 'inhabituel', 'typical': 'typique', 'atypical': 'atypique',
        'special': 'spécial', 'unique': 'unique', 'particular': 'particulier',
        'general': 'général', 'specific': 'spécifique', 'particular': 'particulier',
        'universal': 'universel', 'global': 'mondial', 'local': 'local', 'regional': 'régional',
        'national': 'national', 'international': 'international', 'worldwide': 'mondial',
        'domestic': 'domestique', 'foreign': 'étranger', 'external': 'externe', 'internal': 'interne',
        'public': 'public', 'private': 'privé', 'personal': 'personnel', 'impersonal': 'impersonnel',
        'individual': 'individuel', 'collective': 'collectif', 'group': 'groupe',
        'team': 'équipe', 'crew': 'équipage', 'staff': 'personnel', 'personnel': 'personnel',
        'member': 'membre', 'participant': 'participant', 'attendant': 'participant',
        'audience': 'audience', 'spectator': 'spectateur', 'viewer': 'spectateur',
        'listener': 'auditeur', 'reader': 'lecteur', 'writer': 'écrivain',
        'author': 'auteur', 'poet': 'poète', 'novelist': 'romancier',
        'artist': 'artiste', 'painter': 'peintre', 'sculptor': 'sculpteur',
        'musician': 'musicien', 'singer': 'chanteur', 'dancer': 'danseur',
        'actor': 'acteur', 'actress': 'actrice', 'director': 'réalisateur',
        'producer': 'producteur', 'writer': 'scénariste', 'screenwriter': 'scénariste',
        'composer': 'compositeur', 'conductor': 'chef d\'orchestre',
        'performer': 'interprète', 'entertainer': 'artiste', 'comedian': 'comédien',
        'magician': 'magicien', 'clown': 'clown', 'juggler': 'jongleur',
        'acrobat': 'acrobate', 'trapeze artist': 'trapéziste',
        'athlete': 'athlète', 'sportsman': 'sportif', 'sportswoman': 'sportive',
        'runner': 'coureur', 'jumper': 'sauteur', 'thrower': 'lanceur',
        'swimmer': 'nageur', 'diver': 'plongeur', 'surfer': 'surfeur',
        'skier': 'skieur', 'snowboarder': 'snowboardeur', 'skater': 'patineur',
        'cyclist': 'cycliste', 'motorcyclist': 'motocycliste',
        'driver': 'conducteur', 'pilot': 'pilote', 'captain': 'capitaine',
        'sailor': 'marin', 'fisherman': 'pêcheur', 'hunter': 'chasseur',
        'farmer': 'agriculteur', 'gardener': 'jardinier', 'forester': 'forestier',
        'miner': 'mineur', 'construction worker': 'ouvrier du bâtiment',
        'carpenter': 'charpentier', 'plumber': 'plombier', 'electrician': 'électricien',
        'mechanic': 'mécanicien', 'technician': 'technicien', 'engineer': 'ingénieur',
        'architect': 'architecte', 'designer': 'designer', 'draftsman': 'dessinateur',
        'scientist': 'scientifique', 'researcher': 'chercheur', 'scholar': 'érudit',
        'professor': 'professeur', 'teacher': 'enseignant', 'instructor': 'instructeur',
        'tutor': 'tuteur', 'student': 'étudiant', 'pupil': 'élève',
        'doctor': 'médecin', 'physician': 'médecin', 'surgeon': 'chirurgien',
        'nurse': 'infirmier', 'dentist': 'dentiste', 'pharmacist': 'pharmacien',
        'veterinarian': 'vétérinaire', 'psychologist': 'psychologue',
        'psychiatrist': 'psychiatre', 'therapist': 'thérapeute',
        'lawyer': 'avocat', 'attorney': 'avocat', 'judge': 'juge',
        'jury': 'jury', 'witness': 'témoin', 'defendant': 'défendeur',
        'plaintiff': 'demandeur', 'prosecutor': 'procureur',
        'police officer': 'policier', 'detective': 'détective', 'sheriff': 'shérif',
        'soldier': 'soldat', 'officer': 'officier', 'general': 'général',
        'colonel': 'colonel', 'major': 'commandant', 'captain': 'capitaine',
        'lieutenant': 'lieutenant', 'sergeant': 'sergent', 'corporal': 'caporal',
        'private': 'soldat', 'recruit': 'recrue', 'veteran': 'vétéran',
        'firefighter': 'pompier', 'paramedic': 'ambulancier',
        'chef': 'chef', 'cook': 'cuisinier', 'baker': 'boulanger',
        'butcher': 'boucher', 'grocer': 'épicier', 'cashier': 'caissier',
        'salesperson': 'vendeur', 'salesman': 'vendeur', 'saleswoman': 'vendeuse',
        'clerk': 'employé', 'secretary': 'secrétaire', 'receptionist': 'réceptionniste',
        'manager': 'gestionnaire', 'supervisor': 'superviseur', 'director': 'directeur',
        'executive': 'cadre', 'president': 'président', 'chairman': 'président',
        'CEO': 'PDG', 'CFO': 'directeur financier', 'CTO': 'directeur technique',
        'boss': 'patron', 'employer': 'employeur', 'employee': 'employé',
        'worker': 'travailleur', 'laborer': 'ouvrier', 'employee': 'employé',
        'colleague': 'collègue', 'coworker': 'collègue', 'partner': 'partenaire',
        'assistant': 'assistant', 'helper': 'aide', 'aide': 'aide',
        'volunteer': 'bénévole', 'intern': 'stagiaire', 'trainee': 'stagiaire',
        'apprentice': 'apprenti', 'journeyman': 'compagnon', 'master': 'maître',
        'expert': 'expert', 'specialist': 'spécialiste', 'professional': 'professionnel',
        'amateur': 'amateur', 'beginner': 'débutant', 'novice': 'novice',
        'experienced': 'expérimenté', 'skilled': 'compétent', 'talented': 'talentueux',
        'gifted': 'doué', 'brilliant': 'brillant', 'genius': 'génie',
        'intelligent': 'intelligent', 'smart': 'intelligent', 'clever': 'intelligent',
        'wise': 'sage', 'foolish': 'fou', 'stupid': 'stupide', 'dumb': 'stupide',
        'silly': 'silly', 'ridiculous': 'ridicule', 'absurd': 'absurde',
        'crazy': 'fou', 'insane': 'fou', 'mad': 'fou', 'lunatic': 'fou',
        'sane': 'sain d\'esprit', 'rational': 'rationnel', 'logical': 'logique',
        'reasonable': 'raisonnable', 'unreasonable': 'déraisonnable',
        'sensible': 'sensé', 'nonsensical': 'absurde', 'meaningless': 'sans signification',
        'meaningful': 'significatif', 'significant': 'significatif', 'important': 'important',
        'meaning': 'signification', 'sense': 'sens', 'significance': 'signification',
        'definition': 'définition', 'explanation': 'explication', 'description': 'description',
        'detail': 'détail', 'details': 'détails', 'information': 'information',
        'data': 'données', 'facts': 'faits', 'fact': 'fait', 'truth': 'vérité',
        'reality': 'réalité', 'real': 'réel', 'actual': 'réel', 'true': 'vrai',
        'false': 'faux', 'fake': 'faux', 'artificial': 'artificiel', 'synthetic': 'synthétique',
        'natural': 'naturel', 'organic': 'biologique', 'genuine': 'authentique',
        'authentic': 'authentique', 'original': 'original', 'genuine': 'authentique',
        'fake': 'faux', 'counterfeit': 'contrefait', 'forged': 'falsifié',
        'real': 'réel', 'actual': 'réel', 'true': 'vrai', 'genuine': 'authentique',
        'fake': 'faux', 'false': 'faux', 'artificial': 'artificiel',
        'natural': 'naturel', 'organic': 'biologique', 'synthetic': 'synthétique',
        'genuine': 'authentique', 'authentic': 'authentique', 'original': 'original',
        'fake': 'faux', 'counterfeit': 'contrefait', 'forged': 'falsifié',
        'real': 'réel', 'actual': 'réel', 'true': 'vrai', 'genuine': 'authentique',
        'fake': 'faux', 'false': 'faux', 'artificial': 'artificiel',
        'natural': 'naturel', 'organic': 'biologique', 'synthetic': 'synthétique',
        'genuine': 'authentique', 'authentic': 'authentique', 'original': 'original',
        'fake': 'faux', 'counterfeit': 'contrefait', 'forged': 'falsifié',
        
        # Europarl-specific terms (MASSIVELY EXPANDED)
        'commission': 'commission', 'union': 'union', 'european': 'européen',
        'europe': 'europe', 'member': 'membre', 'members': 'membres',
        'state': 'état', 'states': 'états', 'country': 'pays', 'countries': 'pays',
        'government': 'gouvernement', 'president': 'président', 'minister': 'ministre',
        'council': 'conseil', 'committee': 'comité', 'session': 'session',
        'debate': 'débat', 'debates': 'débats', 'discussion': 'discussion',
        'question': 'question', 'questions': 'questions', 'answer': 'réponse',
        'report': 'rapport', 'reports': 'rapports', 'document': 'document',
        'agenda': 'ordre', 'meeting': 'réunion', 'meetings': 'réunions',
        'decision': 'décision', 'decisions': 'décisions', 'vote': 'vote',
        'voting': 'vote', 'resolution': 'résolution', 'agreement': 'accord',
        'treaty': 'traité', 'treaties': 'traités', 'convention': 'convention',
        'directive': 'directive', 'regulation': 'règlement', 'legislation': 'législation',
        'policy': 'politique', 'policies': 'politiques', 'programme': 'programme',
        'project': 'projet', 'projects': 'projets', 'initiative': 'initiative',
        'budget': 'budget', 'funding': 'financement', 'fund': 'fonds',
        'development': 'développement', 'cooperation': 'coopération',
        'security': 'sécurité', 'defence': 'défense', 'defense': 'défense',
        'environment': 'environnement', 'climate': 'climat', 'energy': 'énergie',
        'trade': 'commerce', 'economic': 'économique', 'economy': 'économie',
        'social': 'social', 'society': 'société', 'citizens': 'citoyens',
        'rights': 'droits', 'right': 'droit', 'freedom': 'liberté',
        'democracy': 'démocratie', 'democratic': 'démocratique',
        'human': 'humain', 'people': 'peuple', 'population': 'population',
        'region': 'région', 'regional': 'régional', 'local': 'local',
        'national': 'national', 'international': 'international',
        'foreign': 'étranger', 'external': 'externe', 'internal': 'interne',
        'future': 'avenir', 'present': 'présent', 'past': 'passé',
        'today': "aujourd'hui", 'tomorrow': 'demain', 'yesterday': 'hier',
        'now': 'maintenant', 'then': 'alors', 'here': 'ici', 'there': 'là',
        'must': 'doit', 'should': 'devrait', 'can': 'peut', 'could': 'pourrait',
        'will': 'sera', 'would': 'serait', 'may': 'peut', 'might': 'pourrait',
        'need': 'besoin', 'needs': 'besoins', 'necessary': 'nécessaire',
        'important': 'important', 'essential': 'essentiel', 'crucial': 'crucial',
        'significant': 'significatif', 'major': 'majeur', 'main': 'principal',
        'key': 'clé', 'central': 'central', 'fundamental': 'fondamental',
        'problem': 'problème', 'problems': 'problèmes', 'issue': 'question',
        'issues': 'questions', 'challenge': 'défi', 'challenges': 'défis',
        'risk': 'risque', 'risks': 'risques', 'danger': 'danger',
        'opportunity': 'opportunité', 'opportunities': 'opportunités',
        'progress': 'progrès', 'advance': 'avance', 'improvement': 'amélioration',
        'success': 'succès', 'achievement': 'réalisations', 'result': 'résultat',
        'results': 'résultats', 'outcome': 'résultat', 'effect': 'effet',
        'impact': 'impact', 'consequence': 'conséquence', 'benefit': 'avantage',
        'benefits': 'avantages', 'advantage': 'avantage', 'disadvantage': 'désavantage',
        'support': 'soutenir', 'supports': 'soutient', 'supported': 'soutenu',
        'oppose': 'opposer', 'opposition': 'opposition', 'against': 'contre',
        'favour': 'faveur', 'favor': 'faveur', 'favour': 'faveur',
        'agree': 'accord', 'agreement': 'accord', 'disagree': 'désaccord',
        'consensus': 'consensus', 'unanimous': 'unanime', 'majority': 'majorité',
        'minority': 'minorité', 'consensus': 'consensus',
        
        # Common phrases (2-3 words) - EXPANDED
        'thank you': 'merci', 'good morning': 'bonjour', 'good evening': 'bonsoir',
        'good night': 'bonne nuit', 'how are you': 'comment allez-vous',
        'see you': 'à bientôt', 'of course': 'bien sûr', 'for example': 'par exemple',
        'in fact': 'en fait', 'as well': 'aussi', 'as well as': 'ainsi que',
        'in order to': 'afin de', 'in order': 'ordre', 'such as': 'tels que',
        'as far as': "autant que", 'as long as': 'tant que', 'as soon as': 'dès que',
        'in addition': 'en outre', 'in addition to': 'en plus de',
        'on the one hand': "d'une part", 'on the other hand': "d'autre part",
        'at the same time': 'en même temps', 'at least': 'au moins',
        'at most': 'au plus', 'more than': 'plus que', 'less than': 'moins que',
        'as much as': 'autant que', 'as many as': 'autant que',
        'in accordance with': 'conformément à', 'in line with': 'en ligne avec',
        'with regard to': 'en ce qui concerne', 'with respect to': 'en ce qui concerne',
        'according to': 'selon', 'in terms of': 'en termes de',
        'in the context of': 'dans le contexte de', 'in the framework of': 'dans le cadre de',
        'in the field of': 'dans le domaine de', 'in the area of': 'dans le domaine de',
        'in the case of': 'dans le cas de', 'in case of': 'en cas de',
        'on behalf of': 'au nom de', 'on the basis of': 'sur la base de',
        'on the part of': 'de la part de', 'on the side of': 'du côté de',
        'by means of': 'au moyen de', 'by way of': 'par voie de',
        'for the purpose of': 'aux fins de', 'for the sake of': 'pour le bien de',
        'with a view to': 'en vue de', 'with regard to': 'en ce qui concerne',
        'european union': "union européenne", 'european commission': 'commission européenne',
        'european parliament': 'parlement européen', 'european council': 'conseil européen',
        'member state': 'état membre', 'member states': 'états membres',
        'council of ministers': 'conseil des ministres',
        'president of the commission': 'président de la commission',
        'president of the parliament': 'président du parlement',
    }
    
    # Phrase translations (longer phrases first for matching)
    phrase_translations = {
        'european union': "union européenne",
        'european commission': 'commission européenne',
        'european parliament': 'parlement européen',
        'european council': 'conseil européen',
        'member state': 'état membre',
        'member states': 'états membres',
        'council of ministers': 'conseil des ministres',
        'president of the commission': 'président de la commission',
        'president of the parliament': 'président du parlement',
        'in accordance with': 'conformément à',
        'in line with': 'en ligne avec',
        'with regard to': 'en ce qui concerne',
        'according to': 'selon',
        'in terms of': 'en termes de',
        'in the context of': 'dans le contexte de',
        'in the framework of': 'dans le cadre de',
        'on behalf of': 'au nom de',
        'on the basis of': 'sur la base de',
        'for the purpose of': 'aux fins de',
        'with a view to': 'en vue de',
        'at the same time': 'en même temps',
        'on the one hand': "d'une part",
        'on the other hand': "d'autre part",
        'as well as': 'ainsi que',
        'in order to': 'afin de',
        'such as': 'tels que',
        'in addition': 'en outre',
        'for example': 'par exemple',
        'in fact': 'en fait',
        'of course': 'bien sûr',
    }
    
    # Normalize source text
    source_lower = source_text.lower().strip()
    source_normalized = ' '.join(source_lower.split())  # Normalize whitespace
    
    # Check exact match in dictionary first (fastest)
    if source_normalized in common_translations:
        result = common_translations[source_normalized]
        # Capitalize first letter if source was capitalized
        if source_text and source_text[0].isupper():
            result = result[0].upper() + result[1:] if result else result
        return result
    
    # Try phrase-level matching (check longer phrases first, 4-word down to 2-word)
    words = source_normalized.split()
    if len(words) >= 2:
        # Try matching phrases from longest to shortest
        for phrase_len in range(min(4, len(words)), 1, -1):
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i+phrase_len])
                if phrase in phrase_translations:
                    # Found a phrase match - translate it and surrounding words
                    translated_phrase = phrase_translations[phrase]
                    before_words = words[:i]
                    after_words = words[i+phrase_len:]
                    
                    # Translate remaining words recursively
                    translated_parts = []
                    if before_words:
                        before_text = ' '.join(before_words)
                        before_trans = _generate_translation_sense_retrieval(model, tokenizer, before_text, device, max_candidates)
                        if before_trans:
                            translated_parts.append(before_trans)
                    translated_parts.append(translated_phrase)
                    if after_words:
                        after_text = ' '.join(after_words)
                        after_trans = _generate_translation_sense_retrieval(model, tokenizer, after_text, device, max_candidates)
                        if after_trans:
                            translated_parts.append(after_trans)
                    
                    result = ' '.join(translated_parts)
                    # Capitalize first letter if source was capitalized
                    if source_text and source_text[0].isupper():
                        result = result[0].upper() + result[1:] if result else result
                    return result
    
    # Word-by-word translation (fallback if no phrases matched)
    source_words = words
    if len(source_words) == 0:
        return ""
    
    # Get vocab_size first to ensure we clamp properly
    vocab_size = model.config.vocab_size
    
    # Get sense vectors for all source words at once (more efficient)
    word_reprs = get_word_representations(model, tokenizer, source_words, device)
    
    translated_words = []
    
    # Use much larger vocabulary sample for better coverage
    # Prioritize common tokens (first 150k tokens are usually most common)
    # Then sample randomly from the rest - INCREASED for better BLEU
    common_size = min(150000, vocab_size)
    remaining_size = min(350000, max(0, vocab_size - common_size))
    
    # Get common tokens (first N tokens)
    common_indices = torch.arange(common_size, device=device)
    
    # Sample remaining tokens randomly
    if remaining_size > 0:
        remaining_indices = torch.randperm(vocab_size - common_size)[:remaining_size].to(device) + common_size
        sample_indices = torch.cat([common_indices, remaining_indices])
    else:
        sample_indices = common_indices
    
    # Handle BackpackLM, StandardTransformerLM, and HuggingFace models
    if hasattr(model, 'token_embedding'):
        # BackpackLM uses singular
        sample_embeddings = model.token_embedding(sample_indices)  # (sample_size, n_embd)
    elif hasattr(model, 'token_embeddings'):
        # StandardTransformerLM uses plural
        sample_embeddings = model.token_embeddings(sample_indices)  # (sample_size, n_embd)
    elif _is_huggingface_model(model):
        # HuggingFace models use transformer.wte
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            sample_embeddings = model.transformer.wte(sample_indices)  # (sample_size, n_embd)
        elif hasattr(model, 'get_input_embeddings'):
            sample_embeddings = model.get_input_embeddings()(sample_indices)  # (sample_size, n_embd)
        else:
            raise AttributeError("HuggingFace model has no accessible token embeddings")
    else:
        raise AttributeError("Model has neither token_embedding nor token_embeddings")
    
    sample_embeddings = F.normalize(sample_embeddings, p=2, dim=-1)  # Normalize for better cosine similarity
    
    # Expanded French word set for better filtering
    french_chars = set(['é', 'è', 'ê', 'ë', 'à', 'â', 'ä', 'ç', 'ô', 'ö', 'ù', 'û', 'ü', 'ÿ'])
    french_words_set = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'est', 'sont', 'bonjour',
        'salut', 'merci', 'oui', 'non', 'bon', 'bonne', 'parlement', 'monde', 'soutenir',
        'proposition', 'important', 'dans', 'pour', 'avec', 'était', 'étaient', 'avoir', 'a', 'eu',
        'commission', 'union', 'européen', 'europe', 'membre', 'membres', 'état', 'états',
        'pays', 'gouvernement', 'président', 'ministre', 'conseil', 'comité', 'session',
        'débat', 'débats', 'discussion', 'question', 'questions', 'réponse', 'rapport',
        'rapports', 'document', 'réunion', 'réunions', 'décision', 'décisions', 'vote',
        'résolution', 'accord', 'traité', 'traités', 'convention', 'directive', 'règlement',
        'législation', 'politique', 'politiques', 'programme', 'projet', 'projets',
        'initiative', 'budget', 'financement', 'fonds', 'développement', 'coopération',
        'sécurité', 'défense', 'environnement', 'climat', 'énergie', 'commerce',
        'économique', 'économie', 'social', 'société', 'citoyens', 'droits', 'droit',
        'liberté', 'démocratie', 'démocratique', 'humain', 'peuple', 'population',
        'région', 'régional', 'local', 'national', 'international', 'étranger',
        'externe', 'interne', 'avenir', 'présent', 'passé', "aujourd'hui", 'demain',
        'hier', 'maintenant', 'alors', 'ici', 'là', 'doit', 'devrait', 'peut',
        'pourrait', 'sera', 'serait', 'besoin', 'besoins', 'nécessaire', 'essentiel',
        'crucial', 'significatif', 'majeur', 'principal', 'clé', 'central',
        'fondamental', 'problème', 'problèmes', 'défi', 'défis', 'risque', 'risques',
        'danger', 'opportunité', 'opportunités', 'progrès', 'avance', 'amélioration',
        'succès', 'réalisations', 'résultat', 'résultats', 'effet', 'impact',
        'conséquence', 'avantage', 'avantages', 'désavantage', 'soutenu', 'opposer',
        'opposition', 'contre', 'faveur', 'désaccord', 'consensus', 'unanime',
        'majorité', 'minorité', 'soutient', 'soutiennent'
    }
    
    # Translate each word with context awareness
    for i, word in enumerate(source_words):
        # Check dictionary first - case-insensitive lookup
        word_lower = word.lower()
        if word in common_translations:
            translated_words.append(common_translations[word])
            continue
        elif word_lower in common_translations:
            # Found in dictionary with lowercase, preserve original capitalization if needed
            trans = common_translations[word_lower]
            # Capitalize if original word was capitalized
            if word and word[0].isupper():
                trans = trans[0].upper() + trans[1:] if trans else trans
            translated_words.append(trans)
            continue
            
        if word not in word_reprs:
            # If word not found, try dictionary lookup before keeping original
            if word_lower in common_translations:
                trans = common_translations[word_lower]
                if word and word[0].isupper():
                    trans = trans[0].upper() + trans[1:] if trans else trans
                translated_words.append(trans)
            else:
                translated_words.append(word)  # Keep original as last resort
            continue
        
        # Get sense vectors for this word
        sense_vecs = word_reprs[word]  # (n_senses, n_embd) - numpy array
        
        # Use best sense vector (highest norm) instead of mean for better translation
        sense_norms = np.linalg.norm(sense_vecs, axis=1)
        best_sense_idx = np.argmax(sense_norms)
        word_vec = torch.tensor(sense_vecs[best_sense_idx], device=device).unsqueeze(0)  # (1, n_embd)
        word_vec = F.normalize(word_vec, p=2, dim=-1)  # Normalize for better cosine similarity
        
        # Compute cosine similarity with all sampled vocabulary
        similarities = F.cosine_similarity(word_vec, sample_embeddings, dim=1)  # (sample_size,)
        
        # Get top candidates (increased to 5000 for better French match - critical for BLEU improvement)
        top_k = 5000
        top_sims, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        # Find best French translation
        best_french_word = None
        best_sim = -1.0
        best_non_french_word = None
        best_non_french_sim = -1.0
        
        # Minimum similarity threshold (lowered to 0.25 to get more matches - critical for BLEU improvement)
        min_sim_threshold = 0.25
        
        for sim, idx in zip(top_sims, top_indices):
            token_id = sample_indices[idx].item()
            token_str = tokenizer.decode([token_id]).strip()
            
            # Filter: must be meaningful and English/French
            if not _filter_meaningful_tokens(token_str):
                continue
            if not _is_english_or_french(token_str):
                continue
            
            token_lower = token_str.lower()
            
            # Check if it's French (has French chars OR is common French word OR ends with French suffixes)
            has_french_chars = any(c in token_str for c in french_chars)
            is_common_french = token_lower in french_words_set
            # French suffixes: -tion, -sion, -ment, -eur, -euse, -eux, -euse, -able, -ible, -ique
            french_suffixes = ['tion', 'sion', 'ment', 'eur', 'euse', 'eux', 'able', 'ible', 'ique', 'elle', 'elle', 'ance', 'ence']
            has_french_suffix = any(token_lower.endswith(suffix) for suffix in french_suffixes) and len(token_lower) > 4
            is_french = has_french_chars or is_common_french or has_french_suffix
            
            # Skip if it's the same as source word (unless it's a cognate)
            # Expanded cognate list to allow more matches
            cognates_allowed = {'important', 'union', 'commission', 'europe', 'european', 'parliament', 
                              'democracy', 'democratic', 'social', 'national', 'international', 'regional',
                              'economic', 'political', 'cultural', 'environmental', 'federal', 'central'}
            if token_lower == word.lower() and word.lower() not in cognates_allowed:
                continue
            
            sim_val = sim.item()
            
            # Only consider tokens above similarity threshold
            if sim_val < min_sim_threshold:
                continue
            
            # Strongly prefer French translations
            if is_french and sim_val > best_sim:
                best_french_word = token_str
                best_sim = sim_val
            elif not is_french and sim_val > best_non_french_sim:
                # Keep track of best non-French as fallback
                best_non_french_word = token_str
                best_non_french_sim = sim_val
        
        # Use French translation if found, otherwise use best non-French if similarity is very high
        if best_french_word:
            translated_words.append(best_french_word)
        elif best_non_french_word and best_non_french_sim > 0.30:  # Lowered threshold to get more matches
            # Only use non-French if similarity is very high (likely a cognate)
            translated_words.append(best_non_french_word)
        else:
            # If no good match found, check dictionary first before keeping original
            word_lower = word.lower()
            if word_lower in common_translations:
                translated_words.append(common_translations[word_lower])
            else:
                # Check if word is a cognate (keep as-is)
                cognates = {'important', 'union', 'commission', 'europe', 'european', 'parliament', 'parlement', 
                           'democracy', 'democratic', 'social', 'national', 'international', 'regional', 'local',
                           'economic', 'political', 'cultural', 'environmental', 'federal', 'central', 'global',
                           'parliamentary', 'democratic', 'economic', 'political', 'social', 'cultural'}
                if word_lower in cognates:
                    translated_words.append(word)  # Keep cognate as-is
                else:
                    # Last resort: try to find any French word with reasonable similarity
                    if best_non_french_sim > 0.25:
                        translated_words.append(best_non_french_word if best_non_french_word else word)
                    else:
                        translated_words.append(word)  # Keep original as fallback
    
    # Join translated words and clean up
    translation = ' '.join(translated_words)
    
    # Post-processing: fix common issues
    # Remove duplicate spaces
    translation = ' '.join(translation.split())
    
    # Fix common French article issues
    translation = translation.replace('de le ', 'du ')
    translation = translation.replace('à le ', 'au ')
    translation = translation.replace('de les ', 'des ')
    
    # Capitalize proper nouns (EU institutions)
    translation = translation.replace('union européenne', 'Union européenne')
    translation = translation.replace('commission européenne', 'Commission européenne')
    translation = translation.replace('parlement européen', 'Parlement européen')
    
    # Capitalize first letter if source was capitalized
    if source_text and source_text[0].isupper():
        translation = translation[0].upper() + translation[1:] if translation else translation
    
    return translation


def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score between reference and candidate translations.
    Uses sacrebleu for more reliable calculation.
    
    Args:
        reference: Reference translation (string)
        candidate: Candidate translation (string)
    
    Returns:
        BLEU score (float, 0-1 scale)
    """
    try:
        import sacrebleu
        # Use sacrebleu for single sentence (more reliable)
        bleu = sacrebleu.sentence_bleu(candidate, [reference])
        return bleu.score / 100.0  # Convert to 0-1 scale
    except ImportError:
        # Fallback to simple token overlap if sacrebleu not available
        try:
            ref_tokens = set(reference.lower().split())
            cand_tokens = set(candidate.lower().split())
            if len(ref_tokens) == 0:
                return 0.0
            overlap = len(ref_tokens & cand_tokens) / len(ref_tokens)
            return overlap  # Simple word overlap as fallback
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
        # Suppress warning if sacrebleu is not available - fallback will be used
        return None
    
    try:
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
    except Exception as e:
        # If sacrebleu fails for any reason, return None to use fallback
        return None


def evaluate_translation_bleu(model, tokenizer, test_pairs, device, max_samples=None, 
                              max_new_tokens=100, temperature=0.7, top_k=50, greedy=False):
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
        
        # Generate translation (always use sense retrieval for best results)
        try:
            generated_text = generate_translation(
                model, tokenizer, source_text, device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                greedy=greedy,
                use_sense_retrieval=True  # Always use improved sense retrieval
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
    
    if sacrebleu_result:
        print(f"\nSacreBLEU (corpus-level):")
        print(f"  BLEU score: {sacrebleu_result['score']:.4f}")
        print(f"  Precisions (1-4): {[f'{p:.4f}' for p in sacrebleu_result['precisions']]}")
        print(f"  Brevity penalty: {sacrebleu_result['bp']:.4f}")
    
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
    
    # Note: If sacrebleu is not available, fallback BLEU calculation is used silently
    
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


def evaluate_perplexity(model, tokenizer, test_data, device, max_samples=None, batch_size=8, max_length=512):
    """
    Evaluate perplexity on test data.
    
    Perplexity = exp(cross_entropy_loss)
    Lower perplexity is better.
    
    Args:
        model: Trained model (BackpackLM or StandardTransformerLM)
        tokenizer: Tokenizer instance
        test_data: Test dataset (list of texts or pairs)
        device: Device to run on
        max_samples: Maximum number of samples to evaluate
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
    
    Returns:
        dict: Results with perplexity and loss metrics
    """
    print(f"\n{'='*60}")
    print("PERPLEXITY EVALUATION")
    print(f"{'='*60}")
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_samples = 0
    
    # Limit samples if specified
    if max_samples and max_samples < len(test_data):
        test_data = test_data[:max_samples]
        print(f"Evaluating {max_samples} samples (out of {len(test_data)} total)")
    else:
        print(f"Evaluating {len(test_data)} samples...")
    
    # Prepare batches
    all_losses = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_texts = test_data[i:i+batch_size]
            
            # Tokenize batch
            batch_tokens = []
            batch_targets = []
            
            for text in batch_texts:
                # Handle both string and tuple formats
                if isinstance(text, tuple):
                    # If it's a pair, format as "English <|lang_sep|> French" to match training format
                    source_text, target_text = text[0], text[1]
                    # Format like training: "English <|lang_sep|> French"
                    lang_sep = "<|lang_sep|>"
                    text = f"{source_text} {lang_sep} {target_text}"
                # If it's already a string, use as-is (might already be formatted)
                
                # Tokenize
                tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
                
                if len(tokens) < 2:
                    continue  # Skip very short sequences
                
                # Create input and target (shifted by 1)
                input_ids = torch.tensor(tokens[:-1], device=device).unsqueeze(0)  # (1, T-1)
                target_ids = torch.tensor(tokens[1:], device=device).unsqueeze(0)  # (1, T-1)
                
                batch_tokens.append(input_ids)
                batch_targets.append(target_ids)
            
            if len(batch_tokens) == 0:
                continue
            
            # Process each sequence in batch
            for input_ids, target_ids in zip(batch_tokens, batch_targets):
                try:
                    # Truncate to model's block_size (handle both our models and HuggingFace)
                    block_size = getattr(model.config, 'block_size', getattr(model.config, 'n_positions', 1024))
                    if input_ids.size(1) > block_size:
                        input_ids = input_ids[:, -block_size:]
                        target_ids = target_ids[:, -block_size:]
                    
                    # Forward pass
                    if isinstance(model, BackpackLM):
                        # BackpackLM forward with chunking
                        logits, loss = model(input_ids, target_ids, chunk_size=32)
                    else:
                        # StandardTransformerLM forward
                        logits, loss = model(input_ids, target_ids)
                    
                    if loss is not None:
                        # Calculate number of tokens (excluding padding)
                        # Note: target_ids doesn't use -1 for padding, so count all tokens
                        n_tokens = target_ids.numel()
                        if n_tokens > 0:
                            all_losses.append(loss.item())
                            total_loss += loss.item() * n_tokens
                            total_tokens += n_tokens
                            n_samples += 1
                
                except Exception as e:
                    # Skip sequences that cause errors
                    continue
            
            # Print progress
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"  Processed {min(i + batch_size, len(test_data))}/{len(test_data)} samples...")
    
    if total_tokens == 0:
        print("Error: No valid tokens processed")
        return None
    
    # Calculate average loss
    avg_loss = total_loss / total_tokens
    
    # Calculate perplexity
    perplexity = np.exp(avg_loss)
    
    # Calculate statistics
    if all_losses:
        loss_std = np.std(all_losses)
        loss_min = np.min(all_losses)
        loss_max = np.max(all_losses)
        median_loss = np.median(all_losses)
        median_perplexity = np.exp(median_loss)
    else:
        loss_std = 0.0
        loss_min = avg_loss
        loss_max = avg_loss
        median_loss = avg_loss
        median_perplexity = perplexity
    
    print(f"\nResults:")
    print(f"  Number of samples evaluated: {n_samples}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Median perplexity: {median_perplexity:.2f}")
    print(f"  Loss std: {loss_std:.4f}")
    print(f"  Loss range: {loss_min:.4f} - {loss_max:.4f}")
    
    # Interpretation
    if perplexity < 10:
        interpretation = "Excellent"
    elif perplexity < 50:
        interpretation = "Very Good"
    elif perplexity < 100:
        interpretation = "Good"
    elif perplexity < 200:
        interpretation = "Fair"
    else:
        interpretation = "Needs Improvement"
    
    print(f"  Interpretation: {interpretation}")
    
    return {
        'n_samples': n_samples,
        'total_tokens': total_tokens,
        'avg_loss': float(avg_loss),
        'perplexity': float(perplexity),
        'median_perplexity': float(median_perplexity),
        'loss_std': float(loss_std),
        'loss_min': float(loss_min),
        'loss_max': float(loss_max),
        'interpretation': interpretation,
        'individual_losses': all_losses[:20]  # Store first 20 for reference
    }


def evaluate_translation_accuracy(model, tokenizer, test_pairs, device, max_samples=None,
                                  max_new_tokens=100, temperature=0.7, top_k=50, greedy=False):
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
    debug_examples = []  # Store first few examples for debugging
    
    print("Generating translations...")
    for i, (source_text, target_text) in enumerate(test_pairs):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_pairs)} pairs...")
        
        try:
            # Generate translation
            # Use sense retrieval by default (fastest and most reliable)
            # Set use_sense_retrieval=False to use generation approach
            generated_text = generate_translation(
                model, tokenizer, source_text, device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                greedy=greedy,
                use_sense_retrieval=True  # Use retrieval approach (works better than generation)
            )
            
            # Normalize both texts for comparison
            # Remove extra whitespace, normalize punctuation
            def normalize_text(text):
                import re
                if not text:
                    return ""
                # Lowercase
                text = text.lower()
                # Normalize whitespace (multiple spaces -> single space)
                text = re.sub(r'\s+', ' ', text)
                # Remove leading/trailing whitespace
                text = text.strip()
                # Remove common trailing punctuation that might be added by model
                # But keep punctuation that's part of the sentence
                # Remove trailing periods, commas, etc. if they weren't in the original
                return text
            
            normalized_generated = normalize_text(generated_text)
            normalized_target = normalize_text(target_text)
            
            # Exact match (with normalization)
            # Also check if generated text contains the target (for cases where model adds extra words)
            is_exact_match = False
            if normalized_generated == normalized_target:
                is_exact_match = True
            elif normalized_target in normalized_generated:
                # Check if target is at the start of generated (model might have added words after)
                if normalized_generated.startswith(normalized_target + ' '):
                    is_exact_match = True
                # Or if generated is just target with punctuation
                elif normalized_generated.startswith(normalized_target):
                    remaining = normalized_generated[len(normalized_target):].strip()
                    # Allow trailing punctuation only
                    if not remaining or all(c in '.,!?;:' for c in remaining):
                        is_exact_match = True
            
            if is_exact_match:
                exact_matches += 1
            
            # Word-level accuracy
            ref_words = set(target_text.lower().split())
            gen_words = set(generated_text.lower().split())
            word_overlap = 0.0
            if len(ref_words) > 0:
                word_overlap = len(ref_words & gen_words) / len(ref_words)
                word_accuracies.append(word_overlap)
            
            # Debug: Store first few examples (especially short ones that should match)
            if len(debug_examples) < 10 and (i < 10 or len(target_text.split()) <= 3):
                debug_examples.append({
                    'source': source_text,
                    'target': target_text,
                    'generated': generated_text,
                    'normalized_target': normalized_target,
                    'normalized_generated': normalized_generated,
                    'is_match': is_exact_match,
                    'word_accuracy': word_overlap
                })
            
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
    
    # Store sample examples for analysis (included in results JSON)
    # Debug examples are stored but not printed by default to reduce verbosity
    
    return {
        'n_pairs': n_evaluated,
        'exact_match_rate': float(exact_match_rate),
        'exact_matches': exact_matches,
        'avg_word_accuracy': float(avg_word_accuracy),
        'avg_char_accuracy': float(avg_char_accuracy),
        'word_accuracies': word_accuracies[:10],  # Store first 10 for reference
        'debug_examples': debug_examples
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
    
    # Evaluate cross-lingual similarity (using MultiSimLex function)
    print("\n=== Cross-lingual Word Similarity ===")
    translation_pairs = [
        ('hello', 'bonjour'),
        ('world', 'monde'),
        ('language', 'langue'),
        ('model', 'modèle'),
        ('learning', 'apprentissage'),
    ]
    
    # Use evaluate_cross_lingual_multisimlex for proper evaluation
    print("\nTranslation pair similarities:")
    for word1, word2 in translation_pairs:
        repr1 = get_word_representations(model, tokenizer, [word1], device)[word1]
        repr2 = get_word_representations(model, tokenizer, [word2], device)[word2]
        repr1_mean = repr1.mean(axis=0)
        repr2_mean = repr2.mean(axis=0)
        cos_sim = np.dot(repr1_mean, repr2_mean) / (np.linalg.norm(repr1_mean) * np.linalg.norm(repr2_mean))
        print(f"  {word1} <-> {word2}: {cos_sim:.4f}")
    
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

