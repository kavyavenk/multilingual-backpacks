"""
Prepare Hansards dataset for multilingual Backpack training

Hansards is a French-English parallel corpus from Canadian parliamentary proceedings.
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_hansards_data():
    """
    Download and prepare Hansards dataset for training.
    Creates train.bin and val.bin files with tokenized data.
    """
    print("Loading Hansards dataset...")
    
    # Try to load Hansards dataset
    # Note: You may need to download this separately or use a different source
    try:
        # Option 1: Try loading from HuggingFace if available
        dataset = load_dataset("hansards", split="train")
    except:
        # Option 2: Use OPUS dataset which includes Hansards
        try:
            dataset = load_dataset("opus100", "en-fr", split="train")
            print("Using OPUS100 en-fr dataset as alternative to Hansards")
        except:
            print("ERROR: Could not load Hansards or OPUS dataset.")
            print("Please download the Hansards dataset manually or install opus100:")
            print("  pip install datasets opus100")
            print("\nAlternatively, you can use any French-English parallel corpus.")
            return
    
    print(f"Loaded {len(dataset)} parallel sentences")
    
    # Initialize multilingual tokenizer
    # Using a multilingual tokenizer that handles both English and French
    tokenizer_name = "xlm-roberta-base"  # Good multilingual tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Combine English and French sentences
    # For multilingual training, we'll interleave or concatenate sentences
    all_texts = []
    for item in tqdm(dataset, desc="Processing sentences"):
        # Combine both languages with a language separator token
        # This helps the model learn to distinguish languages
        en_text = item.get('translation', {}).get('en', item.get('en', ''))
        fr_text = item.get('translation', {}).get('fr', item.get('fr', ''))
        
        if en_text and fr_text:
            # Option 1: Interleave sentences
            combined = f"{en_text} <|lang_sep|> {fr_text}"
            all_texts.append(combined)
            
            # Option 2: Also add reverse order for better learning
            combined_reverse = f"{fr_text} <|lang_sep|> {en_text}"
            all_texts.append(combined_reverse)
    
    print(f"Total combined texts: {len(all_texts)}")
    
    # Tokenize all texts
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(all_texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_tokens.extend(tokens)
    
    print(f"Total tokens: {len(all_tokens):,}")
    
    # Split into train and validation
    n = len(all_tokens)
    train_data = all_tokens[:int(n * 0.9)]
    val_data = all_tokens[int(n * 0.9):]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    # Save as binary files
    output_dir = os.path.dirname(__file__)
    train_filename = os.path.join(output_dir, 'train.bin')
    val_filename = os.path.join(output_dir, 'val.bin')
    
    train_data = np.array(train_data, dtype=np.uint16)
    val_data = np.array(val_data, dtype=np.uint16)
    
    train_data.tofile(train_filename)
    val_data.tofile(val_filename)
    
    # Save metadata
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'tokenizer_name': tokenizer_name,
        'languages': ['en', 'fr'],
        'train_tokens': len(train_data),
        'val_tokens': len(val_data),
    }
    
    meta_filename = os.path.join(output_dir, 'meta.pkl')
    with open(meta_filename, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Saved train data to {train_filename}")
    print(f"Saved val data to {val_filename}")
    print(f"Saved metadata to {meta_filename}")
    print(f"\nVocabulary size: {meta['vocab_size']}")
    print("Data preparation complete!")


if __name__ == '__main__':
    prepare_hansards_data()

