"""
Prepare Europarl dataset for multilingual Backpack training

Europarl is a parallel corpus from European Parliament proceedings.
Available in multiple language pairs including English-French.
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_europarl_data(language_pair='en-fr'):
    """
    Download and prepare Europarl dataset for training.
    Creates train.bin and val.bin files with tokenized data.
    
    Args:
        language_pair: Language pair code (e.g., 'en-fr', 'en-de', etc.)
    """
    print(f"Loading Europarl dataset for {language_pair}...")
    
    # Try to load Europarl dataset
    try:
        # Option 1: Try loading from HuggingFace datasets
        dataset = load_dataset("europarl_bilingual", language_pair, split="train")
        print(f"Loaded Europarl {language_pair} from HuggingFace")
    except:
        # Option 2: Try OPUS dataset which includes Europarl
        try:
            dataset = load_dataset("opus100", language_pair, split="train")
            print(f"Using OPUS100 {language_pair} dataset (includes Europarl)")
        except:
            print("ERROR: Could not load Europarl or OPUS dataset.")
            print("Please install datasets and try again:")
            print("  pip install datasets")
            print("\nAlternatively, you can download Europarl manually from:")
            print("  https://www.statmt.org/europarl/")
            return
    
    print(f"Loaded {len(dataset)} parallel sentences")
    
    # Initialize multilingual tokenizer
    tokenizer_name = "xlm-roberta-base"  # Good multilingual tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Combine English and French sentences
    # For multilingual training, we'll interleave or concatenate sentences
    all_texts = []
    for item in tqdm(dataset, desc="Processing sentences"):
        # Get language codes
        lang1, lang2 = language_pair.split('-')
        
        # Extract sentences based on dataset structure
        if 'translation' in item:
            text1 = item['translation'].get(lang1, '')
            text2 = item['translation'].get(lang2, '')
        else:
            # Try direct keys
            text1 = item.get(lang1, item.get('en', ''))
            text2 = item.get(lang2, item.get('fr', ''))
        
        if text1 and text2:
            # Option 1: Interleave sentences with language separator
            combined = f"{text1} <|lang_sep|> {text2}"
            all_texts.append(combined)
            
            # Option 2: Also add reverse order for better learning
            combined_reverse = f"{text2} <|lang_sep|> {text1}"
            all_texts.append(combined_reverse)
    
    print(f"Total combined texts: {len(all_texts)}")
    
    # Tokenize all texts
    print("Tokenizing texts...")
    # Output binary files
    output_dir = os.path.dirname(__file__)
    train_filename = os.path.join(output_dir, 'train.bin')
    val_filename = os.path.join(output_dir, 'val.bin')


    # Split into train and validation
    n = len(all_texts)
    train_val_cutoff = int(n*0.9)

    with open(train_filename, 'wb') as f:
        for text in tqdm(all_texts[:train_val_cutoff], desc="Tokenizing"):
            tokens = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
            np.array(tokens, dtype=np.uint32).tofile(f)
            
    with open(val_filename, 'wb') as f:
        for text in tqdm(all_texts[train_val_cutoff:], desc="Tokenizing"):
            tokens = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
            np.array(tokens, dtype=np.uint32).tofile(f)
    
    # Save metadata
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'tokenizer_name': tokenizer_name,
        'languages': language_pair.split('-'),
        'language_pair': language_pair,
        'dataset': 'europarl'
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
    import argparse
    parser = argparse.ArgumentParser(description='Prepare Europarl dataset')
    parser.add_argument('--language_pair', type=str, default='en-fr', 
                       help='Language pair (e.g., en-fr, en-de, en-es)')
    args = parser.parse_args()
    prepare_europarl_data(args.language_pair)

