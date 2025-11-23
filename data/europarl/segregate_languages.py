"""
Segregate parallel sentences from Europarl into separate language files with tags.

This script creates separate files for each language with sentence-level tags
that can be used for reference and analysis.
"""

import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset


def segregate_languages(language_pair='en-fr', output_dir=None):
    """
    Segregate parallel sentences into separate language files with tags.
    
    Creates:
    - {lang1}.txt: Sentences in first language with tags
    - {lang2}.txt: Sentences in second language with tags
    - metadata.json: Reference information about the dataset
    
    Tags format: <sentence_id>|<language_pair>|<source>
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading Europarl dataset for {language_pair}...")
    
    # Load dataset
    try:
        dataset = load_dataset("europarl_bilingual", language_pair, split="train")
    except:
        try:
            dataset = load_dataset("opus100", language_pair, split="train")
        except:
            print("ERROR: Could not load Europarl or OPUS dataset.")
            return
    
    lang1, lang2 = language_pair.split('-')
    
    print(f"Processing {len(dataset)} parallel sentences...")
    
    # Files for each language
    lang1_file = open(os.path.join(output_dir, f'{lang1}.txt'), 'w', encoding='utf-8')
    lang2_file = open(os.path.join(output_dir, f'{lang2}.txt'), 'w', encoding='utf-8')
    
    # Metadata
    metadata = {
        'language_pair': language_pair,
        'languages': [lang1, lang2],
        'total_sentences': len(dataset),
        'dataset': 'europarl',
        'tag_format': '<sentence_id>|<language_pair>|<source>',
    }
    
    # Process sentences
    for idx, item in enumerate(tqdm(dataset, desc="Segregating sentences")):
        # Extract sentences
        if 'translation' in item:
            text1 = item['translation'].get(lang1, '').strip()
            text2 = item['translation'].get(lang2, '').strip()
        else:
            text1 = item.get(lang1, item.get('en', '')).strip()
            text2 = item.get(lang2, item.get('fr', '')).strip()
        
        if text1 and text2:
            # Create tag: sentence_id|language_pair|source
            tag = f"{idx}|{language_pair}|europarl"
            
            # Write to respective language files
            lang1_file.write(f"{tag}\t{text1}\n")
            lang2_file.write(f"{tag}\t{text2}\n")
    
    lang1_file.close()
    lang2_file.close()
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nSegregation complete!")
    print(f"Language 1 ({lang1}) file: {os.path.join(output_dir, f'{lang1}.txt')}")
    print(f"Language 2 ({lang2}) file: {os.path.join(output_dir, f'{lang2}.txt')}")
    print(f"Metadata file: {metadata_file}")
    print(f"\nTotal sentences: {len(dataset)}")
    print(f"\nTag format: <sentence_id>|<language_pair>|<source>")
    print(f"Example tag: 0|en-fr|europarl")


def create_alignment_file(language_pair='en-fr', output_dir=None):
    """
    Create an alignment file that maps sentence IDs between languages.
    
    Creates:
    - alignment.json: Maps sentence IDs to parallel sentences
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    print(f"Creating alignment file for {language_pair}...")
    
    # Load dataset
    try:
        dataset = load_dataset("europarl_bilingual", language_pair, split="train")
    except:
        try:
            dataset = load_dataset("opus100", language_pair, split="train")
        except:
            print("ERROR: Could not load dataset.")
            return
    
    lang1, lang2 = language_pair.split('-')
    
    alignments = []
    
    for idx, item in enumerate(tqdm(dataset, desc="Creating alignments")):
        if 'translation' in item:
            text1 = item['translation'].get(lang1, '').strip()
            text2 = item['translation'].get(lang2, '').strip()
        else:
            text1 = item.get(lang1, item.get('en', '')).strip()
            text2 = item.get(lang2, item.get('fr', '')).strip()
        
        if text1 and text2:
            alignments.append({
                'id': idx,
                lang1: text1,
                lang2: text2,
                'language_pair': language_pair,
            })
    
    alignment_file = os.path.join(output_dir, 'alignment.json')
    with open(alignment_file, 'w', encoding='utf-8') as f:
        json.dump(alignments, f, indent=2, ensure_ascii=False)
    
    print(f"Alignment file created: {alignment_file}")
    print(f"Total aligned pairs: {len(alignments)}")


def main():
    parser = argparse.ArgumentParser(description='Segregate Europarl dataset by language')
    parser.add_argument('--language_pair', type=str, default='en-fr',
                       help='Language pair (e.g., en-fr, en-de, en-es)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as script)')
    parser.add_argument('--create_alignment', action='store_true',
                       help='Also create alignment JSON file')
    
    args = parser.parse_args()
    
    # Segregate languages
    segregate_languages(args.language_pair, args.output_dir)
    
    # Create alignment file if requested
    if args.create_alignment:
        create_alignment_file(args.language_pair, args.output_dir)


if __name__ == '__main__':
    main()

