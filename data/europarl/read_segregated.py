"""
Helper script to read and work with segregated language files.

This provides utilities for working with the tagged language files created by
segregate_languages.py for reference and analysis.
"""

import os
import json
from typing import Dict, List, Tuple, Optional


class SegregatedDataReader:
    """Reader for segregated language files with tags"""
    
    def __init__(self, data_dir: str, language_pair: str = 'en-fr'):
        """
        Initialize reader for segregated language files.
        
        Args:
            data_dir: Directory containing the language files
            language_pair: Language pair (e.g., 'en-fr')
        """
        self.data_dir = data_dir
        self.language_pair = language_pair
        lang1, lang2 = language_pair.split('-')
        self.lang1 = lang1
        self.lang2 = lang2
        
        self.lang1_file = os.path.join(data_dir, f'{lang1}.txt')
        self.lang2_file = os.path.join(data_dir, f'{lang2}.txt')
        self.metadata_file = os.path.join(data_dir, 'metadata.json')
        self.alignment_file = os.path.join(data_dir, 'alignment.json')
        
        # Load metadata if available
        self.metadata = None
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
    
    def read_language_file(self, language: str) -> Dict[str, str]:
        """
        Read sentences from a language file.
        
        Returns:
            Dict mapping sentence_id to sentence text
        """
        lang_file = os.path.join(self.data_dir, f'{language}.txt')
        if not os.path.exists(lang_file):
            raise FileNotFoundError(f"Language file not found: {lang_file}")
        
        sentences = {}
        with open(lang_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse tag and sentence
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    continue
                
                tag, sentence = parts
                sentence_id, lang_pair, source = tag.split('|')
                sentences[sentence_id] = sentence
        
        return sentences
    
    def get_parallel_pairs(self, limit: Optional[int] = None) -> List[Tuple[str, str, str]]:
        """
        Get parallel sentence pairs.
        
        Returns:
            List of (sentence_id, lang1_text, lang2_text) tuples
        """
        lang1_sentences = self.read_language_file(self.lang1)
        lang2_sentences = self.read_language_file(self.lang2)
        
        pairs = []
        for sent_id in lang1_sentences.keys():
            if sent_id in lang2_sentences:
                pairs.append((
                    sent_id,
                    lang1_sentences[sent_id],
                    lang2_sentences[sent_id]
                ))
                if limit and len(pairs) >= limit:
                    break
        
        return pairs
    
    def get_alignment_data(self) -> Optional[List[Dict]]:
        """Load alignment JSON file if available"""
        if os.path.exists(self.alignment_file):
            with open(self.alignment_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def search_by_id(self, sentence_id: str) -> Optional[Tuple[str, str]]:
        """Get parallel sentences by sentence ID"""
        lang1_sentences = self.read_language_file(self.lang1)
        lang2_sentences = self.read_language_file(self.lang2)
        
        if sentence_id in lang1_sentences and sentence_id in lang2_sentences:
            return (lang1_sentences[sentence_id], lang2_sentences[sentence_id])
        return None
    
    def get_statistics(self) -> Dict:
        """Get statistics about the dataset"""
        lang1_sentences = self.read_language_file(self.lang1)
        lang2_sentences = self.read_language_file(self.lang2)
        
        # Count sentences
        lang1_count = len(lang1_sentences)
        lang2_count = len(lang2_sentences)
        
        # Count aligned pairs
        aligned_ids = set(lang1_sentences.keys()) & set(lang2_sentences.keys())
        aligned_count = len(aligned_ids)
        
        # Average sentence lengths
        lang1_avg_len = sum(len(s) for s in lang1_sentences.values()) / lang1_count if lang1_count > 0 else 0
        lang2_avg_len = sum(len(s) for s in lang2_sentences.values()) / lang2_count if lang2_count > 0 else 0
        
        stats = {
            'language_pair': self.language_pair,
            'total_sentences_lang1': lang1_count,
            'total_sentences_lang2': lang2_count,
            'aligned_pairs': aligned_count,
            'avg_length_lang1': round(lang1_avg_len, 2),
            'avg_length_lang2': round(lang2_avg_len, 2),
        }
        
        if self.metadata:
            stats.update(self.metadata)
        
        return stats


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Read and analyze segregated language files')
    parser.add_argument('--data_dir', type=str, default='data/europarl',
                       help='Directory containing language files')
    parser.add_argument('--language_pair', type=str, default='en-fr',
                       help='Language pair')
    parser.add_argument('--show_pairs', type=int, default=5,
                       help='Number of parallel pairs to display')
    parser.add_argument('--sentence_id', type=str, default=None,
                       help='Get specific sentence by ID')
    
    args = parser.parse_args()
    
    reader = SegregatedDataReader(args.data_dir, args.language_pair)
    
    # Show statistics
    print("=== Dataset Statistics ===")
    stats = reader.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Show example pairs
    print(f"\n=== Example Parallel Pairs (first {args.show_pairs}) ===")
    pairs = reader.get_parallel_pairs(limit=args.show_pairs)
    for sent_id, lang1_text, lang2_text in pairs:
        print(f"\nSentence ID: {sent_id}")
        print(f"  {reader.lang1}: {lang1_text[:100]}...")
        print(f"  {reader.lang2}: {lang2_text[:100]}...")
    
    # Get specific sentence if requested
    if args.sentence_id:
        print(f"\n=== Sentence ID {args.sentence_id} ===")
        pair = reader.search_by_id(args.sentence_id)
        if pair:
            print(f"  {reader.lang1}: {pair[0]}")
            print(f"  {reader.lang2}: {pair[1]}")
        else:
            print(f"  Sentence ID {args.sentence_id} not found")


if __name__ == '__main__':
    main()

