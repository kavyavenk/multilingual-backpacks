#!/usr/bin/env python3
"""
Analyze what each of the 16 senses represents by examining predictions across multiple words.
"""

import argparse
import torch
from collections import defaultdict, Counter
from evaluate import load_model, analyze_sense_vectors
from transformers import AutoTokenizer


def analyze_sense_representations(model, tokenizer, words, device, top_k=10):
    """
    Analyze what each sense index (0-15) typically represents.
    
    Returns a dictionary mapping sense index to common predictions.
    """
    print("="*70)
    print("ANALYZING SENSE REPRESENTATIONS")
    print("="*70)
    print(f"\nAnalyzing {len(words)} words to identify sense patterns...\n")
    
    # Collect predictions for each sense index across all words
    sense_predictions = defaultdict(list)  # sense_idx -> list of (word, token, prob)
    
    results = analyze_sense_vectors(model, tokenizer, words, device, top_k=top_k, verbose=False, filter_tokens=True)
    
    for word, word_data in results.items():
        predictions = word_data['predictions']
        for sense_idx, sense_data in enumerate(predictions):
            for token, prob in zip(sense_data['tokens'], sense_data['probs']):
                sense_predictions[sense_idx].append((word, token, prob))
    
    # Analyze each sense
    print("="*70)
    print("SENSE REPRESENTATION ANALYSIS")
    print("="*70)
    
    sense_summaries = {}
    
    for sense_idx in sorted(sense_predictions.keys()):
        predictions = sense_predictions[sense_idx]
        
        # Count most common tokens across all words
        token_counter = Counter()
        word_token_pairs = []
        
        for word, token, prob in predictions:
            token_counter[token] += prob  # Weight by probability
            word_token_pairs.append((word, token, prob))
        
        # Get top tokens for this sense
        top_tokens = token_counter.most_common(15)
        
        # Find words where this sense is most active (highest probability)
        sense_activations = defaultdict(float)
        for word, word_data in results.items():
            predictions = word_data['predictions']
            if sense_idx < len(predictions):
                sense_activations[word] = predictions[sense_idx]['top_prob']
        
        top_words = sorted(sense_activations.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\n{'='*70}")
        print(f"SENSE {sense_idx}")
        print(f"{'='*70}")
        
        print(f"\nMost Common Predictions (across all words):")
        for i, (token, count) in enumerate(top_tokens[:10], 1):
            print(f"  {i:2d}. {token:20s} (weighted count: {count:.3f})")
        
        print(f"\nWords where this sense is most active:")
        for word, prob in top_words:
            print(f"  {word:15s} (top prob: {prob*100:.2f}%)")
        
        # Try to identify the semantic category
        top_token_strs = [token for token, _ in top_tokens[:5]]
        
        # Categorize based on common patterns
        category = "General"
        if any('sep' in t.lower() or 'lang' in t.lower() for t in top_token_strs):
            category = "Language/Separator"
        elif any(t in ['parlement', 'parliament', 'débat', 'debate'] for t in top_token_strs):
            category = "Parliament/Political"
        elif any(t in ['bonjour', 'hello', 'salut', 'hi'] for t in top_token_strs):
            category = "Greetings"
        elif any(t in ['monde', 'world', 'terre', 'earth'] for t in top_token_strs):
            category = "World/Geography"
        elif any(t in ['proposition', 'proposal', 'suggestion'] for t in top_token_strs):
            category = "Proposals/Actions"
        elif any(t in ['soutenir', 'support', 'aide', 'help'] for t in top_token_strs):
            category = "Support/Actions"
        elif any(t in ['risque', 'risk', 'danger', 'danger'] for t in top_token_strs):
            category = "Risk/Concern"
        elif any(t in ['manque', 'lack', 'absence'] for t in top_token_strs):
            category = "Absence/Lack"
        elif any(t in ['seul', 'alone', 'only', 'seulement'] for t in top_token_strs):
            category = "Isolation/Exclusivity"
        elif any(t in ['avenir', 'future', 'futur'] for t in top_token_strs):
            category = "Time/Future"
        elif any(t in ['côté', 'side', 'part'] for t in top_token_strs):
            category = "Position/Aspect"
        elif any(t in ['proc', 'process', 'processus'] for t in top_token_strs):
            category = "Process/Procedure"
        elif any(t in ['partage', 'share', 'sharing'] for t in top_token_strs):
            category = "Sharing/Distribution"
        elif any(t in ['progrès', 'progress', 'avancement'] for t in top_token_strs):
            category = "Progress/Development"
        
        print(f"\nInferred Category: {category}")
        
        sense_summaries[sense_idx] = {
            'top_tokens': top_tokens[:10],
            'top_words': top_words,
            'category': category
        }
    
    # Summary table
    print(f"\n{'='*70}")
    print("SENSE SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n{'Sense':<8} {'Category':<25} {'Top Token':<20}")
    print("-" * 70)
    for sense_idx in sorted(sense_summaries.keys()):
        summary = sense_summaries[sense_idx]
        top_token = summary['top_tokens'][0][0] if summary['top_tokens'] else 'N/A'
        print(f"{sense_idx:<8} {summary['category']:<25} {top_token:<20}")
    
    return sense_summaries


def main():
    parser = argparse.ArgumentParser(description='Analyze what each sense represents')
    parser.add_argument('--out_dir', type=str, default='out/backpack_full',
                       help='Directory containing trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top predictions to analyze per sense')
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
    
    # Analyze a diverse set of words
    test_words = [
        # Greetings
        'hello', 'bonjour', 'hi', 'salut',
        # World/Geography
        'world', 'monde', 'earth', 'terre',
        # Parliament/Political
        'parliament', 'parlement', 'debate', 'débat',
        # Support/Actions
        'support', 'soutenir', 'help', 'aide',
        # Proposals
        'proposal', 'proposition', 'suggestion',
        # Common words
        'language', 'langue', 'model', 'modèle',
        'learning', 'apprentissage', 'data', 'données',
        'system', 'système', 'government', 'gouvernement',
        'people', 'peuple', 'country', 'pays',
        'time', 'temps', 'year', 'année',
        'work', 'travail', 'project', 'projet',
        'important', 'important', 'necessary', 'nécessaire',
        'problem', 'problème', 'solution', 'solution',
        'change', 'changement', 'development', 'développement',
    ]
    
    sense_summaries = analyze_sense_representations(
        model, tokenizer, test_words, args.device, top_k=args.top_k
    )
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    # Save results
    import json
    output_file = f"{args.out_dir}/sense_representations.json"
    
    # Convert to JSON-serializable format
    json_output = {}
    for sense_idx, summary in sense_summaries.items():
        json_output[f"sense_{sense_idx}"] = {
            'category': summary['category'],
            'top_tokens': [(token, float(count)) for token, count in summary['top_tokens']],
            'top_words': [(word, float(prob)) for word, prob in summary['top_words']]
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
