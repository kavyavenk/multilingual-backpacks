"""
Analyze top activating words from training log
Shows which words have highest sense vector activations
"""

import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path


def load_training_log(log_file):
    """Load training log from JSON file"""
    with open(log_file, 'r') as f:
        log = json.load(f)
    return log


def analyze_top_activating_words(log_file, top_n=20):
    """
    Analyze top activating words across training
    
    Args:
        log_file: Path to training_log.json
        top_n: Number of top words to show
    """
    log = load_training_log(log_file)
    
    if 'top_activating_words' not in log or len(log['top_activating_words']) == 0:
        print("No top activating words data found in log file")
        return
    
    # Aggregate words across all iterations
    word_activations = defaultdict(list)
    word_counts = Counter()
    
    for entry in log['top_activating_words']:
        iteration = entry['iteration']
        for word_info in entry['words']:
            word = word_info['word']
            activation = word_info['activation']
            word_activations[word].append(activation)
            word_counts[word] += 1
    
    # Compute average activation for each word
    avg_activations = {}
    for word, activations in word_activations.items():
        avg_activations[word] = sum(activations) / len(activations)
    
    # Sort by average activation
    sorted_words = sorted(avg_activations.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"Top {top_n} Activating Words (by average sense weight)")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Word':<30} {'Avg Activation':<15} {'Count':<10}")
    print(f"{'-'*60}")
    
    for rank, (word, avg_act) in enumerate(sorted_words[:top_n], 1):
        count = word_counts[word]
        print(f"{rank:<6} {word[:28]:<30} {avg_act:<15.6f} {count:<10}")
    
    return sorted_words[:top_n]


def plot_activation_evolution(log_file, top_n=10, output_file=None, show_plot=True):
    """
    Plot how activation of top words evolves over training
    
    Args:
        log_file: Path to training_log.json
        top_n: Number of top words to track
        output_file: Optional path to save figure
        show_plot: Whether to display the plot
    """
    log = load_training_log(log_file)
    
    if 'top_activating_words' not in log or len(log['top_activating_words']) == 0:
        print("No top activating words data found in log file")
        return
    
    # Get top words overall
    word_activations_all = defaultdict(list)
    for entry in log['top_activating_words']:
        for word_info in entry['words']:
            word = word_info['word']
            activation = word_info['activation']
            word_activations_all[word].append(activation)
    
    avg_activations = {word: sum(acts)/len(acts) for word, acts in word_activations_all.items()}
    top_words = sorted(avg_activations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_word_set = {word for word, _ in top_words}
    
    # Track evolution
    evolution = defaultdict(list)
    iterations = []
    
    for entry in log['top_activating_words']:
        iteration = entry['iteration']
        iterations.append(iteration)
        
        word_dict = {w['word']: w['activation'] for w in entry['words']}
        
        for word in top_word_set:
            if word in word_dict:
                evolution[word].append(word_dict[word])
            else:
                evolution[word].append(0.0)  # Not in top-k at this iteration
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for word, activations in evolution.items():
        ax.plot(iterations, activations, label=word, linewidth=2, alpha=0.7, marker='o', markersize=3)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Sense Weight Activation', fontsize=12)
    ax.set_title(f'Top {top_n} Activating Words Evolution', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved activation evolution plot to: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze top activating words from training log')
    parser.add_argument('--log_file', type=str, required=True, help='Path to training_log.json')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top words to show')
    parser.add_argument('--plot_evolution', action='store_true', help='Plot activation evolution')
    parser.add_argument('--output', type=str, default=None, help='Output file for evolution plot')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    
    args = parser.parse_args()
    
    # Print top activating words
    analyze_top_activating_words(args.log_file, args.top_n)
    
    # Plot evolution if requested
    if args.plot_evolution:
        plot_activation_evolution(args.log_file, args.top_n, args.output, show_plot=not args.no_show)


if __name__ == '__main__':
    main()

