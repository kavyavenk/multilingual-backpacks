"""
Visualize sense vectors using dimensionality reduction (PCA/t-SNE)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from evaluate import load_model, get_word_representations
from transformers import AutoTokenizer
import torch


def visualize_sense_vectors(model, tokenizer, words, device, method='pca', n_components=2):
    """
    Visualize sense vectors in 2D space.
    
    Args:
        method: 'pca' or 'tsne'
    """
    # Get representations
    reprs = get_word_representations(model, tokenizer, words, device)
    
    # Flatten sense vectors: (n_words, n_senses, n_embd) -> (n_words * n_senses, n_embd)
    all_vectors = []
    word_labels = []
    sense_labels = []
    
    for word, vectors in reprs.items():
        for sense_idx in range(vectors.shape[0]):
            all_vectors.append(vectors[sense_idx])
            word_labels.append(word)
            sense_labels.append(f"{word}_sense{sense_idx}")
    
    all_vectors = np.array(all_vectors)
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(all_vectors)-1))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Reducing dimensions using {method.upper()}...")
    reduced = reducer.fit_transform(all_vectors)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by word
    unique_words = list(set(word_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_words)))
    word_to_color = {word: colors[i] for i, word in enumerate(unique_words)}
    
    for i, (word, sense_label) in enumerate(zip(word_labels, sense_labels)):
        color = word_to_color[word]
        ax.scatter(reduced[i, 0], reduced[i, 1], c=[color], label=word if i < len(unique_words) else "", alpha=0.7)
        ax.annotate(sense_label.split('_')[-1], (reduced[i, 0], reduced[i, 1]), fontsize=8)
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Sense Vector Visualization ({method.upper()})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_multilingual_senses(model, tokenizer, word_pairs, device, method='pca'):
    """
    Compare sense vectors for translation pairs.
    """
    en_words = [pair[0] for pair in word_pairs]
    fr_words = [pair[1] for pair in word_pairs]
    
    # Get representations
    en_reprs = get_word_representations(model, tokenizer, en_words, device)
    fr_reprs = get_word_representations(model, tokenizer, fr_words, device)
    
    # Combine and visualize
    all_vectors = []
    labels = []
    
    for word_en, word_fr in word_pairs:
        en_vecs = en_reprs[word_en]  # (n_senses, n_embd)
        fr_vecs = fr_reprs[word_fr]  # (n_senses, n_embd)
        
        for sense_idx in range(en_vecs.shape[0]):
            all_vectors.append(en_vecs[sense_idx])
            labels.append(f"{word_en}_en_s{sense_idx}")
        
        for sense_idx in range(fr_vecs.shape[0]):
            all_vectors.append(fr_vecs[sense_idx])
            labels.append(f"{word_fr}_fr_s{sense_idx}")
    
    all_vectors = np.array(all_vectors)
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_vectors)-1))
    
    reduced = reducer.fit_transform(all_vectors)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by language and word pair
    for i, label in enumerate(labels):
        if '_en_' in label:
            color = 'blue'
            marker = 'o'
            alpha = 0.6
        else:
            color = 'red'
            marker = 's'
            alpha = 0.6
        
        ax.scatter(reduced[i, 0], reduced[i, 1], c=color, marker=marker, alpha=alpha, s=100)
        ax.annotate(label.split('_')[0], (reduced[i, 0], reduced[i, 1]), fontsize=8)
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title('Multilingual Sense Vector Comparison')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='English'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='French')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize sense vectors')
    parser.add_argument('--out_dir', type=str, required=True, help='Model output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--tokenizer_name', type=str, default='xlm-roberta-base', help='Tokenizer name')
    parser.add_argument('--method', type=str, default='pca', choices=['pca', 'tsne'], help='Dimensionality reduction method')
    parser.add_argument('--output', type=str, default='sense_visualization.png', help='Output file')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    # Load model
    print(f"Loading model from {args.out_dir}...")
    model, config = load_model(args.out_dir, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Test words
    english_words = ['hello', 'world', 'language', 'model']
    french_words = ['bonjour', 'monde', 'langue', 'modèle']
    
    # Visualize English words
    print("Visualizing English sense vectors...")
    fig1 = visualize_sense_vectors(model, tokenizer, english_words, device, method=args.method)
    fig1.savefig(args.output.replace('.png', '_english.png'))
    print(f"Saved to {args.output.replace('.png', '_english.png')}")
    
    # Visualize French words
    print("Visualizing French sense vectors...")
    fig2 = visualize_sense_vectors(model, tokenizer, french_words, device, method=args.method)
    fig2.savefig(args.output.replace('.png', '_french.png'))
    print(f"Saved to {args.output.replace('.png', '_french.png')}")
    
    # Compare multilingual
    print("Comparing multilingual sense vectors...")
    word_pairs = list(zip(english_words, french_words))
    fig3 = compare_multilingual_senses(model, tokenizer, word_pairs, device, method=args.method)
    fig3.savefig(args.output.replace('.png', '_multilingual.png'))
    print(f"Saved to {args.output.replace('.png', '_multilingual.png')}")
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()

