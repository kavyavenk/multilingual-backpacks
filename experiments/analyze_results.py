"""
Analysis script for comparing results across different training runs
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_training_logs(out_dir):
    """Load training logs if available"""
    log_file = os.path.join(out_dir, 'training_log.json')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    return None


def compare_models(model_dirs, metrics=['val_loss', 'train_loss']):
    """Compare metrics across different model checkpoints"""
    results = {}
    
    for model_dir in model_dirs:
        logs = load_training_logs(model_dir)
        if logs:
            results[model_dir] = logs
        else:
            # Try to load checkpoint and extract final metrics
            ckpt_path = os.path.join(model_dir, 'ckpt.pt')
            if os.path.exists(ckpt_path):
                import torch
                ckpt = torch.load(ckpt_path, map_location='cpu')
                results[model_dir] = {
                    'best_val_loss': ckpt.get('best_val_loss', None),
                    'iter_num': ckpt.get('iter_num', None),
                }
    
    return results


def plot_training_curves(results, output_file='training_comparison.png'):
    """Plot training curves for comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for model_name, logs in results.items():
        if 'train_loss' in logs and 'val_loss' in logs:
            iters = logs.get('iterations', [])
            train_losses = logs['train_loss']
            val_losses = logs['val_loss']
            
            axes[0].plot(iters, train_losses, label=f'{model_name} (train)')
            axes[1].plot(iters, val_losses, label=f'{model_name} (val)')
    
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Val Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")


def analyze_sense_alignment(sense_results_en, sense_results_fr):
    """
    Analyze if sense vectors align across languages.
    
    Args:
        sense_results_en: Dict of {word: [sense_predictions]}
        sense_results_fr: Dict of {word: [sense_predictions]}
    """
    print("\n=== Sense Vector Alignment Analysis ===")
    
    # For each translation pair, check if senses predict similar words
    # This is a simplified analysis - in practice, you'd want more sophisticated metrics
    
    alignment_scores = []
    
    for word_en, word_fr in zip(sense_results_en.keys(), sense_results_fr.keys()):
        en_senses = sense_results_en[word_en]
        fr_senses = sense_results_fr[word_fr]
        
        # Compare top predictions for each sense
        alignments = []
        for sense_idx in range(min(len(en_senses), len(fr_senses))):
            en_preds = set(en_senses[sense_idx])
            fr_preds = set(fr_senses[sense_idx])
            
            # Simple overlap metric
            overlap = len(en_preds & fr_preds) / max(len(en_preds | fr_preds), 1)
            alignments.append(overlap)
        
        avg_alignment = np.mean(alignments) if alignments else 0
        alignment_scores.append((word_en, word_fr, avg_alignment))
        
        print(f"{word_en} <-> {word_fr}: avg alignment = {avg_alignment:.3f}")
    
    return alignment_scores


def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--model_dirs', nargs='+', required=True, help='Model output directories to compare')
    parser.add_argument('--output_dir', type=str, default='analysis', help='Output directory for analysis')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compare models
    print("Comparing models...")
    results = compare_models(args.model_dirs)
    
    print("\n=== Model Comparison ===")
    for model_dir, metrics in results.items():
        print(f"\n{model_dir}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    # Plot training curves if available
    if any('train_loss' in r for r in results.values()):
        plot_training_curves(results, os.path.join(args.output_dir, 'training_comparison.png'))
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

