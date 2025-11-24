"""
Plot loss curves from training log JSON file
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_log(log_file):
    """Load training log from JSON file"""
    with open(log_file, 'r') as f:
        log = json.load(f)
    return log


def plot_loss_curves(log_file, output_file=None, show_plot=True):
    """
    Plot training and validation loss curves
    
    Args:
        log_file: Path to training_log.json
        output_file: Optional path to save figure
        show_plot: Whether to display the plot
    """
    log = load_training_log(log_file)
    
    iterations = log['iterations']
    train_loss = log['train_loss']
    val_loss = log['val_loss']
    
    if len(iterations) == 0:
        print("No training data found in log file")
        return
    
    # Create fig
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot curves
    ax.plot(iterations, train_loss, label='Train Loss', linewidth=2, alpha=0.8)
    ax.plot(iterations, val_loss, label='Validation Loss', linewidth=2, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add some relevant stats
    if len(train_loss) > 0:
        final_train = train_loss[-1]
        final_val = val_loss[-1]
        min_val_idx = np.argmin(val_loss)
        min_val_iter = iterations[min_val_idx]
        min_val_loss = val_loss[min_val_idx]
        
        stats_text = f'Final Train Loss: {final_train:.4f}\n'
        stats_text += f'Final Val Loss: {final_val:.4f}\n'
        stats_text += f'Best Val Loss: {min_val_loss:.4f} (iter {min_val_iter})'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save when we have output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved loss curves to: {output_file}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_multiple_curves(log_files, labels=None, output_file=None, show_plot=True):
    """
    Plot loss curves from multiple training runs for comparison
    
    Args:
        log_files: List of paths to training_log.json files
        labels: Optional list of labels for each run
        output_file: Optional path to save figure
        show_plot: Whether to display the plot
    """
    if labels is None:
        labels = [Path(f).parent.name for f in log_files]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for log_file, label in zip(log_files, labels):
        log = load_training_log(log_file)
        iterations = log['iterations']
        train_loss = log['train_loss']
        val_loss = log['val_loss']
        
        axes[0].plot(iterations, train_loss, label=label, linewidth=2, alpha=0.8)
        axes[1].plot(iterations, val_loss, label=label, linewidth=2, alpha=0.8)
    
    # Formatting
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Train Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot loss curves from training log')
    parser.add_argument('--log_file', type=str, required=True, help='Path to training_log.json')
    parser.add_argument('--output', type=str, default=None, help='Output file path (e.g., loss_curves.pdf)')
    parser.add_argument('--compare', nargs='+', default=None, help='Additional log files to compare')
    parser.add_argument('--labels', nargs='+', default=None, help='Labels for comparison plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    
    args = parser.parse_args()
    
    if args.compare:
        # Multiple files for comparison
        all_files = [args.log_file] + args.compare
        all_labels = args.labels if args.labels else None
        plot_multiple_curves(all_files, all_labels, args.output, show_plot=not args.no_show)
    else:
        # Single file
        plot_loss_curves(args.log_file, args.output, show_plot=not args.no_show)


if __name__ == '__main__':
    main()