"""
Quick training script for tiny Backpack model
Can run on CPU or GPU
"""

import sys
import os

if __name__ == '__main__':
    print("="*70)
    print("TRAINING TINY BACKPACK MODEL")
    print("="*70)
    print(f"Parameters: ~500K")
    print(f"Dataset: Europarl (en-fr)")
    print(f"Device: cpu")
    print(f"Iterations: 1000")
    print("="*70)
    print()
    
    # Set up arguments for train.py
    sys.argv = [
        'train.py',
        '--config', 'train_europarl_tiny',
        '--out_dir', 'out/tiny',
        '--data_dir', 'europarl',
        '--device', 'cpu',
        '--dtype', 'float32',
        '--model_type', 'backpack'
    ]
    
    # Import and run train.py main
    from train import main
    main()
