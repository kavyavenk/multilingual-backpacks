"""
Configuration for training Backpack LM from scratch on Europarl dataset
Uses parameters from train_backpack_clean but with 16 senses instead of 4
"""

from configurator import ModelConfig

config = ModelConfig(
    # Model architecture (from train_backpack_clean)
    block_size=128,  # Smaller context for memory efficiency
    n_layer=4,  # 4 layers (from train_backpack_clean)
    n_head=4,  # 4 heads (from train_backpack_clean)
    n_embd=256,  # 256 embedding dim (from train_backpack_clean)
    n_senses=16,  # 16 sense vectors (increased from 4)
    dropout=0.1,
    bias=False,
    
    # Training (from train_backpack_clean)
    batch_size=8,  # Smaller batch size for memory efficiency
    learning_rate=3e-4,
    max_iters=50000,  # Train as much as possible - can resume from checkpoints
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    
    # Evaluation (from train_backpack_clean)
    eval_interval=100,  # More frequent evaluation
    eval_iters=50,  # Fewer eval iterations
    log_interval=10,
    
    # System (from train_backpack_clean)
    device='cuda',
    dtype='float16',
    compile=False,  # Disable compile for memory efficiency
    
    # Data
    dataset='europarl',
    tokenizer_name='xlm-roberta-base',
    languages=['en', 'fr'],
)

