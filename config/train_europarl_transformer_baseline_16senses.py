"""
Configuration for training Standard Transformer baseline from scratch on Europarl dataset
Uses parameters from train_backpack_clean (matching backpack config but without senses)
"""

from configurator import ModelConfig

config = ModelConfig(
    # Model architecture (same as Backpack config)
    block_size=128,  # Smaller context for memory efficiency
    n_layer=4,  # 4 layers (from train_backpack_clean)
    n_head=4,  # 4 heads (from train_backpack_clean)
    n_embd=256,  # 256 embedding dim (from train_backpack_clean)
    # Note: n_senses is not used in StandardTransformerLM, but kept for compatibility
    n_senses=1,  # Not used, but set to 1 for config compatibility
    dropout=0.1,
    bias=False,
    
    # Training (same as Backpack config)
    batch_size=8,  # Smaller batch size for memory efficiency
    learning_rate=3e-4,
    max_iters=150000,  # Train as much as possible - can resume from checkpoints
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    
    # Evaluation (same as Backpack config)
    eval_interval=100,  # More frequent evaluation
    eval_iters=50,  # Fewer eval iterations
    log_interval=10,
    
    # System (same as Backpack config)
    device='cuda',
    dtype='float16',
    compile=False,  # Disable compile for memory efficiency
    
    # Data
    dataset='europarl',
    tokenizer_name='xlm-roberta-base',
    languages=['en', 'fr'],
)

