"""
Configuration for training Standard Transformer baseline from scratch on Europarl dataset
This is identical to train_europarl_scratch but will use StandardTransformerLM instead of BackpackLM
"""

from configurator import ModelConfig

config = ModelConfig(
    # Model architecture (same as Backpack scratch config)
    block_size=512,  # Smaller context for faster training
    n_layer=6,  # Smaller model for faster iteration
    n_head=6,
    n_embd=384,
    # Note: n_senses is not used in StandardTransformerLM, but kept for compatibility
    n_senses=1,  # Not used, but set to 1 for config compatibility
    dropout=0.1,
    bias=False,
    
    # Training (same as Backpack scratch config)
    batch_size=32,  # Adjust based on GPU memory
    learning_rate=3e-4,
    max_iters=50000,  # Train as much as possible - can resume from checkpoints
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    
    # Evaluation
    eval_interval=500,
    eval_iters=200,
    log_interval=10,
    
    # System
    device='cuda',
    dtype='float16',
    compile=True,
    
    # Data
    dataset='europarl',
    tokenizer_name='xlm-roberta-base',
    languages=['en', 'fr'],
)

