"""
Configuration for training Backpack LM from scratch on Europarl dataset
"""

from configurator import ModelConfig

config = ModelConfig(
    # Model architecture
    block_size=512,  # Smaller context for faster training
    n_layer=6,  # Smaller model for faster iteration
    n_head=6,
    n_embd=384,
    n_senses=16,
    dropout=0.1,
    bias=False,
    
    # Training
    batch_size=32,  # Adjust based on GPU memory
    learning_rate=3e-4,
    max_iters=10000,  # Adjust based on convergence
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

