"""
Tiny configuration for training Backpack LM (~500K parameters)
For quick experiments and loss curve demonstrations

Note: Actual parameter count depends on vocab_size.
To get ~500K params, use a smaller vocab (e.g., subset of tokens)
or reduce n_embd/n_senses further.
"""

from configurator import ModelConfig

config = ModelConfig(
    # Model architecture - tiny model (~500K params)
    # Note: Parameter count = vocab_size * n_embd * n_senses (sense_embeddings) + other layers
    # For ~500K with vocab_size ~10K, need: n_embd * n_senses ~ 50
    block_size=128,  # Very small context
    n_layer=2,  # Very few layers
    n_head=2,  # Minimal attention heads
    n_embd=48,  # Very small embedding dimension
    n_senses=4,  # Minimal senses (n_embd * n_senses = 192 per vocab token)
    dropout=0.1,
    bias=False,
    
    # Training
    batch_size=16,  # Smaller batch
    learning_rate=3e-4,
    max_iters=5000,  # Fewer iterations for quick training
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    
    # Evaluation
    eval_interval=100,  # More frequent evaluation for better curves
    eval_iters=50,  # Fewer eval iterations for speed
    log_interval=10,
    
    # System
    device='cuda',
    dtype='float16',
    compile=False,  # Disable compile for tiny model
    
    # Data
    dataset='europarl',
    tokenizer_name='xlm-roberta-base',
    languages=['en', 'fr'],
)

