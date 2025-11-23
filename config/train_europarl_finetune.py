"""
Configuration for finetuning pre-trained Backpack LM on Europarl dataset
"""

from configurator import ModelConfig

config = ModelConfig(
    # Model architecture (should match pretrained model)
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    n_senses=16,
    dropout=0.1,
    bias=False,
    
    # Training (lower learning rate for finetuning)
    batch_size=16,  # Smaller batch for larger model
    learning_rate=1e-5,  # Lower LR for finetuning
    max_iters=5000,  # Fewer iterations for finetuning
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    
    # Evaluation
    eval_interval=250,
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

