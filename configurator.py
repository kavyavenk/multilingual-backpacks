"""
Configuration management for Backpack LM training
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    # Model architecture
    block_size: int = 1024  # Context length
    vocab_size: int = None  # Will be set based on tokenizer
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_senses: int = 16  # Number of sense vectors per word
    dropout: float = 0.1
    bias: bool = False  # Use bias in LayerNorm and Linear layers
    
    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 200
    log_interval: int = 10
    
    # System
    device: str = 'cuda'
    dtype: str = 'float16'  # 'float32' or 'bfloat16' or 'float16'
    compile: bool = True  # Use torch.compile
    
    # Data
    dataset: str = 'hansards'
    tokenizer_name: str = 'multilingual'  # 'multilingual' or specific tokenizer
    
    # Multilingual specific
    languages: list = None  # Will be set to ['en', 'fr'] for hansards
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en', 'fr']


def get_config(config_name: str = 'default') -> ModelConfig:
    """Load configuration from file or return default"""
    config_path = f'config/{config_name}.py'
    if os.path.exists(config_path):
        # Load from Python config file
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.config
    else:
        # Return default config
        return ModelConfig()


def save_config(config: ModelConfig, path: str):
    """Save configuration to JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(asdict(config), f, indent=2)


def load_config(path: str) -> ModelConfig:
    """Load configuration from JSON file"""
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return ModelConfig(**config_dict)

