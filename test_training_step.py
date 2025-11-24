"""
Test a full training step on GPU to verify everything works
"""

import torch
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import BackpackLM, StandardTransformerLM
from configurator import get_config


def test_training_step(model_class, config_name, model_name, device='cuda'):
    """Test a complete training step"""
    print(f"\n{'='*70}")
    print(f"Testing Training Step: {model_name}")
    print(f"{'='*70}")
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = 'cpu'
    else:
        print(f"✓ Using device: {device}")
        if device == 'cuda':
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # Load config
    config = get_config(config_name)
    config.vocab_size = 250002
    config.device = device
    config.batch_size = 4  # Small batch for testing
    config.block_size = 64  # Smaller context for testing
    
    print(f"\nTest Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Block size: {config.block_size}")
    print(f"  Architecture: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
    
    # Create model
    model = model_class(config)
    model.to(device)
    model.train()  # Set to training mode
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model created: {total_params:,} parameters ({total_params/1e6:.2f}M)")
    
    # Create optimizer
    if hasattr(model, 'configure_optimizers'):
        optimizer = model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=device
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    print(f"✓ Optimizer created: {type(optimizer).__name__}")
    
    # Create dummy batch
    batch_size = config.batch_size
    seq_len = config.block_size
    
    X = torch.randint(0, min(1000, config.vocab_size), (batch_size, seq_len), device=device)
    Y = torch.randint(0, min(1000, config.vocab_size), (batch_size, seq_len), device=device)
    
    print(f"\n✓ Created batch:")
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {Y.shape}")
    
    # Training step
    print(f"\nRunning training step...")
    try:
        # Forward pass
        logits, loss = model(X, Y)
        print(f"✓ Forward pass: loss = {loss.item():.4f}")
        print(f"  Logits shape: {logits.shape}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        print(f"✓ Backward pass completed")
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        print(f"✓ Gradient clipping: norm = {grad_norm.item():.4f}")
        
        # Optimizer step
        optimizer.step()
        print(f"✓ Optimizer step completed")
        
        # Check memory
        if device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\nGPU Memory:")
            print(f"  Allocated: {memory_allocated:.3f} GB")
            print(f"  Reserved: {memory_reserved:.3f} GB")
        
        # Second forward pass (to verify training mode works)
        logits2, loss2 = model(X, Y)
        print(f"\n✓ Second forward pass: loss = {loss2.item():.4f}")
        print(f"  Loss changed: {abs(loss2.item() - loss.item()):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("TRAINING STEP FUNCTIONALITY TEST")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Test Backpack
    backpack_ok = test_training_step(
        BackpackLM,
        'train_europarl_scratch',
        'Backpack Model',
        device=device
    )
    
    # Test Transformer
    transformer_ok = test_training_step(
        StandardTransformerLM,
        'train_europarl_transformer_baseline',
        'Standard Transformer Baseline',
        device=device
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Backpack Model: {'✓ PASS' if backpack_ok else '❌ FAIL'}")
    print(f"Standard Transformer: {'✓ PASS' if transformer_ok else '❌ FAIL'}")
    
    if backpack_ok and transformer_ok:
        print("\n🎉 All training steps work correctly!")
        print("✓ Models are ready for full training runs")
    else:
        print("\n⚠ Some tests failed")
    
    print("="*70)


if __name__ == '__main__':
    main()

