"""
Test script to verify Backpack and Standard Transformer models work on GPU
Runs forward passes and checks outputs
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import BackpackLM, StandardTransformerLM
from configurator import get_config


def test_model_on_gpu(model_class, config_name, model_name):
    """Test a model on GPU"""
    print(f"\n{'='*70}")
    print(f"Testing {model_name}")
    print(f"{'='*70}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"✓ Using device: {device}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    
    # Load config
    config = get_config(config_name)
    config.vocab_size = 250002  # XLM-RoBERTa vocab size
    config.device = device
    
    print(f"\nConfig:")
    print(f"  Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd")
    print(f"  Block size: {config.block_size}")
    print(f"  Vocab size: {config.vocab_size:,}")
    if hasattr(config, 'n_senses'):
        print(f"  Sense vectors: {config.n_senses}")
    
    # Create model
    print(f"\nCreating model...")
    model = model_class(config)
    model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    print(f"\nCreating test input:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Random token IDs (within vocab range)
    input_ids = torch.randint(0, min(1000, config.vocab_size), (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, min(1000, config.vocab_size), (batch_size, seq_len), device=device)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Target shape: {target_ids.shape}")
    
    # Forward pass
    print(f"\nRunning forward pass...")
    try:
        with torch.no_grad():
            logits, loss = model(input_ids, target_ids)
        
        print(f"✓ Forward pass successful!")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Check output shapes
        expected_logits_shape = (batch_size, seq_len, config.vocab_size)
        if logits.shape == expected_logits_shape:
            print(f"✓ Logits shape correct: {logits.shape}")
        else:
            print(f"⚠ Logits shape mismatch: expected {expected_logits_shape}, got {logits.shape}")
        
        # Check loss is reasonable
        if loss.item() > 0 and loss.item() < 20:
            print(f"✓ Loss is reasonable: {loss.item():.4f}")
        else:
            print(f"⚠ Loss seems unusual: {loss.item():.4f}")
        
        # Memory usage
        if device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            print(f"\nGPU Memory:")
            print(f"  Allocated: {memory_allocated:.2f} GB")
            print(f"  Reserved: {memory_reserved:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model_class, config_name, model_name):
    """Test text generation"""
    print(f"\n{'='*70}")
    print(f"Testing Generation: {model_name}")
    print(f"{'='*70}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_config(config_name)
    config.vocab_size = 250002
    config.device = device
    
    model = model_class(config)
    model.to(device)
    model.eval()
    
    # Create context
    context = torch.randint(0, min(1000, config.vocab_size), (1, 5), device=device)
    print(f"Context shape: {context.shape}")
    
    try:
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=10, temperature=1.0)
        
        print(f"✓ Generation successful!")
        print(f"  Generated shape: {generated.shape}")
        print(f"  Generated tokens: {generated[0].tolist()[:15]}...")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False


def main():
    print("="*70)
    print("GPU MODEL FUNCTIONALITY TEST")
    print("="*70)
    
    # Test Backpack model
    backpack_ok = test_model_on_gpu(
        BackpackLM, 
        'train_europarl_scratch',
        'Backpack Model'
    )
    
    # Test Standard Transformer
    transformer_ok = test_model_on_gpu(
        StandardTransformerLM,
        'train_europarl_transformer_baseline',
        'Standard Transformer Baseline'
    )
    
    # Test generation for both
    print("\n" + "="*70)
    print("TESTING GENERATION")
    print("="*70)
    
    backpack_gen_ok = test_generation(
        BackpackLM,
        'train_europarl_scratch',
        'Backpack Model'
    )
    
    transformer_gen_ok = test_generation(
        StandardTransformerLM,
        'train_europarl_transformer_baseline',
        'Standard Transformer Baseline'
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Backpack Model:")
    print(f"  Forward pass: {'✓ PASS' if backpack_ok else '❌ FAIL'}")
    print(f"  Generation: {'✓ PASS' if backpack_gen_ok else '❌ FAIL'}")
    print(f"\nStandard Transformer:")
    print(f"  Forward pass: {'✓ PASS' if transformer_ok else '❌ FAIL'}")
    print(f"  Generation: {'✓ PASS' if transformer_gen_ok else '❌ FAIL'}")
    
    if backpack_ok and transformer_ok and backpack_gen_ok and transformer_gen_ok:
        print("\n🎉 All tests passed! Models are ready for training.")
    else:
        print("\n⚠ Some tests failed. Please check errors above.")
    
    print("="*70)


if __name__ == '__main__':
    main()

