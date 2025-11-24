"""
Calculate exact parameter count for Backpack models
Verifies parameter counts for different configurations
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from model import BackpackLM
from configurator import get_config


def calculate_backpack_params(config):
    """
    Calculate exact parameter count for Backpack model.
    
    Returns:
        dict: Detailed breakdown of parameters by component
    """
    # Create model to get actual parameter count
    model = BackpackLM(config)
    
    # Get total count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Detailed breakdown
    breakdown = {}
    
    # Sense embeddings
    breakdown['sense_embeddings'] = model.sense_embeddings.weight.numel()
    
    # Position embeddings
    breakdown['position_embeddings'] = model.pos_embeddings.weight.numel()
    
    # Sense predictor
    sense_pred_params = sum(p.numel() for p in model.sense_predictor.parameters())
    breakdown['sense_predictor'] = sense_pred_params
    
    # Transformer blocks
    block_params = sum(p.numel() for p in model.blocks.parameters())
    breakdown['transformer_blocks'] = block_params
    
    # Final layer norm
    breakdown['final_layernorm'] = model.ln_f.weight.numel()
    
    # LM head
    breakdown['lm_head'] = model.lm_head.weight.numel()
    
    # Verify sum matches
    breakdown['total_calculated'] = sum(breakdown.values())
    breakdown['total_actual'] = total_params
    breakdown['trainable'] = trainable_params
    
    return breakdown, total_params


def print_parameter_report(config_name='train_europarl_scratch'):
    """Print detailed parameter report for a configuration"""
    config = get_config(config_name)
    
    # Set vocab size if not set
    if config.vocab_size is None:
        # XLM-RoBERTa-base vocab size
        config.vocab_size = 250002
    
    print("=" * 70)
    print(f"PARAMETER REPORT: {config_name}")
    print("=" * 70)
    print(f"\nArchitecture Configuration:")
    print(f"  block_size:     {config.block_size}")
    print(f"  n_layer:        {config.n_layer}")
    print(f"  n_head:         {config.n_head}")
    print(f"  n_embd:         {config.n_embd}")
    print(f"  n_senses:       {config.n_senses}")
    print(f"  vocab_size:     {config.vocab_size:,}")
    print(f"  dropout:        {config.dropout}")
    print(f"  bias:           {config.bias}")
    
    breakdown, total = calculate_backpack_params(config)
    
    print(f"\n{'=' * 70}")
    print("PARAMETER BREAKDOWN:")
    print(f"{'=' * 70}")
    
    print(f"\n1. Sense Embeddings:")
    print(f"   {breakdown['sense_embeddings']:,} parameters")
    print(f"   ({breakdown['sense_embeddings']/total*100:.2f}% of total)")
    print(f"   Formula: vocab_size × (n_embd × n_senses)")
    print(f"   = {config.vocab_size:,} × ({config.n_embd} × {config.n_senses})")
    print(f"   = {config.vocab_size:,} × {config.n_embd * config.n_senses:,}")
    
    print(f"\n2. Position Embeddings:")
    print(f"   {breakdown['position_embeddings']:,} parameters")
    print(f"   ({breakdown['position_embeddings']/total*100:.2f}% of total)")
    print(f"   Formula: block_size × n_embd")
    print(f"   = {config.block_size} × {config.n_embd}")
    
    print(f"\n3. Sense Predictor:")
    print(f"   {breakdown['sense_predictor']:,} parameters")
    print(f"   ({breakdown['sense_predictor']/total*100:.2f}% of total)")
    print(f"   Layer 1: n_embd × n_embd = {config.n_embd} × {config.n_embd} = {config.n_embd * config.n_embd:,}")
    print(f"   Layer 2: n_embd × n_senses = {config.n_embd} × {config.n_senses} = {config.n_embd * config.n_senses:,}")
    
    print(f"\n4. Transformer Blocks ({config.n_layer} blocks):")
    print(f"   {breakdown['transformer_blocks']:,} parameters")
    print(f"   ({breakdown['transformer_blocks']/total*100:.2f}% of total)")
    print(f"   Per block: {breakdown['transformer_blocks']//config.n_layer:,} parameters")
    
    # Calculate per-block breakdown
    per_block = breakdown['transformer_blocks'] // config.n_layer
    print(f"\n   Per-block breakdown:")
    print(f"   - LayerNorm 1: {config.n_embd:,}")
    print(f"   - Attention (c_attn): {config.n_embd} × {3*config.n_embd} = {config.n_embd * 3 * config.n_embd:,}")
    print(f"   - Attention (c_proj): {config.n_embd} × {config.n_embd} = {config.n_embd * config.n_embd:,}")
    print(f"   - LayerNorm 2: {config.n_embd:,}")
    print(f"   - MLP (c_fc): {config.n_embd} × {4*config.n_embd} = {config.n_embd * 4 * config.n_embd:,}")
    print(f"   - MLP (c_proj): {4*config.n_embd} × {config.n_embd} = {4*config.n_embd * config.n_embd:,}")
    
    print(f"\n5. Final LayerNorm:")
    print(f"   {breakdown['final_layernorm']:,} parameters")
    print(f"   ({breakdown['final_layernorm']/total*100:.2f}% of total)")
    
    print(f"\n6. Language Modeling Head:")
    print(f"   {breakdown['lm_head']:,} parameters")
    print(f"   ({breakdown['lm_head']/total*100:.2f}% of total)")
    print(f"   Formula: n_embd × vocab_size")
    print(f"   = {config.n_embd} × {config.vocab_size:,}")
    
    print(f"\n{'=' * 70}")
    print("SUMMARY:")
    print(f"{'=' * 70}")
    print(f"Total Parameters:     {total:,}")
    print(f"                      {total/1e6:.2f}M")
    print(f"                      {total/1e9:.3f}B")
    print(f"\nTrainable Parameters: {breakdown['trainable']:,}")
    print(f"                      {breakdown['trainable']/1e6:.2f}M")
    
    # Memory estimate (assuming float16)
    memory_mb = (total * 2) / (1024 * 1024)  # 2 bytes per float16
    print(f"\nModel Size (float16): {memory_mb:.2f} MB")
    print(f"                      {memory_mb/1024:.2f} GB")
    
    # Comparison to standard transformer
    std_embeddings = config.vocab_size * config.n_embd
    std_total = (total - breakdown['sense_embeddings']) + std_embeddings
    print(f"\n{'=' * 70}")
    print("COMPARISON TO STANDARD TRANSFORMER:")
    print(f"{'=' * 70}")
    print(f"Standard Transformer (same arch, no sense vectors):")
    print(f"  Regular embeddings: {std_embeddings:,}")
    print(f"  Total:             {std_total:,} ({std_total/1e6:.2f}M)")
    print(f"\nBackpack Model:")
    print(f"  Sense embeddings:  {breakdown['sense_embeddings']:,}")
    print(f"  Total:             {total:,} ({total/1e6:.2f}M)")
    print(f"\nBackpack is {total/std_total:.1f}x larger")
    
    print(f"\n{'=' * 70}\n")
    
    return breakdown, total


if __name__ == '__main__':
    import sys
    
    config_name = sys.argv[1] if len(sys.argv) > 1 else 'train_europarl_scratch'
    
    print_parameter_report(config_name)
    
    # Also print finetune config if requested
    if 'scratch' in config_name and len(sys.argv) < 3:
        print("\n" + "=" * 70)
        print("FINETUNE CONFIGURATION:")
        print("=" * 70)
        finetune_name = config_name.replace('scratch', 'finetune')
        try:
            print_parameter_report(finetune_name)
        except Exception as e:
            print(f"Could not load {finetune_name} config: {e}")

