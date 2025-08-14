#!/usr/bin/env python3
"""
Test script for FLOP counter and token counter functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from accessory.util.flop_counter import FLOPCounter, TokenCounter

def test_flop_counter():
    """Test FLOP counter with a simple model configuration."""
    print("Testing FLOP Counter...")
    
    # Example model configuration (similar to LLaMA-7B)
    model_config = {
        'vocab_size': 32000,
        'dim': 4096,
        'n_layers': 32,
        'n_heads': 32,
        'multiple_of': 256,
    }
    
    flop_counter = FLOPCounter(model_config)
    
    # Test with different batch sizes and sequence lengths
    test_cases = [
        (1, 512),    # Single sample, 512 tokens
        (4, 2048),   # Batch of 4, 2048 tokens each
        (8, 1024),   # Batch of 8, 1024 tokens each
    ]
    
    print(f"Model config: {model_config}")
    print(f"FFN dimension: {flop_counter.ffn_dim}")
    print()
    
    for batch_size, seq_len in test_cases:
        forward_flops = flop_counter.calculate_forward_flops(batch_size, seq_len)
        backward_flops = flop_counter.calculate_backward_flops(batch_size, seq_len)
        total_flops = flop_counter.calculate_total_flops(batch_size, seq_len)
        
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        print(f"  Forward FLOPs:  {FLOPCounter.format_flops(forward_flops)}")
        print(f"  Backward FLOPs: {FLOPCounter.format_flops(backward_flops)}")
        print(f"  Total FLOPs:    {FLOPCounter.format_flops(total_flops)}")
        print(f"  FLOPs per token: {FLOPCounter.format_flops(total_flops // (batch_size * seq_len))}")
        print()

def test_token_counter():
    """Test token counter functionality."""
    print("Testing Token Counter...")
    
    token_counter = TokenCounter()
    
    # Simulate some batches with padding
    test_batches = [
        # (batch_size, seq_len, num_padding_tokens)
        (4, 512, 50),   # 4 samples, 512 tokens each, 50 padding tokens total
        (2, 1024, 100), # 2 samples, 1024 tokens each, 100 padding tokens total
        (8, 256, 20),   # 8 samples, 256 tokens each, 20 padding tokens total
    ]
    
    for i, (batch_size, seq_len, padding_tokens) in enumerate(test_batches):
        # Create a mock padding mask (True for valid tokens, False for padding)
        total_tokens = batch_size * seq_len
        valid_tokens = total_tokens - padding_tokens
        
        # Create padding mask
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # Set some tokens to False (padding)
        padding_mask.view(-1)[-padding_tokens:] = False
        
        token_counter.update(batch_size, seq_len, padding_mask)
        
        print(f"Batch {i+1}:")
        print(f"  Batch tokens: {TokenCounter.format_tokens(token_counter.get_batch_tokens())}")
        print(f"  Total tokens: {TokenCounter.format_tokens(token_counter.get_total_tokens())}")
        print(f"  Valid/Total ratio: {valid_tokens}/{total_tokens} = {valid_tokens/total_tokens:.2%}")
        print()

def test_format_functions():
    """Test formatting functions."""
    print("Testing Format Functions...")
    
    # Test FLOP formatting
    flop_values = [123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 1234567890, 12345678901234]
    print("FLOP formatting:")
    for val in flop_values:
        print(f"  {val:>15,} -> {FLOPCounter.format_flops(val)}")
    print()
    
    # Test token formatting
    token_values = [123, 1234, 12345, 123456, 1234567, 12345678, 123456789, 1234567890]
    print("Token formatting:")
    for val in token_values:
        print(f"  {val:>12,} -> {TokenCounter.format_tokens(val)}")
    print()

if __name__ == '__main__':
    print("=" * 60)
    print("FLOP Counter and Token Counter Test")
    print("=" * 60)
    print()
    
    test_flop_counter()
    print("-" * 60)
    test_token_counter()
    print("-" * 60)
    test_format_functions()
    
    print("All tests completed successfully!")