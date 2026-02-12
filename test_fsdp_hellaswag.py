#!/usr/bin/env python
"""
Test script to debug FSDP hanging issues with HellaSwag evaluation
Run with: torchrun --nproc_per_node=2 test_fsdp_hellaswag.py
"""
import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path

# Add accessory to path
sys.path.append(str(Path(__file__).parent))

from accessory.util.hellaswag_eval_debug import (
    run_hellaswag_evaluation_minimal,
    test_model_forward_minimal,
    debug_print
)


def test_basic_fsdp():
    """Test basic FSDP functionality"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    debug_print(f"Test 1: Basic FSDP setup", rank)
    debug_print(f"World size: {world_size}", rank)
    
    # Test barrier
    debug_print("Testing barrier...", rank)
    dist.barrier()
    debug_print("Barrier passed!", rank)
    
    # Test all_reduce
    debug_print("Testing all_reduce...", rank)
    tensor = torch.tensor([rank], dtype=torch.float32, device='cuda')
    dist.all_reduce(tensor)
    debug_print(f"All_reduce result: {tensor.item()} (expected: {sum(range(world_size))})", rank)
    
    return True


def test_model_loading():
    """Test loading model with FSDP"""
    rank = dist.get_rank()
    
    debug_print("Test 2: Model loading", rank)
    
    # Try to load a simple model
    try:
        # Create a minimal model for testing
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(1000, 128)
                self.linear = nn.Linear(128, 1000)
                
                # Add a mock tokenizer
                class MockTokenizer:
                    def encode(self, text, bos=True, eos=False):
                        # Return dummy tokens
                        return [1, 2, 3, 4, 5]
                    
                    @property
                    def pad_id(self):
                        return 0
                
                self.tokenizer = MockTokenizer()
            
            def forward(self, input_ids, labels=None):
                x = self.embed(input_ids)
                logits = self.linear(x)
                
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    return loss, logits
                return logits
        
        model = SimpleModel().cuda()
        
        # Wrap with FSDP
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model)
        
        debug_print("Model created and wrapped with FSDP", rank)
        
        # Test forward pass
        test_model_forward_minimal(model, model.module.tokenizer)
        
        return model
        
    except Exception as e:
        debug_print(f"Model loading failed: {e}", rank)
        import traceback
        debug_print(f"Traceback:\n{traceback.format_exc()}", rank)
        return None


def test_hellaswag_eval(model):
    """Test HellaSwag evaluation"""
    rank = dist.get_rank()
    
    debug_print("Test 3: HellaSwag evaluation", rank)
    
    if model is None:
        debug_print("No model available, skipping", rank)
        return False
    
    try:
        # Run minimal evaluation
        metrics = run_hellaswag_evaluation_minimal(
            model=model,
            data_dir='.',  # Dummy, we use test data
            tokenizer=model.module.tokenizer,
            batch_size=1,
            max_samples=2,
            max_length=32,
            device='cuda'
        )
        
        debug_print(f"Evaluation succeeded! Metrics: {metrics}", rank)
        return True
        
    except Exception as e:
        debug_print(f"Evaluation failed: {e}", rank)
        import traceback
        debug_print(f"Traceback:\n{traceback.format_exc()}", rank)
        return False


def main():
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    
    debug_print("="*60, rank)
    debug_print("Starting FSDP HellaSwag debugging", rank)
    debug_print("="*60, rank)
    
    # Test 1: Basic FSDP
    if test_basic_fsdp():
        debug_print("✅ Test 1 passed: Basic FSDP", rank)
    else:
        debug_print("❌ Test 1 failed: Basic FSDP", rank)
        return
    
    dist.barrier()
    
    # Test 2: Model loading
    model = test_model_loading()
    if model is not None:
        debug_print("✅ Test 2 passed: Model loading", rank)
    else:
        debug_print("❌ Test 2 failed: Model loading", rank)
        return
    
    dist.barrier()
    
    # Test 3: HellaSwag evaluation
    if test_hellaswag_eval(model):
        debug_print("✅ Test 3 passed: HellaSwag evaluation", rank)
    else:
        debug_print("❌ Test 3 failed: HellaSwag evaluation", rank)
    
    dist.barrier()
    
    debug_print("="*60, rank)
    debug_print("All tests completed!", rank)
    debug_print("="*60, rank)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()