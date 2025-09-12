"""
Debug version of HellaSwag evaluation to identify FSDP hanging issues
"""
import os
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, Optional
import time
import traceback


def debug_print(msg, rank=None):
    """Print with rank and timestamp"""
    if rank is None:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}][Rank {rank}] {msg}", flush=True)


def run_hellaswag_evaluation_minimal(
    model,
    data_dir: str,
    tokenizer=None,
    batch_size: int = 1,  # Start with batch_size=1
    max_samples: Optional[int] = 2,  # Very small for debugging
    max_length: int = 128,  # Shorter sequences for debugging
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Minimal HellaSwag evaluation for debugging FSDP issues
    """
    
    # Check distributed setup
    is_distributed = dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    debug_print(f"Starting HellaSwag eval. Distributed={is_distributed}, World size={world_size}", rank)
    
    # Get tokenizer
    if tokenizer is None:
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            raise ValueError("No tokenizer available")
    
    # Create minimal test data
    debug_print("Creating test data...", rank)
    
    # Use a simple test case
    test_data = [
        {
            'ctx': 'The cat sat on',
            'endings': ['the mat.', 'the dog.'],
            'label': 0
        }
    ] * max_samples  # Repeat for multiple samples
    
    # Ensure all ranks have same data
    if is_distributed:
        debug_print(f"Syncing test data across ranks...", rank)
        dist.barrier()
    
    # Set model to eval mode
    debug_print("Setting model to eval mode...", rank)
    model.eval()
    
    # Process data WITHOUT DataLoader first
    debug_print("Processing data...", rank)
    
    try:
        with torch.no_grad():
            for i, item in enumerate(test_data):
                debug_print(f"Processing item {i+1}/{len(test_data)}", rank)
                
                ctx = item['ctx']
                endings = item['endings']
                
                # Tokenize
                all_input_ids = []
                all_labels = []
                
                for ending in endings:
                    text = ctx + " " + ending
                    tokens = tokenizer.encode(text, bos=True, eos=False)
                    
                    if len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    
                    # Pad to max_length
                    pad_id = getattr(tokenizer, 'pad_id', 0)
                    padded = tokens + [pad_id] * (max_length - len(tokens))
                    
                    # Create labels
                    labels = padded.copy()
                    for j in range(len(tokens), max_length):
                        labels[j] = -100
                    labels[0] = -100
                    
                    all_input_ids.append(padded)
                    all_labels.append(labels)
                
                # Convert to tensors
                input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=device)
                labels = torch.tensor(all_labels, dtype=torch.long, device=device)
                
                debug_print(f"Input shape: {input_ids.shape}, Labels shape: {labels.shape}", rank)
                
                # Sync before forward pass
                if is_distributed:
                    debug_print("Syncing before forward pass...", rank)
                    dist.barrier()
                
                # Forward pass
                debug_print("Starting forward pass...", rank)
                start_time = time.time()
                
                # Try different approaches
                try:
                    # Approach 1: Direct forward
                    debug_print("Trying direct forward pass...", rank)
                    outputs = model(input_ids, labels)
                    
                    if isinstance(outputs, tuple):
                        loss = outputs[0]
                    else:
                        loss = outputs
                    
                    debug_print(f"Forward pass completed in {time.time()-start_time:.2f}s", rank)
                    debug_print(f"Loss: {loss.item() if loss.dim() == 0 else 'batch loss'}", rank)
                    
                except Exception as e:
                    debug_print(f"Direct forward failed: {e}", rank)
                    debug_print(f"Traceback: {traceback.format_exc()}", rank)
                    
                    # Approach 2: Try without labels
                    debug_print("Trying forward without labels...", rank)
                    try:
                        with torch.cuda.amp.autocast(enabled=False):
                            outputs = model(input_ids)
                            debug_print("Forward without labels succeeded!", rank)
                    except Exception as e2:
                        debug_print(f"Forward without labels also failed: {e2}", rank)
                
                # Sync after forward pass
                if is_distributed:
                    debug_print("Syncing after forward pass...", rank)
                    dist.barrier()
                    debug_print("Post-forward barrier passed!", rank)
        
        debug_print("Evaluation loop completed!", rank)
        
    except Exception as e:
        debug_print(f"ERROR in evaluation: {e}", rank)
        debug_print(f"Full traceback:\n{traceback.format_exc()}", rank)
        raise
    
    # Final sync
    if is_distributed:
        debug_print("Final sync...", rank)
        dist.barrier()
        debug_print("Final barrier passed!", rank)
    
    # Return dummy metrics
    metrics = {
        'accuracy': 0.5,
        'correct_samples': 1,
        'total_samples': 2
    }
    
    debug_print(f"Evaluation completed successfully! Metrics: {metrics}", rank)
    
    return metrics


def test_model_forward_minimal(model, tokenizer, device='cuda'):
    """
    Test the most minimal forward pass possible
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    debug_print("Testing minimal forward pass...", rank)
    
    # Create the simplest possible input
    text = "Hello world"
    tokens = tokenizer.encode(text, bos=True, eos=False)[:10]  # Very short
    
    # Single sample, no batch
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[0, 0] = -100  # Ignore first token
    
    debug_print(f"Minimal input shape: {input_ids.shape}", rank)
    
    model.eval()
    
    with torch.no_grad():
        try:
            # Try 1: With labels
            debug_print("Attempt 1: Forward with labels...", rank)
            output = model(input_ids, labels)
            debug_print(f"Success! Output type: {type(output)}", rank)
            
        except Exception as e:
            debug_print(f"Failed with labels: {e}", rank)
            
            # Try 2: Without labels
            try:
                debug_print("Attempt 2: Forward without labels...", rank)
                output = model(input_ids)
                debug_print(f"Success without labels! Output type: {type(output)}", rank)
            except Exception as e2:
                debug_print(f"Failed without labels: {e2}", rank)
                raise
    
    if dist.is_initialized():
        dist.barrier()
        debug_print("Barrier after minimal test passed!", rank)
    
    return True