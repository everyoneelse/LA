"""
HellaSwag evaluation with detailed shape debugging for FSDP
"""
import os
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, Optional, List
import jsonlines
import time


def debug_tensor_info(tensor, name, rank=None):
    """Print detailed tensor information"""
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    
    print(f"[Rank {rank}] {name}:", flush=True)
    print(f"  Shape: {tensor.shape}", flush=True)
    print(f"  Dtype: {tensor.dtype}", flush=True)
    print(f"  Device: {tensor.device}", flush=True)
    print(f"  Min/Max: {tensor.min().item()}/{tensor.max().item()}", flush=True)
    if tensor.numel() < 50:  # Print small tensors
        print(f"  Values: {tensor.tolist()}", flush=True)
    else:
        print(f"  First 10: {tensor.flatten()[:10].tolist()}", flush=True)


def sync_and_compare_shapes(input_ids, labels, rank, world_size):
    """Synchronize and compare tensor shapes across all ranks"""
    print(f"\n[Rank {rank}] ========== SHAPE SYNCHRONIZATION CHECK ==========", flush=True)
    
    # Gather shapes from all ranks
    local_shapes = {
        'input_ids': list(input_ids.shape),
        'labels': list(labels.shape),
        'rank': rank
    }
    
    if dist.is_initialized():
        all_shapes = [None] * world_size
        dist.all_gather_object(all_shapes, local_shapes)
        
        # Print all shapes
        print(f"[Rank {rank}] Shapes across all ranks:", flush=True)
        for r_shapes in all_shapes:
            r = r_shapes['rank']
            print(f"  Rank {r}: input_ids={r_shapes['input_ids']}, labels={r_shapes['labels']}", flush=True)
        
        # Check if all shapes are identical
        first_input_shape = all_shapes[0]['input_ids']
        first_label_shape = all_shapes[0]['labels']
        
        all_same = all(
            s['input_ids'] == first_input_shape and s['labels'] == first_label_shape 
            for s in all_shapes
        )
        
        if all_same:
            print(f"[Rank {rank}] ✅ All ranks have IDENTICAL shapes!", flush=True)
        else:
            print(f"[Rank {rank}] ❌ WARNING: Ranks have DIFFERENT shapes!", flush=True)
            print(f"[Rank {rank}] This will cause FSDP to hang!", flush=True)
        
        return all_same
    else:
        print(f"[Rank {rank}] Not distributed, skipping shape comparison", flush=True)
        return True


def run_hellaswag_evaluation_debug_shapes(
    model,
    data_dir: str,
    tokenizer=None,
    batch_size: int = 2,
    max_samples: Optional[int] = 4,  # Small for debugging
    max_length: int = 128,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    HellaSwag evaluation with detailed shape debugging
    """
    
    # Check distributed setup
    is_distributed = dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    print(f"\n[Rank {rank}] Starting HellaSwag evaluation with shape debugging", flush=True)
    print(f"[Rank {rank}] Distributed: {is_distributed}, World size: {world_size}", flush=True)
    
    # Get tokenizer
    if tokenizer is None:
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            raise ValueError("No tokenizer available")
    
    pad_id = getattr(tokenizer, 'pad_id', 0)
    
    # Load data
    data_file = os.path.join(data_dir, 'hellaswag_val.jsonl')
    if not os.path.exists(data_file):
        # Use dummy data for testing
        print(f"[Rank {rank}] Using dummy test data", flush=True)
        data = [
            {
                'ctx': 'The cat sat on',
                'endings': ['the mat.', 'the dog.', 'the chair.', 'the floor.'],
                'label': 0
            },
            {
                'ctx': 'The dog ran to',
                'endings': ['the park.', 'the house.', 'the store.', 'the car.'],
                'label': 1
            }
        ] * (max_samples // 2)
    else:
        data = []
        with jsonlines.open(data_file) as reader:
            for item in reader:
                data.append(item)
                if max_samples is not None and len(data) >= max_samples:
                    break
    
    print(f"[Rank {rank}] Loaded {len(data)} examples", flush=True)
    
    # CRITICAL: Ensure all ranks have the same data
    if is_distributed:
        # Broadcast data length from rank 0
        data_len = torch.tensor([len(data)], dtype=torch.long, device=device)
        dist.broadcast(data_len, src=0)
        expected_len = data_len.item()
        
        if len(data) != expected_len:
            print(f"[Rank {rank}] WARNING: Data length mismatch! Local: {len(data)}, Expected: {expected_len}", flush=True)
    
    # Set model to eval mode
    model.eval()
    
    # Process data in batches
    all_predictions = []
    all_labels = []
    
    # Process examples
    for batch_idx in range(0, len(data), batch_size):
        batch_data = data[batch_idx:batch_idx + batch_size]
        
        print(f"\n[Rank {rank}] Processing batch {batch_idx // batch_size + 1}", flush=True)
        print(f"[Rank {rank}] Batch size: {len(batch_data)}", flush=True)
        
        # Collect all endings from the batch
        all_texts = []
        metadata = []
        
        for item in batch_data:
            ctx = item['ctx']
            endings = item['endings']
            
            for ending in endings:
                all_texts.append(ctx + " " + ending)
            
            metadata.append({
                'num_endings': len(endings),
                'label': item.get('label', -1)
            })
        
        print(f"[Rank {rank}] Total texts to process: {len(all_texts)}", flush=True)
        
        # Tokenize all texts
        all_tokens = []
        for text in all_texts:
            tokens = tokenizer.encode(text, bos=True, eos=False)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            all_tokens.append(tokens)
        
        # Find the maximum length in this batch
        max_len = max(len(tokens) for tokens in all_tokens)
        print(f"[Rank {rank}] Token lengths: min={min(len(t) for t in all_tokens)}, max={max_len}", flush=True)
        
        # CRITICAL: Ensure all ranks use the same max_len
        if is_distributed:
            print(f"[Rank {rank}] Synchronizing max_len across ranks...", flush=True)
            max_len_tensor = torch.tensor([max_len], dtype=torch.long, device=device)
            dist.all_reduce(max_len_tensor, op=dist.ReduceOp.MAX)
            global_max_len = max_len_tensor.item()
            
            if global_max_len != max_len:
                print(f"[Rank {rank}] Adjusting max_len from {max_len} to {global_max_len} (global max)", flush=True)
                max_len = global_max_len
        
        # Pad all sequences to max_len
        batch_input_ids = []
        batch_labels = []
        
        for tokens in all_tokens:
            # Pad to max_len
            padded = tokens + [pad_id] * (max_len - len(tokens))
            
            # Create labels
            labels = padded.copy()
            for i in range(len(tokens), max_len):
                labels[i] = -100  # Ignore padding
            labels[0] = -100  # Ignore BOS
            
            batch_input_ids.append(padded)
            batch_labels.append(labels)
        
        # Convert to tensors
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
        
        # DEBUG: Print detailed tensor information
        print(f"\n[Rank {rank}] ===== TENSOR SHAPES BEFORE FORWARD PASS =====", flush=True)
        debug_tensor_info(input_ids, "input_ids", rank)
        debug_tensor_info(labels, "labels", rank)
        
        # CRITICAL: Verify all ranks have identical shapes
        shapes_match = sync_and_compare_shapes(input_ids, labels, rank, world_size)
        
        if not shapes_match:
            print(f"[Rank {rank}] ⚠️  STOPPING: Shape mismatch detected!", flush=True)
            if is_distributed:
                # Try to sync before failing
                dist.barrier()
            return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
        
        # Sync before forward pass
        if is_distributed:
            print(f"[Rank {rank}] Syncing before forward pass...", flush=True)
            dist.barrier()
            print(f"[Rank {rank}] Pre-forward barrier passed", flush=True)
        
        # Forward pass
        print(f"[Rank {rank}] Starting forward pass...", flush=True)
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Try with autocast
                model_dtype = next(model.parameters()).dtype
                print(f"[Rank {rank}] Model dtype: {model_dtype}", flush=True)
                
                if model_dtype == torch.bfloat16:
                    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
                elif model_dtype == torch.float16:
                    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
                else:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=False)
                
                with autocast_ctx:
                    print(f"[Rank {rank}] Calling model(input_ids, labels)...", flush=True)
                    outputs = model(input_ids, labels)
                    print(f"[Rank {rank}] Forward pass completed in {time.time() - start_time:.2f}s", flush=True)
                    
                    if isinstance(outputs, tuple):
                        loss = outputs[0]
                        print(f"[Rank {rank}] Output is tuple, loss shape: {loss.shape}", flush=True)
                    else:
                        loss = outputs
                        print(f"[Rank {rank}] Output is tensor, loss shape: {loss.shape}", flush=True)
                    
                    # Calculate perplexities
                    if loss.dim() == 0:
                        # Scalar loss (averaged)
                        avg_ppl = torch.exp(loss).item()
                        print(f"[Rank {rank}] Scalar loss: {loss.item():.4f}, Perplexity: {avg_ppl:.4f}", flush=True)
                        perplexities = [avg_ppl] * len(all_texts)
                    else:
                        # Per-sample losses
                        perplexities = [torch.exp(l).item() for l in loss]
                        print(f"[Rank {rank}] Per-sample losses, first 3 perplexities: {perplexities[:3]}", flush=True)
        
        except Exception as e:
            print(f"[Rank {rank}] ❌ Forward pass FAILED: {e}", flush=True)
            import traceback
            print(f"[Rank {rank}] Traceback:\n{traceback.format_exc()}", flush=True)
            
            if is_distributed:
                # Try to sync even after error
                try:
                    dist.barrier()
                except:
                    pass
            
            return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
        
        # Sync after forward pass
        if is_distributed:
            print(f"[Rank {rank}] Syncing after forward pass...", flush=True)
            dist.barrier()
            print(f"[Rank {rank}] Post-forward barrier passed", flush=True)
        
        # Process predictions
        ppl_idx = 0
        for meta in metadata:
            num_endings = meta['num_endings']
            ending_ppls = perplexities[ppl_idx:ppl_idx + num_endings]
            ppl_idx += num_endings
            
            predicted = int(np.argmin(ending_ppls))
            all_predictions.append(predicted)
            
            if meta['label'] >= 0:
                all_labels.append(meta['label'])
        
        print(f"[Rank {rank}] Batch processing completed", flush=True)
    
    # Calculate metrics
    if len(all_labels) > 0:
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        total = len(all_labels)
        accuracy = correct / total
    else:
        correct = 0
        total = 0
        accuracy = 0.0
    
    print(f"\n[Rank {rank}] ===== EVALUATION COMPLETED =====", flush=True)
    print(f"[Rank {rank}] Accuracy: {accuracy:.4f} ({correct}/{total})", flush=True)
    
    # Final sync
    if is_distributed:
        print(f"[Rank {rank}] Final sync...", flush=True)
        dist.barrier()
        print(f"[Rank {rank}] Final barrier passed", flush=True)
    
    return {
        'accuracy': accuracy,
        'correct_samples': correct,
        'total_samples': total
    }