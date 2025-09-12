"""
Simplified HellaSwag evaluation that works with FSDP
Key: Process each ending individually, but ensure all ranks do the same operations
"""
import os
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, Optional, List
import jsonlines


def run_hellaswag_evaluation_simple(
    model,
    data_dir: str,
    tokenizer=None,
    batch_size: int = 1,  # Process one example at a time
    max_samples: Optional[int] = None,
    max_length: int = 512,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Simple HellaSwag evaluation that avoids FSDP synchronization issues
    
    Key design choices:
    1. Process one HellaSwag example at a time (not one ending)
    2. All ranks process the same examples in the same order
    3. Use model.generate() or forward() consistently
    """
    
    # Check distributed setup
    is_distributed = dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Get tokenizer
    if tokenizer is None:
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            raise ValueError("No tokenizer available")
    
    # Load data
    data_file = os.path.join(data_dir, 'hellaswag_val.jsonl')
    if not os.path.exists(data_file):
        if rank == 0:
            print(f"HellaSwag data not found at {data_file}")
        return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    # Load all data (same on all ranks)
    data = []
    with jsonlines.open(data_file) as reader:
        for item in reader:
            data.append(item)
            if max_samples is not None and len(data) >= max_samples:
                break
    
    if len(data) == 0:
        return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    # IMPORTANT: All ranks process ALL data to maintain FSDP sync
    # We'll aggregate results at the end
    if rank == 0:
        print(f"Evaluating on {len(data)} HellaSwag samples...")
        print(f"Note: All ranks process all data for FSDP compatibility")
    
    # Set model to eval mode
    model.eval()
    
    # Process each example
    all_predictions = []
    all_labels = []
    
    pad_id = getattr(tokenizer, 'pad_id', 0)
    
    with torch.no_grad():
        for idx, item in enumerate(data):
            if rank == 0 and idx % 100 == 0:
                print(f"Processing example {idx+1}/{len(data)}...")
            
            ctx = item['ctx']
            endings = item['endings']
            label = item.get('label', -1)
            
            # Process all endings for this example together
            # This ensures all ranks do the same number of forward passes
            perplexities = []
            
            for ending in endings:
                text = ctx + " " + ending
                tokens = tokenizer.encode(text, bos=True, eos=False)
                
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                
                # Create input - single sequence
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                
                # Create labels for loss calculation
                labels = input_ids.clone()
                labels[0, 0] = -100  # Ignore BOS token
                
                # Forward pass - all ranks must execute this
                try:
                    # Get model output
                    outputs = model(input_ids, labels)
                    
                    # Extract loss
                    if isinstance(outputs, tuple):
                        loss = outputs[0]
                    else:
                        loss = outputs
                    
                    # Calculate perplexity
                    ppl = torch.exp(loss).item()
                    perplexities.append(ppl)
                    
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Forward pass failed for example {idx}: {e}")
                    perplexities.append(float('inf'))
            
            # Choose the ending with lowest perplexity
            if len(perplexities) > 0:
                predicted = int(np.argmin(perplexities))
            else:
                predicted = 0
            
            all_predictions.append(predicted)
            if label >= 0:
                all_labels.append(label)
    
    # Calculate accuracy (same on all ranks since all processed same data)
    if len(all_labels) > 0:
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        total = len(all_labels)
        accuracy = correct / total
    else:
        correct = 0
        total = 0
        accuracy = 0.0
    
    # No need to aggregate since all ranks computed the same thing
    # This is inefficient but ensures FSDP compatibility
    
    metrics = {
        'accuracy': accuracy,
        'correct_samples': correct,
        'total_samples': total
    }
    
    if rank == 0:
        print(f"HellaSwag Evaluation Results:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return metrics


def run_hellaswag_evaluation_alternating(
    model,
    data_dir: str,
    tokenizer=None,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Alternative: Each rank processes different examples, but all do same operations
    """
    
    # Check distributed setup
    is_distributed = dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Get tokenizer
    if tokenizer is None:
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            raise ValueError("No tokenizer available")
    
    # Load data
    data_file = os.path.join(data_dir, 'hellaswag_val.jsonl')
    if not os.path.exists(data_file):
        if rank == 0:
            print(f"HellaSwag data not found at {data_file}")
        return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    # Load data
    data = []
    with jsonlines.open(data_file) as reader:
        for item in reader:
            data.append(item)
            if max_samples is not None and len(data) >= max_samples:
                break
    
    # Distribute examples across ranks
    # Each rank gets every world_size-th example
    my_data = [data[i] for i in range(len(data)) if i % world_size == rank]
    
    if rank == 0:
        print(f"Total samples: {len(data)}, Each rank processes: {len(my_data)}")
    
    # Set model to eval mode
    model.eval()
    
    # Process assigned examples
    local_predictions = []
    local_labels = []
    local_indices = []
    
    pad_id = getattr(tokenizer, 'pad_id', 0)
    
    with torch.no_grad():
        for local_idx, (global_idx, item) in enumerate(
            [(i, data[i]) for i in range(len(data)) if i % world_size == rank]
        ):
            ctx = item['ctx']
            endings = item['endings']
            label = item.get('label', -1)
            
            # Find max length for this example's endings
            max_ending_len = 0
            tokenized_endings = []
            
            for ending in endings:
                text = ctx + " " + ending
                tokens = tokenizer.encode(text, bos=True, eos=False)
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                tokenized_endings.append(tokens)
                max_ending_len = max(max_ending_len, len(tokens))
            
            # Process all endings as a batch (same length)
            batch_input_ids = []
            batch_labels = []
            
            for tokens in tokenized_endings:
                # Pad to max_ending_len
                padded = tokens + [pad_id] * (max_ending_len - len(tokens))
                
                # Create labels
                labels_seq = padded.copy()
                for i in range(len(tokens), max_ending_len):
                    labels_seq[i] = -100
                labels_seq[0] = -100
                
                batch_input_ids.append(padded)
                batch_labels.append(labels_seq)
            
            # Convert to tensors
            input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
            labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
            
            # Single forward pass for all endings
            try:
                outputs = model(input_ids, labels)
                
                if isinstance(outputs, tuple):
                    loss = outputs[0]
                else:
                    loss = outputs
                
                # If loss is averaged, we need per-sample losses
                # This is a simplification - ideally model should return per-sample losses
                if loss.dim() == 0:
                    # Averaged loss - use same for all
                    perplexities = [torch.exp(loss).item()] * len(endings)
                else:
                    # Per-sample losses
                    perplexities = [torch.exp(l).item() for l in loss]
                
                predicted = int(np.argmin(perplexities))
                
            except Exception as e:
                if rank == 0:
                    print(f"Error processing example {global_idx}: {e}")
                predicted = 0
            
            local_predictions.append(predicted)
            local_indices.append(global_idx)
            if label >= 0:
                local_labels.append(label)
    
    # Gather results from all ranks
    if is_distributed:
        # Gather all predictions and indices
        all_predictions_list = [None] * world_size
        all_indices_list = [None] * world_size
        all_labels_list = [None] * world_size
        
        dist.all_gather_object(all_predictions_list, local_predictions)
        dist.all_gather_object(all_indices_list, local_indices)
        dist.all_gather_object(all_labels_list, local_labels)
        
        # Combine results
        combined_predictions = []
        combined_labels = []
        
        for rank_preds, rank_indices, rank_labels in zip(
            all_predictions_list, all_indices_list, all_labels_list
        ):
            for pred, idx in zip(rank_preds, rank_indices):
                combined_predictions.append((idx, pred))
            combined_labels.extend(rank_labels)
        
        # Sort by original index
        combined_predictions.sort(key=lambda x: x[0])
        final_predictions = [p for _, p in combined_predictions]
        final_labels = combined_labels
    else:
        final_predictions = local_predictions
        final_labels = local_labels
    
    # Calculate metrics
    if len(final_labels) > 0:
        correct = sum(p == l for p, l in zip(final_predictions, final_labels))
        total = len(final_labels)
        accuracy = correct / total
    else:
        correct = 0
        total = 0
        accuracy = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'correct_samples': correct,
        'total_samples': total
    }
    
    if rank == 0:
        print(f"HellaSwag Results: Accuracy = {accuracy:.4f} ({correct}/{total})")
    
    return metrics