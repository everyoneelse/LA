"""
FSDP-compatible HellaSwag evaluation with proper data padding
"""
import os
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, Optional, List
from tqdm import tqdm
import jsonlines


def process_hellaswag_batch_fsdp(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    metadata: List[Dict],
    device: str = 'cuda'
) -> List[int]:
    """
    Process a batch of HellaSwag examples with FSDP compatibility
    
    Key: All sequences in the batch have the SAME length (critical for FSDP)
    """
    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    # Get model dtype for autocast
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.bfloat16:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif model_dtype == torch.float16:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        autocast_ctx = torch.cuda.amp.autocast(enabled=False)
    
    with torch.no_grad():
        with autocast_ctx:
            # CRITICAL: Single forward pass for entire batch
            # This ensures FSDP synchronization works correctly
            batch_loss, logits = model(input_ids, labels)
            
            if logits is not None:
                # Calculate per-sample perplexity from logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Calculate loss per token
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                token_losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(labels.size(0), -1)
                
                # Average over valid tokens for each sample
                valid_tokens = (shift_labels != -100).sum(dim=1)
                sample_losses = token_losses.sum(dim=1) / valid_tokens.clamp(min=1)
                perplexities = torch.exp(sample_losses)
            else:
                # Fallback: use average loss
                avg_ppl = torch.exp(batch_loss)
                perplexities = torch.full((input_ids.size(0),), avg_ppl.item(), device=device)
    
    # Process predictions for each example
    predictions = []
    ppl_idx = 0
    
    for meta in metadata:
        num_endings = meta['num_endings']
        
        # Get perplexities for this example's endings
        ending_ppls = perplexities[ppl_idx:ppl_idx + num_endings].cpu().numpy()
        ppl_idx += num_endings
        
        # Predict the ending with lowest perplexity
        predicted = int(np.argmin(ending_ppls))
        predictions.append(predicted)
    
    return predictions


def run_hellaswag_evaluation_fsdp(
    model,
    data_dir: str,
    tokenizer=None,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Run HellaSwag evaluation with FSDP compatibility
    
    This version ensures:
    1. All sequences are padded to the same length within each batch
    2. Single forward pass per batch (FSDP requirement)
    3. Proper distributed synchronization
    """
    
    # Check distributed setup
    is_distributed = dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Load data
    data_file = os.path.join(data_dir, 'hellaswag_val.jsonl')
    if not os.path.exists(data_file):
        if rank == 0:
            print(f"HellaSwag data not found at {data_file}")
        return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    data = []
    with jsonlines.open(data_file) as reader:
        for item in reader:
            data.append(item)
            if max_samples is not None and len(data) >= max_samples:
                break
    
    if len(data) == 0:
        return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    # Distribute data across ranks
    if is_distributed:
        # Each rank processes a subset of data
        data_per_rank = len(data) // world_size
        start_idx = rank * data_per_rank
        end_idx = start_idx + data_per_rank if rank < world_size - 1 else len(data)
        data = data[start_idx:end_idx]
    
    if rank == 0:
        print(f"Evaluating on {len(data) * world_size if is_distributed else len(data)} HellaSwag samples...")
    
    # Set model to eval mode
    model.eval()
    
    # Process data in batches
    all_predictions = []
    all_labels = []
    
    # Batch processing
    for batch_start in tqdm(
        range(0, len(data), batch_size),
        desc=f"HellaSwag [Rank {rank}]",
        disable=(rank != 0)
    ):
        batch_end = min(batch_start + batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # Prepare batch tensors with SAME length for all sequences
        batch_input_ids = []
        batch_labels = []
        batch_metadata = []
        
        # First, find the maximum length in this batch
        batch_max_length = 0
        all_tokenized = []
        
        for item in batch_data:
            ctx = item['ctx']
            endings = item['endings']
            item_tokenized = []
            
            for ending in endings:
                full_text = ctx + " " + ending
                
                # Use model's tokenizer if available
                if hasattr(model, 'tokenizer'):
                    tokens = model.tokenizer.encode(full_text, bos=True, eos=False)
                elif tokenizer is not None:
                    tokens = tokenizer.encode(full_text, bos=True, eos=False)
                else:
                    raise ValueError("No tokenizer available")
                
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                
                item_tokenized.append(tokens)
                batch_max_length = max(batch_max_length, len(tokens))
            
            all_tokenized.append(item_tokenized)
        
        # Ensure batch_max_length doesn't exceed max_length
        batch_max_length = min(batch_max_length, max_length)
        
        # Now pad all sequences to batch_max_length
        if hasattr(model, 'tokenizer'):
            pad_id = model.tokenizer.pad_id
        elif tokenizer is not None:
            pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0
        else:
            pad_id = 0
        
        for item_idx, (item, item_tokenized) in enumerate(zip(batch_data, all_tokenized)):
            label = item.get('label', None)
            
            for tokens in item_tokenized:
                seq_len = len(tokens)
                
                # Pad to batch_max_length
                padded_tokens = tokens + [pad_id] * (batch_max_length - seq_len)
                
                # Create attention mask
                attention_mask = [1] * seq_len + [0] * (batch_max_length - seq_len)
                
                # Create labels for loss calculation
                labels_seq = padded_tokens.copy()
                # Ignore padding tokens in loss
                for i in range(seq_len, batch_max_length):
                    labels_seq[i] = -100
                # Ignore BOS token
                labels_seq[0] = -100
                
                batch_input_ids.append(padded_tokens)
                batch_labels.append(labels_seq)
            
            batch_metadata.append({
                'label': label,
                'num_endings': len(item['endings'])
            })
        
        # Convert to tensors
        input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)  # Simple attention mask
        labels = torch.tensor(batch_labels, dtype=torch.long)
        
        # Process batch with FSDP-compatible function
        predictions = process_hellaswag_batch_fsdp(
            model, input_ids, attention_mask, labels, batch_metadata, device
        )
        
        # Collect results
        for pred, meta in zip(predictions, batch_metadata):
            all_predictions.append(pred)
            if meta['label'] is not None and meta['label'] >= 0:
                all_labels.append(meta['label'])
    
    # Calculate local accuracy
    if len(all_labels) > 0:
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        total = len(all_labels)
    else:
        correct = 0
        total = 0
    
    # Aggregate results across all ranks if distributed
    if is_distributed:
        correct_tensor = torch.tensor([correct], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'correct_samples': correct,
        'total_samples': total
    }
    
    if rank == 0:
        print(f"HellaSwag Evaluation: Accuracy = {accuracy:.4f} ({correct}/{total})")
    
    return metrics