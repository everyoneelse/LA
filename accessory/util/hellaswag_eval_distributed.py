"""
Distributed HellaSwag evaluation utilities
"""
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
import json
import jsonlines

from accessory.data.hellaswag_dataset import HellaSwagDataset, collate_hellaswag


def run_hellaswag_evaluation_distributed(
    model, 
    tokenizer,
    data_file: str,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Run HellaSwag evaluation with proper distributed data loading
    
    This function properly handles:
    1. Distributed data loading (each rank processes different data)
    2. FSDP-compatible batch processing
    3. Result aggregation across all ranks
    """
    
    # Check if we're in distributed mode
    is_distributed = dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Create dataset
    dataset = HellaSwagDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples
    )
    
    # Create distributed sampler if needed
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Don't shuffle for evaluation
            drop_last=False
        )
    else:
        sampler = None
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler else False,
        collate_fn=collate_hellaswag,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )
    
    # Set model to eval mode
    model.eval()
    
    # Evaluation loop
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (input_ids, labels, metadata) in enumerate(tqdm(
            dataloader, 
            desc=f"HellaSwag Eval [Rank {rank}]",
            disable=(rank != 0)  # Only show progress on rank 0
        )):
            # Move to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Get model dtype for autocast
            model_dtype = next(model.parameters()).dtype
            if model_dtype == torch.bfloat16:
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
            elif model_dtype == torch.float16:
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=False)
            
            with autocast_ctx:
                # FSDP-compatible: Process entire batch at once
                batch_loss, logits = model(input_ids, labels)
                
                # Calculate per-sample perplexity
                if logits is not None:
                    # Calculate loss per sample
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                    token_losses = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    ).view(labels.size(0), -1)
                    
                    # Average over valid tokens
                    valid_tokens = (shift_labels != -100).sum(dim=1)
                    sample_losses = token_losses.sum(dim=1) / valid_tokens.clamp(min=1)
                    perplexities = torch.exp(sample_losses)
                else:
                    # Fallback: use batch loss
                    avg_ppl = torch.exp(batch_loss)
                    perplexities = torch.full((input_ids.size(0),), avg_ppl.item(), device=device)
            
            # Process results for each example in the batch
            ppl_idx = 0
            for meta in metadata:
                num_endings = meta['num_endings']
                label = meta['label']
                
                # Get perplexities for this example's endings
                ending_ppls = perplexities[ppl_idx:ppl_idx + num_endings].cpu().numpy()
                ppl_idx += num_endings
                
                # Predict the ending with lowest perplexity
                predicted = np.argmin(ending_ppls)
                
                all_predictions.append(predicted)
                if label >= 0:
                    all_labels.append(label)
    
    # Calculate local metrics
    if len(all_labels) > 0:
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        total = len(all_labels)
    else:
        correct = 0
        total = 0
    
    # Aggregate results across all ranks if distributed
    if is_distributed:
        # Convert to tensors for all_reduce
        correct_tensor = torch.tensor([correct], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        
        # Sum across all ranks
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'correct_samples': correct,
        'total_samples': total
    }
    
    # Only rank 0 prints results
    if rank == 0:
        print(f"HellaSwag Evaluation Results:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return metrics