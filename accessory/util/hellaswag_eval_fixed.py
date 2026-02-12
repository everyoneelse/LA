"""
Fixed HellaSwag evaluation with proper rank display and FSDP compatibility
"""
import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from typing import Dict, Optional, List
from tqdm import tqdm
import jsonlines


def get_rank_info():
    """Get rank information from various sources"""
    rank_info = {}
    
    # Try distributed
    if dist.is_initialized():
        rank_info['dist_rank'] = dist.get_rank()
        rank_info['world_size'] = dist.get_world_size()
    else:
        rank_info['dist_rank'] = 0
        rank_info['world_size'] = 1
    
    # Try environment variables
    rank_info['env_rank'] = int(os.environ.get('RANK', 0))
    rank_info['local_rank'] = int(os.environ.get('LOCAL_RANK', 0))
    rank_info['world_size_env'] = int(os.environ.get('WORLD_SIZE', 1))
    
    # Use the most reliable source
    if dist.is_initialized():
        rank = rank_info['dist_rank']
        world_size = rank_info['world_size']
    else:
        rank = rank_info['env_rank']
        world_size = rank_info['world_size_env']
    
    return rank, world_size, rank_info


def safe_print(msg, rank=None, force_flush=True):
    """Print with rank information, handling various logging systems"""
    if rank is None:
        rank, _, _ = get_rank_info()
    
    # Format message with rank
    formatted_msg = f"[Rank {rank}] {msg}"
    
    # Print to stdout
    print(formatted_msg, flush=force_flush)
    
    # Also print to stderr in case stdout is redirected
    print(formatted_msg, file=sys.stderr, flush=force_flush)


class HellaSwagDataset(Dataset):
    """Standard PyTorch Dataset for HellaSwag"""
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = getattr(tokenizer, 'pad_id', 0)
        
        # Load data
        self.data = []
        if os.path.exists(data_file):
            with jsonlines.open(data_file) as reader:
                for item in reader:
                    self.data.append(item)
        else:
            raise FileNotFoundError(f"Data file not found: {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        ctx = item['ctx']
        endings = item['endings']
        label = item.get('label', -1)
        
        # Tokenize all endings
        all_tokens = []
        for ending in endings:
            full_text = ctx + " " + ending
            tokens = self.tokenizer.encode(full_text, bos=True, eos=False)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            all_tokens.append(tokens)
        
        return {
            'tokens': all_tokens,
            'label': label,
            'num_endings': len(endings)
        }


def collate_hellaswag_fixed(batch, pad_id=0):
    """
    Collate function ensuring all sequences have the same length
    CRITICAL for FSDP compatibility
    """
    # Collect all tokens
    all_tokens = []
    metadata = []
    
    for item in batch:
        for tokens in item['tokens']:
            all_tokens.append(tokens)
        metadata.append({
            'label': item['label'],
            'num_endings': item['num_endings']
        })
    
    if len(all_tokens) == 0:
        return None, None, metadata
    
    # Find global max length
    max_len = max(len(tokens) for tokens in all_tokens)
    
    # Pad all sequences to max_len
    padded_input_ids = []
    padded_labels = []
    
    for tokens in all_tokens:
        # Pad tokens
        padded = tokens + [pad_id] * (max_len - len(tokens))
        padded_input_ids.append(padded)
        
        # Create labels
        labels = padded.copy()
        for i in range(len(tokens), max_len):
            labels[i] = -100
        labels[0] = -100
        padded_labels.append(labels)
    
    # Convert to tensors
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    labels = torch.tensor(padded_labels, dtype=torch.long)
    
    return input_ids, labels, metadata


def run_hellaswag_evaluation_fixed(
    model,
    data_file: str,
    tokenizer=None,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    device: str = 'cuda',
    num_workers: int = 0
) -> Dict[str, float]:
    """
    Fixed HellaSwag evaluation with proper rank display and FSDP compatibility
    """
    
    # Get rank information
    rank, world_size, rank_info = get_rank_info()
    is_distributed = dist.is_initialized()
    
    # Print debug information
    safe_print(f"Starting HellaSwag evaluation", rank)
    safe_print(f"Rank info: {rank_info}", rank)
    safe_print(f"Distributed: {is_distributed}, World size: {world_size}", rank)
    
    # Get tokenizer
    if tokenizer is None:
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        else:
            raise ValueError("No tokenizer available")
    
    # Check data file
    if not os.path.exists(data_file):
        safe_print(f"HellaSwag data not found at {data_file}", rank)
        return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    # Create dataset
    dataset = HellaSwagDataset(data_file, tokenizer, max_length)
    
    # Limit samples if requested
    if max_samples is not None and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    total_samples = len(dataset)
    
    # Create sampler for distributed training
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        samples_per_rank = len(sampler)
    else:
        sampler = None
        samples_per_rank = total_samples
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler else False,
        collate_fn=lambda b: collate_hellaswag_fixed(b, tokenizer.pad_id),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Print evaluation info
    safe_print(f"Total samples: {total_samples}", rank)
    safe_print(f"Samples for this rank: {samples_per_rank}", rank)
    safe_print(f"Batch size: {batch_size}", rank)
    safe_print(f"Number of batches: {len(dataloader)}", rank)
    
    # Set model to eval mode
    model.eval()
    
    # Evaluation loop
    all_predictions = []
    all_labels = []
    
    # Get model dtype
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.bfloat16:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif model_dtype == torch.float16:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        autocast_ctx = torch.cuda.amp.autocast(enabled=False)
    
    with torch.no_grad():
        for batch_idx, (input_ids, labels, metadata) in enumerate(dataloader):
            if input_ids is None:
                continue
            
            # Debug first batch
            if batch_idx == 0:
                safe_print(f"First batch shape: input_ids={input_ids.shape}, labels={labels.shape}", rank)
            
            # Move to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            with autocast_ctx:
                # Forward pass - CRITICAL: all ranks must execute this
                outputs = model(input_ids, labels)
                
                if isinstance(outputs, tuple):
                    loss = outputs[0]
                else:
                    loss = outputs
                
                # Calculate perplexities
                if loss.dim() == 0:
                    # Scalar loss
                    avg_ppl = torch.exp(loss).item()
                    perplexities = [avg_ppl] * input_ids.size(0)
                else:
                    # Per-sample losses
                    perplexities = [torch.exp(l).item() for l in loss]
            
            # Process predictions
            ppl_idx = 0
            for meta in metadata:
                num_endings = meta['num_endings']
                label = meta['label']
                
                ending_ppls = perplexities[ppl_idx:ppl_idx + num_endings]
                ppl_idx += num_endings
                
                predicted = int(np.argmin(ending_ppls))
                all_predictions.append(predicted)
                
                if label >= 0:
                    all_labels.append(label)
            
            # Progress update
            if batch_idx % 10 == 0:
                safe_print(f"Processed {batch_idx+1}/{len(dataloader)} batches", rank)
    
    # Calculate local metrics
    if len(all_labels) > 0:
        correct = sum(p == l for p, l in zip(all_predictions, all_labels))
        total = len(all_labels)
    else:
        correct = 0
        total = 0
    
    safe_print(f"Local results: {correct}/{total} correct", rank)
    
    # Aggregate across ranks
    if is_distributed:
        correct_tensor = torch.tensor([correct], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        correct = correct_tensor.item()
        total = total_tensor.item()
        
        safe_print(f"Global results after aggregation: {correct}/{total}", rank)
    
    # Calculate final metrics
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'correct_samples': correct,
        'total_samples': total
    }
    
    if rank == 0:
        safe_print(f"Final HellaSwag Accuracy: {accuracy:.4f} ({correct}/{total})", rank)
    
    return metrics


# Wrapper function matching original interface
def run_hellaswag_evaluation_fsdp_fixed(
    model,
    data_dir: str,
    tokenizer=None,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Wrapper matching original interface"""
    data_file = os.path.join(data_dir, 'hellaswag_val.jsonl')
    return run_hellaswag_evaluation_fixed(
        model=model,
        data_file=data_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_samples=max_samples,
        max_length=max_length,
        device=device,
        num_workers=0
    )