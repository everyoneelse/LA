"""
Proper FSDP-compatible HellaSwag evaluation using PyTorch DataLoader and DistributedSampler
"""
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from typing import Dict, Optional, List
from tqdm import tqdm
import jsonlines


class HellaSwagDataset(Dataset):
    """
    Standard PyTorch Dataset for HellaSwag
    """
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        """
        Args:
            data_file: Path to hellaswag jsonl file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data once per process
        self.data = []
        if os.path.exists(data_file):
            with jsonlines.open(data_file) as reader:
                for item in reader:
                    self.data.append(item)
        else:
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Pre-compute pad_id
        self.pad_id = getattr(tokenizer, 'pad_id', 0)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns all endings for a single HellaSwag example
        Each ending is tokenized but NOT padded (padding done in collate_fn)
        """
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
            'tokens': all_tokens,  # List of token lists
            'label': label,
            'num_endings': len(endings)
        }


def collate_hellaswag_batch(batch, pad_id=0):
    """
    Collate function that pads all sequences in the batch to the same length
    This is CRITICAL for FSDP compatibility
    """
    # Flatten all endings from all examples
    all_tokens = []
    all_lengths = []
    metadata = []
    
    for item in batch:
        for tokens in item['tokens']:
            all_tokens.append(tokens)
            all_lengths.append(len(tokens))
        
        metadata.append({
            'label': item['label'],
            'num_endings': item['num_endings']
        })
    
    # Find max length in this batch
    if len(all_tokens) == 0:
        return None, None, metadata
    
    max_len = max(all_lengths)
    
    # Pad all sequences to max_len
    padded_input_ids = []
    padded_labels = []
    
    for tokens in all_tokens:
        # Pad tokens
        padded = tokens + [pad_id] * (max_len - len(tokens))
        padded_input_ids.append(padded)
        
        # Create labels (for loss calculation)
        labels = padded.copy()
        # Set padding positions to -100
        for i in range(len(tokens), max_len):
            labels[i] = -100
        # Ignore BOS token
        labels[0] = -100
        
        padded_labels.append(labels)
    
    # Convert to tensors
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    labels = torch.tensor(padded_labels, dtype=torch.long)
    
    return input_ids, labels, metadata


def run_hellaswag_evaluation_proper(
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
    Proper HellaSwag evaluation using PyTorch DataLoader with DistributedSampler
    
    This implementation:
    1. Uses standard PyTorch Dataset/DataLoader
    2. Each rank only loads and processes its portion of data
    3. Properly handles FSDP synchronization
    4. Aggregates results across all ranks
    """
    
    # Check if we're in distributed mode
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
            raise ValueError("No tokenizer provided and model has no tokenizer attribute")
    
    # Check if data file exists
    if not os.path.exists(data_file):
        if rank == 0:
            print(f"HellaSwag data not found at {data_file}")
        return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    # Create dataset
    dataset = HellaSwagDataset(data_file, tokenizer, max_length)
    
    # Limit samples if requested
    if max_samples is not None and max_samples < len(dataset):
        # Create a subset
        indices = list(range(max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Create sampler for distributed training
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Don't shuffle for evaluation
            drop_last=False  # Include all samples
        )
    else:
        sampler = None
    
    # Create DataLoader
    # Each rank will automatically get different data thanks to DistributedSampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler else False,
        collate_fn=lambda b: collate_hellaswag_batch(b, tokenizer.pad_id),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    if rank == 0:
        total_samples = len(dataset)
        samples_per_rank = total_samples // world_size if is_distributed else total_samples
        print(f"Evaluating HellaSwag: {total_samples} total samples")
        if is_distributed:
            print(f"  Each rank processes ~{samples_per_rank} samples")
    
    # Set model to eval mode
    model.eval()
    
    # Evaluation loop
    all_predictions = []
    all_labels = []
    
    # Get model dtype for autocast
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.bfloat16:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif model_dtype == torch.float16:
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        autocast_ctx = torch.cuda.amp.autocast(enabled=False)
    
    with torch.no_grad():
        for batch_idx, (input_ids, labels, metadata) in enumerate(tqdm(
            dataloader,
            desc=f"HellaSwag Eval [Rank {rank}]",
            disable=(rank != 0)
        )):
            if input_ids is None:  # Empty batch
                continue
            
            # Move to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            with autocast_ctx:
                # CRITICAL: Single forward pass for entire batch (FSDP requirement)
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
                    # Fallback: use batch loss
                    avg_ppl = torch.exp(batch_loss)
                    perplexities = torch.full((input_ids.size(0),), avg_ppl.item(), device=device)
            
            # Process predictions for each example in the batch
            ppl_idx = 0
            for meta in metadata:
                num_endings = meta['num_endings']
                label = meta['label']
                
                # Get perplexities for this example's endings
                ending_ppls = perplexities[ppl_idx:ppl_idx + num_endings].cpu().numpy()
                ppl_idx += num_endings
                
                # Predict the ending with lowest perplexity
                predicted = int(np.argmin(ending_ppls))
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
    
    # Debug: Print local stats
    if is_distributed:
        print(f"[Rank {rank}] Local: {correct}/{total} correct")
    
    # Aggregate results across all ranks
    if is_distributed:
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
    
    # Only rank 0 prints final results
    if rank == 0:
        print(f"HellaSwag Final Results:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return metrics


# Convenience function that matches the original interface
def run_hellaswag_evaluation_fsdp_proper(
    model,
    data_dir: str,
    tokenizer=None,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    max_length: int = 512,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Wrapper that matches the original interface but uses proper DataLoader
    """
    data_file = os.path.join(data_dir, 'hellaswag_val.jsonl')
    
    return run_hellaswag_evaluation_proper(
        model=model,
        data_file=data_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_samples=max_samples,
        max_length=max_length,
        device=device,
        num_workers=0  # Avoid multiprocessing issues
    )