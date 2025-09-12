"""
Fixed HellaSwag evaluation code with proper data distribution across ranks
"""
import torch
import torch.distributed as dist
from tqdm import tqdm
from typing import List, Dict, Any

def distribute_data_across_ranks(data: List, rank: int, world_size: int) -> List:
    """
    Split data across ranks for distributed processing.
    Each rank gets a different subset of the data.
    """
    total_samples = len(data)
    samples_per_rank = total_samples // world_size
    remainder = total_samples % world_size
    
    # Calculate start and end indices for this rank
    if rank < remainder:
        # First 'remainder' ranks get one extra sample
        start_idx = rank * (samples_per_rank + 1)
        end_idx = start_idx + samples_per_rank + 1
    else:
        # Remaining ranks get the base number of samples
        start_idx = rank * samples_per_rank + remainder
        end_idx = start_idx + samples_per_rank
    
    # Return the slice of data for this rank
    return data[start_idx:end_idx]

# Your modified code with proper distributed data handling
def evaluate_hellaswag_distributed(model, data, batch_size, device='cuda'):
    """
    Distributed HellaSwag evaluation with proper data splitting
    """
    # Check if we're in distributed mode
    is_distributed = dist.is_initialized() if torch.distributed.is_available() else False
    
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank  # or you can use os.environ.get('LOCAL_RANK', 0)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    try:
        # Split data across ranks BEFORE batching
        if is_distributed:
            # Each rank gets different data
            rank_data = distribute_data_across_ranks(data, rank, world_size)
            total_samples = len(data)  # Total across all ranks
            rank_samples = len(rank_data)  # Samples for this rank
            
            # Print info for each rank
            if rank == 0:
                print(f"[Rank 0] Evaluating HellaSwag: {total_samples} total samples")
                print(f"[Rank 0]   Each rank processes ~{total_samples // world_size} samples")
                print(f"[Rank 0]   This rank processes {rank_samples} samples")
            else:
                print(f"[Rank {rank}] Evaluating HellaSwag: Processing {rank_samples} samples")
        else:
            # Single GPU/CPU - process all data
            rank_data = data
            total_samples = len(data)
            rank_samples = total_samples
            print(f"Evaluating HellaSwag: {total_samples} total samples (single process)")
        
        # Batch the rank-specific data
        batched_data = batch_data(rank_data, batch_size)
        
        # Process batches
        all_results = []
        
        # Only show progress bar on rank 0 in distributed mode
        show_progress = (rank == 0) if is_distributed else True
        
        if show_progress:
            for batch in tqdm(batched_data, desc=f"HellaSwag Eval [Rank {rank}]", leave=False):
                batch_results = evaluate_hellaswag_batch(model, batch, device)
                all_results.extend(batch_results)
        else:
            for batch in batched_data:
                batch_results = evaluate_hellaswag_batch(model, batch, device)
                all_results.extend(batch_results)
        
        # Calculate local metrics for this rank
        correct_predictions = [r for r in all_results if r.get('correct') is True]
        total_predictions = [r for r in all_results if r.get('correct') is not None]
        
        local_correct = len(correct_predictions)
        local_total = len(total_predictions)
        
        # Aggregate results across all ranks if distributed
        if is_distributed:
            # Convert to tensors for all_reduce
            correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=device)
            total_tensor = torch.tensor([local_total], dtype=torch.long, device=device)
            
            # Sum across all ranks
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            
            # Get aggregated results
            global_correct = correct_tensor.item()
            global_total = total_tensor.item()
            
            # Calculate global accuracy
            if global_total > 0:
                accuracy = global_correct / global_total
            else:
                accuracy = 0.0
            
            # Only rank 0 prints final results
            if rank == 0:
                print(f"\n[Final Results] HellaSwag Accuracy: {accuracy:.4f} ({global_correct}/{global_total})")
            
            metrics = {
                'accuracy': accuracy,
                'total_samples': global_total,
                'correct_samples': global_correct,
                'local_samples': local_total,
                'local_correct': local_correct
            }
        else:
            # Single process - local is global
            if local_total > 0:
                accuracy = local_correct / local_total
            else:
                accuracy = 0.0
            
            print(f"\n[Final Results] HellaSwag Accuracy: {accuracy:.4f} ({local_correct}/{local_total})")
            
            metrics = {
                'accuracy': accuracy,
                'total_samples': local_total,
                'correct_samples': local_correct
            }
        
        return all_results, metrics
        
    except Exception as e:
        print(f"[Rank {rank}] Error during evaluation: {e}")
        raise


# Alternative implementation using DistributedSampler (recommended for PyTorch)
def evaluate_hellaswag_with_sampler(model, data, batch_size, device='cuda'):
    """
    Alternative implementation using PyTorch's DistributedSampler
    This is the recommended approach for distributed data loading
    """
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    class HellaSwagDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Check distributed setup
    is_distributed = dist.is_initialized() if torch.distributed.is_available() else False
    
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Create dataset
    dataset = HellaSwagDataset(data)
    
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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler else False,
        num_workers=0,
        pin_memory=True
    )
    
    # Print info
    total_samples = len(dataset)
    if rank == 0:
        print(f"[Rank 0] Evaluating HellaSwag: {total_samples} total samples")
        if is_distributed:
            samples_per_rank = (total_samples + world_size - 1) // world_size
            print(f"[Rank 0]   Each rank processes up to {samples_per_rank} samples")
    
    # Evaluation loop
    all_results = []
    show_progress = (rank == 0) if is_distributed else True
    
    iterator = tqdm(dataloader, desc=f"HellaSwag Eval", leave=False) if show_progress else dataloader
    
    for batch in iterator:
        batch_results = evaluate_hellaswag_batch(model, batch, device)
        all_results.extend(batch_results)
    
    # Rest of the evaluation logic remains the same...
    return all_results


# Placeholder functions - replace with your actual implementations
def batch_data(data_list: List, batch_size: int = 1) -> List[List]:
    """Batch data into smaller chunks"""
    batches = []
    for i in range(0, len(data_list), batch_size):
        batches.append(data_list[i:i + batch_size])
    return batches

def evaluate_hellaswag_batch(model, batch_data: List[Dict[str, Any]], device: str = 'cuda') -> List[Dict[str, Any]]:
    """
    Placeholder for your actual batch evaluation function
    Replace this with your actual implementation
    """
    # Your actual evaluation logic here
    results = []
    for item in batch_data:
        # Process each item
        result = {
            'id': item.get('id'),
            'correct': True,  # Replace with actual evaluation
            # Add other fields as needed
        }
        results.append(result)
    return results