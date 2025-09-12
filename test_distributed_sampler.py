#!/usr/bin/env python
"""
Test script to verify DistributedSampler behavior
Run with: torchrun --nproc_per_node=2 test_distributed_sampler.py
"""
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class SimpleDataset(Dataset):
    def __init__(self, size=100):
        self.data = list(range(size))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def main():
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create dataset
    dataset = SimpleDataset(size=100)
    
    # Create DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        sampler=sampler
    )
    
    # Collect all data processed by this rank
    rank_data = []
    for batch in dataloader:
        rank_data.extend(batch.tolist())
    
    print(f"Rank {rank} processed {len(rank_data)} items:")
    print(f"  First 10: {rank_data[:10]}")
    print(f"  Last 10: {rank_data[-10:]}")
    
    # Verify no overlap
    all_data = [None] * world_size
    dist.all_gather_object(all_data, rank_data)
    
    if rank == 0:
        print("\n" + "="*50)
        print("Verification:")
        
        # Check total count
        total = sum(len(d) for d in all_data)
        print(f"Total items processed: {total} (expected: {len(dataset)})")
        
        # Check for duplicates
        all_items = []
        for i, data in enumerate(all_data):
            print(f"Rank {i}: {len(data)} items")
            all_items.extend(data)
        
        unique_items = set(all_items)
        print(f"Unique items: {len(unique_items)}")
        print(f"Duplicates: {len(all_items) - len(unique_items)}")
        
        if len(unique_items) == len(dataset) and len(all_items) - len(unique_items) == 0:
            print("✅ SUCCESS: Each rank processed different data with no overlap!")
        else:
            print("❌ ERROR: Data overlap detected!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()