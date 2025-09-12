#!/usr/bin/env python
"""
Test script to verify rank information is correctly displayed
Run with: torchrun --nproc_per_node=2 test_rank_info.py
"""
import os
import torch
import torch.distributed as dist
import time


def test_rank_display():
    """Test if rank information is displayed correctly"""
    
    # Initialize distributed
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        is_distributed = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        is_distributed = False
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Print various rank information
    print(f"[Rank {rank}] Process started", flush=True)
    print(f"[Rank {rank}] is_distributed: {is_distributed}", flush=True)
    print(f"[Rank {rank}] world_size: {world_size}", flush=True)
    print(f"[Rank {rank}] local_rank: {local_rank}", flush=True)
    print(f"[Rank {rank}] PID: {os.getpid()}", flush=True)
    
    # Test with timestamp
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}][Rank {rank}] Testing timestamp display", flush=True)
    
    # Sync
    if is_distributed:
        dist.barrier()
        print(f"[Rank {rank}] Barrier passed", flush=True)
    
    # Test without rank prefix (what you're seeing)
    print(f"Testing without rank prefix at rank {rank}", flush=True)
    
    # Clean up
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    test_rank_display()