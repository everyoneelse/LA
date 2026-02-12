#!/usr/bin/env python
"""
Example script showing how to use the new FSDP-compatible HellaSwag evaluation

This demonstrates three different ways to integrate HellaSwag evaluation:
1. Direct function call (simplest)
2. Using DataLoader with dynamic padding (more efficient)
3. Full integration with training loop
"""

import torch
import torch.distributed as dist
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


def example_1_direct_evaluation():
    """
    Example 1: Direct evaluation using the simple FSDP-compatible function
    This is the easiest way to integrate into existing code
    """
    print("=" * 60)
    print("Example 1: Direct FSDP-compatible evaluation")
    print("=" * 60)
    
    from accessory.util.hellaswag_eval_fsdp import run_hellaswag_evaluation_fsdp
    
    # Assuming model is already loaded and wrapped with FSDP
    # model = your_fsdp_model
    
    # Run evaluation
    metrics = run_hellaswag_evaluation_fsdp(
        model=model,  # Your FSDP-wrapped model
        data_dir='data/hellaswag/',
        tokenizer=tokenizer,  # Your tokenizer
        batch_size=4,
        max_samples=100,  # Use small number for testing
        max_length=512,
        device='cuda'
    )
    
    print(f"Results: {metrics}")
    return metrics


def example_2_dataloader_evaluation():
    """
    Example 2: Using DataLoader with dynamic padding for better efficiency
    """
    print("=" * 60)
    print("Example 2: DataLoader with dynamic padding")
    print("=" * 60)
    
    from accessory.data.hellaswag_dataset_improved import (
        create_hellaswag_dataloader,
        HellaSwagDatasetImproved
    )
    from accessory.util.hellaswag_eval_fsdp import process_hellaswag_batch_fsdp
    
    # Check if distributed
    is_distributed = dist.is_initialized()
    if is_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Create DataLoader with dynamic padding
    dataloader = create_hellaswag_dataloader(
        data_file='data/hellaswag/hellaswag_val.jsonl',
        tokenizer=tokenizer,
        batch_size=4,
        max_length=512,
        max_samples=100,
        distributed=is_distributed,
        world_size=world_size,
        rank=rank,
        use_dynamic_padding=True,  # Efficient padding
        group_by_length=True       # Group similar lengths
    )
    
    # Evaluation loop
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, labels, metadata) in enumerate(dataloader):
            # Process batch
            predictions = process_hellaswag_batch_fsdp(
                model, input_ids, attention_mask, labels, metadata, device='cuda'
            )
            
            # Collect results
            for pred, meta in zip(predictions, metadata):
                all_predictions.append(pred)
                if meta['label'] >= 0:
                    all_labels.append(meta['label'])
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy


def example_3_training_integration():
    """
    Example 3: Full integration with training loop
    Shows how to add HellaSwag evaluation to your training script
    """
    print("=" * 60)
    print("Example 3: Training loop integration")
    print("=" * 60)
    
    # In your training script (e.g., main_pretrain.py):
    
    # 1. Add command line arguments
    parser.add_argument('--hellaswag_eval', action='store_true',
                       help='Enable HellaSwag evaluation')
    parser.add_argument('--hellaswag_data_dir', type=str, 
                       default='data/hellaswag/',
                       help='HellaSwag data directory')
    parser.add_argument('--hellaswag_batch_size', type=int, default=4,
                       help='Batch size for HellaSwag evaluation')
    parser.add_argument('--hellaswag_max_samples', type=int, default=None,
                       help='Max samples for HellaSwag (None = all)')
    parser.add_argument('--hellaswag_eval_freq', type=int, default=5000,
                       help='Evaluate HellaSwag every N steps')
    
    # 2. In your training loop
    for epoch in range(start_epoch, num_epochs):
        for step, batch in enumerate(train_dataloader):
            # ... training step ...
            
            # Periodic HellaSwag evaluation
            if args.hellaswag_eval and step % args.hellaswag_eval_freq == 0:
                from accessory.util.hellaswag_eval_fsdp import run_hellaswag_evaluation_fsdp
                
                metrics = run_hellaswag_evaluation_fsdp(
                    model=model,
                    data_dir=args.hellaswag_data_dir,
                    tokenizer=tokenizer,
                    batch_size=args.hellaswag_batch_size,
                    max_samples=args.hellaswag_max_samples,
                    max_length=args.max_words,
                    device='cuda'
                )
                
                if dist.get_rank() == 0:
                    print(f"Step {step}: HellaSwag Accuracy = {metrics['accuracy']:.4f}")
                    # Log to tensorboard/wandb
                    if logger is not None:
                        logger.log({'hellaswag/accuracy': metrics['accuracy']}, step=step)
                
                # Return to training mode
                model.train()


def example_4_standalone_test():
    """
    Example 4: Standalone test script for debugging
    """
    print("=" * 60)
    print("Example 4: Standalone debugging test")
    print("=" * 60)
    
    import torch
    from accessory.model.meta import MetaModel
    from accessory.util.hellaswag_eval_fsdp import run_hellaswag_evaluation_fsdp
    
    # Load model (example with LLaMA)
    model = MetaModel.from_pretrained(
        'path/to/checkpoint',
        'path/to/params.json',
        tokenizer_path='path/to/tokenizer.model'
    )
    
    # Wrap with FSDP if needed
    if dist.is_initialized():
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model, ...)
    
    # Test on small subset
    metrics = run_hellaswag_evaluation_fsdp(
        model=model,
        data_dir='data/hellaswag/',
        tokenizer=model.tokenizer,
        batch_size=2,
        max_samples=10,  # Very small for testing
        max_length=256,
        device='cuda'
    )
    
    print(f"Test results: {metrics}")
    
    # Verify no hanging
    if dist.is_initialized():
        dist.barrier()
        print(f"Rank {dist.get_rank()}: Passed barrier test!")


if __name__ == "__main__":
    """
    Run examples based on command line argument
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, default=1, 
                       choices=[1, 2, 3, 4],
                       help='Which example to run')
    args = parser.parse_args()
    
    print(f"Running Example {args.example}")
    
    # Note: These examples assume you have:
    # 1. A model loaded and available
    # 2. A tokenizer initialized
    # 3. HellaSwag data downloaded
    
    # For actual use, replace with your model/tokenizer
    model = None  # Your model here
    tokenizer = None  # Your tokenizer here
    
    if args.example == 1:
        example_1_direct_evaluation()
    elif args.example == 2:
        example_2_dataloader_evaluation()
    elif args.example == 3:
        example_3_training_integration()
    elif args.example == 4:
        example_4_standalone_test()
    
    print("\nExample completed!")