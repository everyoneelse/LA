#!/usr/bin/env python3
"""
Example script demonstrating pretraining with token and FLOP tracking.

This script shows how to use the enhanced pretraining functionality that tracks:
1. Number of processed tokens per iteration and cumulatively
2. FLOPs (floating point operations) per iteration and cumulatively

Usage:
    python example_pretrain_with_tracking.py \
        --llama_config /path/to/config.json \
        --tokenizer_path /path/to/tokenizer.model \
        --data_meta_path /path/to/data_meta.json \
        --data_root /path/to/data \
        --output_dir ./output \
        --batch_size 4 \
        --accum_iter 4 \
        --lr 0.001 \
        --max_words 2048
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from accessory.main_pretrain import main, get_args_parser

if __name__ == '__main__':
    print("=" * 80)
    print("LLaMA2-Accessory Pretraining with Token & FLOP Tracking")
    print("=" * 80)
    print()
    print("This enhanced version will track and display:")
    print("  • Number of tokens processed per batch and cumulatively")
    print("  • FLOPs (floating point operations) per batch and cumulatively")
    print("  • Statistics are printed every 10 iterations")
    print("  • All metrics are logged to TensorBoard if output_dir is specified")
    print()
    print("Example output during training:")
    print("  [Iter 10] Tokens: 81.92K tokens (batch), 819.2K tokens (total) | FLOPs: 2.34 TFLOPs (batch), 23.4 TFLOPs (total)")
    print()
    print("=" * 80)
    print()
    
    # Parse arguments and run main training function
    args = get_args_parser()
    args = args.parse_args()
    
    # Validate required arguments
    if not args.llama_config:
        print("Error: --llama_config is required")
        sys.exit(1)
    if not args.tokenizer_path:
        print("Error: --tokenizer_path is required")
        sys.exit(1)
    if not args.data_meta_path:
        print("Error: --data_meta_path is required")
        sys.exit(1)
    
    if args.output_dir:
        from pathlib import Path
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run the main training function
    main(args)