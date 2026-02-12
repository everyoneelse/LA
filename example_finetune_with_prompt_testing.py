#!/usr/bin/env python3
"""
Example script showing how to use the new prompt testing feature during finetuning.

This script demonstrates how to run finetuning with periodic prompt testing.
The prompts will be tested every N steps and results will be printed to console.
"""

import subprocess
import sys
import os

def run_finetune_with_prompt_testing():
    """
    Example of running finetuning with prompt testing enabled
    """
    
    # Example prompts to test during training
    test_prompts = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "Write a short poem about the ocean.",
        "How do you make a paper airplane?"
    ]
    
    # Base finetuning command
    cmd = [
        "python", "/workspace/accessory/main_finetune.py",
        
        # Model configuration
        "--llama_type", "llama2_7B",
        "--llama_config", "/path/to/your/config.json",
        "--tokenizer_path", "/path/to/your/tokenizer.model",
        "--pretrained_path", "/path/to/your/pretrained/model",
        
        # Training parameters
        "--batch_size", "4",
        "--accum_iter", "4",
        "--epochs", "3",
        "--lr", "2e-5",
        "--weight_decay", "0.02",
        "--warmup_epochs", "0.1",
        
        # Data configuration
        "--data_config", "/path/to/your/data_config.yaml",
        "--max_words", "1024",
        
        # Output configuration
        "--output_dir", "./output_finetune_with_prompts",
        "--save_interval", "1",
        "--save_iteration_interval", "1000",
        
        # Prompt testing configuration (NEW FEATURES)
        "--test_prompt_interval", "100",  # Test every 100 steps
        "--test_prompt_max_gen_len", "64",
        "--test_prompt_temperature", "0.1",
        "--test_prompt_top_p", "0.9",
        
        # Distributed training
        "--precision", "bf16",
        "--data_parallel", "fsdp",
    ]
    
    # Add test prompts to command
    cmd.extend(["--test_prompts"] + test_prompts)
    
    print("Running finetuning with prompt testing...")
    print("Command:", " ".join(cmd))
    print("\nTest prompts that will be evaluated:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  {i}. {prompt}")
    
    print(f"\nPrompts will be tested every 100 training steps.")
    print("Results will be printed to console during training.\n")
    
    # Uncomment the line below to actually run the command
    # subprocess.run(cmd)
    
    print("Note: Uncomment the subprocess.run(cmd) line to actually execute the training.")

if __name__ == "__main__":
    run_finetune_with_prompt_testing()