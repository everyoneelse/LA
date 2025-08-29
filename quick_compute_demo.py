#!/usr/bin/env python3
"""
Quick Compute Demo for InternLM2 Models

This script provides a simple interface to quickly calculate training compute
for InternLM2 models using different methods.
"""

import json
from pathlib import Path
from internlm2_compute_calculator import InternLM2ComputeCalculator


def quick_demo():
    """Quick demonstration of compute calculation."""
    
    print("🚀 InternLM2 Training Compute Quick Demo")
    print("=" * 50)
    
    # Load a sample config
    config_path = "internlm2_scaling/configs/internlm2-chat-1386M-h16-L16.json"
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        print("Please ensure the config files are available.")
        return
    
    # Initialize calculator
    calculator = InternLM2ComputeCalculator(config_path=config_path)
    
    print(f"📊 Model: InternLM2-1.4B")
    print(f"📈 Parameters: {calculator.format_number(calculator.total_params, ' params')}")
    print()
    
    # Scenario 1: Quick estimate with common settings
    print("🔍 Scenario 1: Common Training Setup")
    print("-" * 30)
    
    batch_size = 4
    seq_len = 2048
    total_tokens = 50_000_000_000  # 50B tokens
    
    results = calculator.calculate_training_compute(
        batch_size=batch_size,
        seq_len=seq_len,
        total_tokens=total_tokens
    )
    
    print(f"Training config: batch_size={batch_size}, seq_len={seq_len}")
    print(f"Total tokens: {calculator.format_number(total_tokens, ' tokens')}")
    print(f"Training steps: {calculator.format_number(results['total_steps'], ' steps')}")
    print()
    
    print("💾 Compute Estimates:")
    print(f"  OpenAI Scaling Law: {calculator.format_number(results['compute_estimates']['openai_scaling_law'], 'FLOPs')}")
    print(f"  Detailed FLOPs:    {calculator.format_number(results['compute_estimates']['detailed_flops'], 'FLOPs')}")
    print()
    
    # Scenario 2: Chinchilla optimal
    print("🔍 Scenario 2: Chinchilla Optimal Training")
    print("-" * 30)
    
    chinchilla_tokens = calculator.calculate_chinchilla_optimal_tokens()
    results_optimal = calculator.calculate_training_compute(
        batch_size=batch_size,
        seq_len=seq_len,
        total_tokens=chinchilla_tokens
    )
    
    print(f"Chinchilla optimal tokens: {calculator.format_number(chinchilla_tokens, ' tokens')}")
    print(f"Training steps: {calculator.format_number(results_optimal['total_steps'], ' steps')}")
    print(f"Compute needed: {calculator.format_number(results_optimal['compute_estimates']['openai_scaling_law'], 'FLOPs')}")
    print()
    
    # Scenario 3: Different batch sizes comparison
    print("🔍 Scenario 3: Batch Size Impact")
    print("-" * 30)
    
    fixed_tokens = 10_000_000_000  # 10B tokens
    batch_sizes = [1, 4, 16, 64]
    
    print("Batch Size | Steps | FLOPs per Step | Total Compute")
    print("-" * 50)
    
    for bs in batch_sizes:
        result = calculator.calculate_training_compute(
            batch_size=bs,
            seq_len=seq_len,
            total_tokens=fixed_tokens
        )
        
        steps = result['total_steps']
        flops_per_step = result['flops_breakdown']['total_flops_per_step']
        total_compute = result['compute_estimates']['detailed_flops']
        
        print(f"{bs:10} | {calculator.format_number(steps):5} | {calculator.format_number(flops_per_step, 'FLOPs'):12} | {calculator.format_number(total_compute, 'FLOPs')}")
    
    print()
    
    # Practical insights
    print("💡 Practical Insights:")
    print("-" * 20)
    print("• OpenAI Scaling Law provides a quick 6×N×D estimate")
    print("• Detailed calculation is ~2x lower due to architectural optimizations")  
    print("• Batch size doesn't affect total compute, only training time")
    print("• Chinchilla optimal: ~20 tokens per parameter")
    print("• Consider 1.5-2x safety margin for resource planning")


def compare_models():
    """Compare compute requirements across different model sizes."""
    
    print("\n🔍 Model Size Comparison")
    print("=" * 50)
    
    config_dir = Path("internlm2_scaling/configs")
    configs = []
    
    # Load all configs
    for json_file in config_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            config = json.load(f)
            configs.append((json_file.stem, config))
    
    # Sort by parameter count
    configs_with_params = []
    for name, config in configs:
        calc = InternLM2ComputeCalculator(config_dict=config)
        configs_with_params.append((name, config, calc.total_params))
    
    configs_with_params.sort(key=lambda x: x[2])
    
    # Training setup
    batch_size = 4
    seq_len = 2048
    
    print(f"Training setup: batch_size={batch_size}, seq_len={seq_len}")
    print("Using Chinchilla optimal token counts\n")
    
    print("Model | Params | Optimal Tokens | Compute (OpenAI) | Training Days*")
    print("-" * 75)
    
    for name, config, params in configs_with_params:
        calc = InternLM2ComputeCalculator(config_dict=config)
        optimal_tokens = calc.calculate_chinchilla_optimal_tokens()
        
        result = calc.calculate_training_compute(
            batch_size=batch_size,
            seq_len=seq_len,
            total_tokens=optimal_tokens
        )
        
        compute = result['compute_estimates']['openai_scaling_law']
        
        # Estimate training time (assuming 100 TFLOPs/s effective throughput)
        seconds = compute / (100e12)  # 100 TFLOPs/s
        days = seconds / (24 * 3600)
        
        model_short = name.replace('internlm2-chat-', '').replace('.json', '')
        print(f"{model_short:5} | {calc.format_number(params):6} | {calc.format_number(optimal_tokens, ' tokens'):12} | {calc.format_number(compute, 'FLOPs'):14} | {days:.1f}")
    
    print("\n* Estimated days assuming 100 TFLOPs/s effective throughput")
    print("  (actual throughput depends on hardware and efficiency)")


if __name__ == "__main__":
    try:
        quick_demo()
        compare_models()
        
        print("\n✅ Demo completed successfully!")
        print("\n📖 For more details, see:")
        print("  - COMPUTE_CALCULATION_RESEARCH.md (detailed research)")
        print("  - internlm2_compute_calculator.py (full calculator)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. Config files exist in internlm2_scaling/configs/")
        print("2. internlm2_compute_calculator.py is in the same directory")