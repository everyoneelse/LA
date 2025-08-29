#!/usr/bin/env python3
"""
InternLM2 Training Compute Calculator

This script calculates the training compute (FLOPs) for InternLM2 models based on:
1. OpenAI Scaling Laws paper methodology
2. Detailed FLOPs calculation for Transformer models
3. InternLM2 specific architecture configurations

References:
- "Scaling Laws for Neural Language Models" (OpenAI, 2020)
- "Training Compute-Optimal Large Language Models" (Chinchilla paper, 2022)
"""

import json
import csv
import math
from typing import Dict, Any, List, Tuple
from pathlib import Path


class InternLM2ComputeCalculator:
    """
    Calculator for InternLM2 training compute based on model configuration.
    
    Implements multiple methods:
    1. OpenAI Scaling Law approximation: C ≈ 6 × N × D
    2. Detailed FLOPs calculation for forward/backward passes
    3. Chinchilla optimal compute calculation
    """
    
    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        """
        Initialize calculator with model configuration.
        
        Args:
            config_path: Path to InternLM2 config JSON file
            config_dict: Direct config dictionary (alternative to config_path)
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
            
        # Extract key parameters
        self.vocab_size = self.config.get('vocab_size', 92544)
        self.hidden_size = self.config.get('hidden_size', 512)
        self.num_layers = self.config.get('num_hidden_layers', 8)
        self.num_heads = self.config.get('num_attention_heads', 4)
        self.num_kv_heads = self.config.get('num_key_value_heads', 2)  # For GQA
        self.intermediate_size = self.config.get('intermediate_size', 2048)
        self.max_position_embeddings = self.config.get('max_position_embeddings', 32768)
        
        # Calculate derived parameters
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_head_dim = self.hidden_size // self.num_kv_heads if self.num_kv_heads else self.head_dim
        
        # Calculate total parameters
        self.total_params = self._calculate_parameters()
    
    def _calculate_parameters(self) -> int:
        """Calculate total number of parameters in the model."""
        params = 0
        
        # Token embeddings
        params += self.vocab_size * self.hidden_size
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Multi-head attention
            # Q, K, V projections (considering GQA)
            params += self.hidden_size * self.hidden_size  # Q projection
            params += self.hidden_size * (self.num_kv_heads * self.kv_head_dim)  # K projection
            params += self.hidden_size * (self.num_kv_heads * self.kv_head_dim)  # V projection
            params += self.hidden_size * self.hidden_size  # Output projection
            
            # Feed-forward network
            params += self.hidden_size * self.intermediate_size  # Up projection
            params += self.intermediate_size * self.hidden_size  # Down projection
            
            # Layer norms (2 per layer: pre-attention and pre-ffn)
            params += 2 * self.hidden_size
        
        # Final layer norm
        params += self.hidden_size
        
        # Output projection (lm_head) - often tied with embeddings
        if not self.config.get('tie_word_embeddings', False):
            params += self.hidden_size * self.vocab_size
            
        return params
    
    def calculate_forward_flops(self, batch_size: int, seq_len: int) -> int:
        """
        Calculate FLOPs for forward pass.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Total FLOPs for forward pass
        """
        flops = 0
        
        # Token embeddings (no computation, just lookup)
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Multi-head attention with Grouped Query Attention (GQA)
            
            # Q projection: batch_size * seq_len * hidden_size * hidden_size
            flops += batch_size * seq_len * self.hidden_size * self.hidden_size
            
            # K, V projections (GQA - fewer KV heads)
            kv_proj_size = self.num_kv_heads * self.kv_head_dim
            flops += batch_size * seq_len * self.hidden_size * kv_proj_size  # K
            flops += batch_size * seq_len * self.hidden_size * kv_proj_size  # V
            
            # Attention computation: Q @ K^T
            # Shape: (batch, heads, seq_len, head_dim) @ (batch, kv_heads, head_dim, seq_len)
            # With GQA, we need to account for head grouping
            group_size = self.num_heads // self.num_kv_heads
            flops += batch_size * self.num_heads * seq_len * seq_len * self.head_dim
            
            # Attention weights @ V
            flops += batch_size * self.num_heads * seq_len * self.head_dim * seq_len
            
            # Output projection
            flops += batch_size * seq_len * self.hidden_size * self.hidden_size
            
            # Feed-forward network
            # Up projection (with activation)
            flops += batch_size * seq_len * self.hidden_size * self.intermediate_size
            
            # Down projection
            flops += batch_size * seq_len * self.intermediate_size * self.hidden_size
            
            # Layer norms (negligible compared to linear layers)
        
        # Final layer norm and output projection
        flops += batch_size * seq_len * self.hidden_size * self.vocab_size
        
        return flops
    
    def calculate_backward_flops(self, batch_size: int, seq_len: int) -> int:
        """
        Calculate FLOPs for backward pass (approximately 2x forward pass).
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Total FLOPs for backward pass
        """
        return 2 * self.calculate_forward_flops(batch_size, seq_len)
    
    def calculate_total_flops_per_step(self, batch_size: int, seq_len: int) -> int:
        """
        Calculate total FLOPs for one training step (forward + backward).
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Total FLOPs for one training step
        """
        forward_flops = self.calculate_forward_flops(batch_size, seq_len)
        backward_flops = self.calculate_backward_flops(batch_size, seq_len)
        return forward_flops + backward_flops
    
    def calculate_openai_scaling_law_compute(self, total_tokens: int) -> int:
        """
        Calculate compute using OpenAI Scaling Law approximation: C ≈ 6 × N × D
        
        Args:
            total_tokens: Total number of training tokens
            
        Returns:
            Total compute in FLOPs
        """
        return 6 * self.total_params * total_tokens
    
    def calculate_chinchilla_optimal_tokens(self) -> int:
        """
        Calculate optimal number of tokens according to Chinchilla scaling laws.
        Chinchilla suggests approximately 20 tokens per parameter.
        
        Returns:
            Optimal number of training tokens
        """
        return 20 * self.total_params
    
    def calculate_training_compute(self, 
                                 batch_size: int, 
                                 seq_len: int, 
                                 total_steps: int = None,
                                 total_tokens: int = None) -> Dict[str, Any]:
        """
        Calculate training compute using different methods.
        
        Args:
            batch_size: Training batch size
            seq_len: Sequence length
            total_steps: Total training steps (alternative to total_tokens)
            total_tokens: Total training tokens (alternative to total_steps)
            
        Returns:
            Dictionary with compute estimates from different methods
        """
        if total_tokens is None and total_steps is not None:
            total_tokens = total_steps * batch_size * seq_len
        elif total_tokens is None and total_steps is None:
            # Use Chinchilla optimal
            total_tokens = self.calculate_chinchilla_optimal_tokens()
            
        if total_steps is None:
            total_steps = total_tokens // (batch_size * seq_len)
        
        # Method 1: OpenAI Scaling Law
        openai_compute = self.calculate_openai_scaling_law_compute(total_tokens)
        
        # Method 2: Detailed FLOPs calculation
        flops_per_step = self.calculate_total_flops_per_step(batch_size, seq_len)
        detailed_compute = flops_per_step * total_steps
        
        # Method 3: Per-token FLOPs (6N per token as per scaling laws)
        per_token_flops = 6 * self.total_params
        token_based_compute = per_token_flops * total_tokens
        
        return {
            'model_params': self.total_params,
            'total_tokens': total_tokens,
            'total_steps': total_steps,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'tokens_per_step': batch_size * seq_len,
            'chinchilla_optimal_tokens': self.calculate_chinchilla_optimal_tokens(),
            'compute_estimates': {
                'openai_scaling_law': openai_compute,
                'detailed_flops': detailed_compute,
                'token_based_6n': token_based_compute,
            },
            'flops_breakdown': {
                'forward_flops_per_step': self.calculate_forward_flops(batch_size, seq_len),
                'backward_flops_per_step': self.calculate_backward_flops(batch_size, seq_len),
                'total_flops_per_step': flops_per_step,
                'flops_per_token': per_token_flops,
            }
        }
    
    @staticmethod
    def format_number(num: int, unit: str = "") -> str:
        """Format large numbers in human-readable format."""
        if num >= 1e12:
            return f"{num / 1e12:.2f}T{unit}"
        elif num >= 1e9:
            return f"{num / 1e9:.2f}G{unit}"
        elif num >= 1e6:
            return f"{num / 1e6:.2f}M{unit}"
        elif num >= 1e3:
            return f"{num / 1e3:.2f}K{unit}"
        else:
            return f"{num}{unit}"


def load_all_configs(config_dir: str = "internlm2_scaling/configs") -> List[Tuple[str, Dict[str, Any]]]:
    """Load all InternLM2 configuration files."""
    config_path = Path(config_dir)
    configs = []
    
    for json_file in config_path.glob("*.json"):
        if json_file.name != "variants_summary.csv":
            with open(json_file, 'r') as f:
                config = json.load(f)
                configs.append((json_file.stem, config))
    
    return configs


def analyze_all_models(config_dir: str = "internlm2_scaling/configs",
                      batch_size: int = 4,
                      seq_len: int = 2048,
                      use_chinchilla_optimal: bool = True) -> None:
    """Analyze compute requirements for all InternLM2 model variants."""
    
    print("=" * 100)
    print("InternLM2 Training Compute Analysis")
    print("=" * 100)
    print(f"Training configuration: batch_size={batch_size}, seq_len={seq_len}")
    print(f"Using Chinchilla optimal tokens: {use_chinchilla_optimal}")
    print()
    
    configs = load_all_configs(config_dir)
    
    for model_name, config in configs:
        print(f"Model: {model_name}")
        print("-" * 60)
        
        calculator = InternLM2ComputeCalculator(config_dict=config)
        
        if use_chinchilla_optimal:
            total_tokens = calculator.calculate_chinchilla_optimal_tokens()
        else:
            # Use a fixed number of tokens for comparison
            total_tokens = 100_000_000_000  # 100B tokens
        
        results = calculator.calculate_training_compute(
            batch_size=batch_size,
            seq_len=seq_len,
            total_tokens=total_tokens
        )
        
        print(f"Parameters: {calculator.format_number(results['model_params'], 'params')}")
        print(f"Training tokens: {calculator.format_number(results['total_tokens'], ' tokens')}")
        print(f"Training steps: {calculator.format_number(results['total_steps'], ' steps')}")
        print()
        
        print("Compute estimates:")
        for method, compute in results['compute_estimates'].items():
            print(f"  {method:20}: {calculator.format_number(compute, 'FLOPs')}")
        print()
        
        print("FLOPs breakdown:")
        for component, flops in results['flops_breakdown'].items():
            print(f"  {component:25}: {calculator.format_number(flops, 'FLOPs')}")
        print()
        
        print("=" * 100)


if __name__ == "__main__":
    # Example usage
    print("InternLM2 Compute Calculator")
    print("=" * 50)
    
    # Analyze a specific model
    config_path = "internlm2_scaling/configs/internlm2-chat-1386M-h16-L16.json"
    
    try:
        calculator = InternLM2ComputeCalculator(config_path=config_path)
        
        # Example training configuration
        batch_size = 8
        seq_len = 2048
        total_tokens = 100_000_000_000  # 100B tokens
        
        results = calculator.calculate_training_compute(
            batch_size=batch_size,
            seq_len=seq_len,
            total_tokens=total_tokens
        )
        
        print(f"Model: {config_path}")
        print(f"Parameters: {calculator.format_number(results['model_params'], ' params')}")
        print(f"Training configuration: batch_size={batch_size}, seq_len={seq_len}")
        print(f"Total tokens: {calculator.format_number(total_tokens, ' tokens')}")
        print()
        
        print("Compute Estimates:")
        for method, compute in results['compute_estimates'].items():
            print(f"  {method}: {calculator.format_number(compute, 'FLOPs')}")
        print()
        
        # Analyze all models
        print("\nAnalyzing all models...")
        analyze_all_models()
        
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Please ensure the internlm2_scaling/configs directory exists with JSON config files.")
        
        # Show available configs
        try:
            configs = load_all_configs()
            print(f"\nAvailable configs ({len(configs)}):")
            for name, _ in configs:
                print(f"  - {name}")
        except:
            print("No config files found.")