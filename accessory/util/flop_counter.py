import torch
import torch.nn as nn
from typing import Dict, Any
import math


class FLOPCounter:
    """
    Utility class to calculate FLOPs for transformer models during training.
    Based on the standard transformer architecture calculations.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize FLOP counter with model configuration.
        
        Args:
            model_config: Dictionary containing model parameters like:
                - vocab_size: vocabulary size
                - dim: model dimension (hidden size)
                - n_layers: number of transformer layers
                - n_heads: number of attention heads
                - multiple_of: for feed-forward dimension calculation
        """
        self.vocab_size = model_config.get('vocab_size', 32000)
        self.dim = model_config.get('dim', 4096)
        self.n_layers = model_config.get('n_layers', 32)
        self.n_heads = model_config.get('n_heads', 32)
        self.multiple_of = model_config.get('multiple_of', 256)
        
        # Calculate feed-forward dimension
        hidden_dim = int(2 * self.dim * 4 / 3)
        self.ffn_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
        
        self.head_dim = self.dim // self.n_heads
        
    def calculate_forward_flops(self, batch_size: int, seq_len: int) -> int:
        """
        Calculate FLOPs for a forward pass.
        
        Args:
            batch_size: batch size
            seq_len: sequence length
            
        Returns:
            Total FLOPs for forward pass
        """
        flops = 0
        
        # Token embedding lookup (no FLOPs, just memory access)
        
        # For each transformer layer
        for _ in range(self.n_layers):
            # Multi-head attention
            # Q, K, V projections: 3 * (batch_size * seq_len * dim * dim)
            flops += 3 * batch_size * seq_len * self.dim * self.dim
            
            # Attention computation: batch_size * n_heads * seq_len * seq_len * head_dim
            flops += batch_size * self.n_heads * seq_len * seq_len * self.head_dim
            
            # Attention output: batch_size * seq_len * n_heads * head_dim * seq_len
            flops += batch_size * seq_len * self.n_heads * self.head_dim * seq_len
            
            # Output projection: batch_size * seq_len * dim * dim
            flops += batch_size * seq_len * self.dim * self.dim
            
            # Feed-forward network
            # First linear layer: batch_size * seq_len * dim * ffn_dim
            flops += batch_size * seq_len * self.dim * self.ffn_dim
            
            # Second linear layer: batch_size * seq_len * ffn_dim * dim
            flops += batch_size * seq_len * self.ffn_dim * self.dim
            
            # Layer norms (negligible compared to linear layers)
            
        # Final layer norm and output projection
        flops += batch_size * seq_len * self.dim * self.vocab_size
        
        return flops
    
    def calculate_backward_flops(self, batch_size: int, seq_len: int) -> int:
        """
        Calculate FLOPs for backward pass (approximately 2x forward pass).
        
        Args:
            batch_size: batch size
            seq_len: sequence length
            
        Returns:
            Total FLOPs for backward pass
        """
        return 2 * self.calculate_forward_flops(batch_size, seq_len)
    
    def calculate_total_flops(self, batch_size: int, seq_len: int) -> int:
        """
        Calculate total FLOPs for one training step (forward + backward).
        
        Args:
            batch_size: batch size
            seq_len: sequence length
            
        Returns:
            Total FLOPs for one training step
        """
        forward_flops = self.calculate_forward_flops(batch_size, seq_len)
        backward_flops = self.calculate_backward_flops(batch_size, seq_len)
        return forward_flops + backward_flops
    
    @staticmethod
    def format_flops(flops: int) -> str:
        """
        Format FLOPs in human-readable format.
        
        Args:
            flops: number of FLOPs
            
        Returns:
            Formatted string (e.g., "1.23 TFLOPs")
        """
        if flops >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops / 1e3:.2f} KFLOPs"
        else:
            return f"{flops} FLOPs"


def get_model_config_from_model(model) -> Dict[str, Any]:
    """
    Extract model configuration from a MetaModel instance.
    
    Args:
        model: MetaModel instance
        
    Returns:
        Dictionary with model configuration
    """
    try:
        # Access the inner LLM model
        llm_model = model.llma
        args = llm_model.args
        
        config = {
            'vocab_size': getattr(args, 'vocab_size', 32000),
            'dim': getattr(args, 'dim', 4096),
            'n_layers': getattr(args, 'n_layers', 32),
            'n_heads': getattr(args, 'n_heads', 32),
            'multiple_of': getattr(args, 'multiple_of', 256),
        }
        
        return config
    except Exception as e:
        print(f"Warning: Could not extract model config, using defaults. Error: {e}")
        return {
            'vocab_size': 32000,
            'dim': 4096,
            'n_layers': 32,
            'n_heads': 32,
            'multiple_of': 256,
        }


class TokenCounter:
    """
    Utility class to track processed tokens during training.
    """
    
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = 0
        
    def update(self, batch_size: int, seq_len: int, padding_mask=None):
        """
        Update token count for current batch.
        
        Args:
            batch_size: batch size
            seq_len: sequence length
            padding_mask: optional mask to exclude padding tokens
        """
        if padding_mask is not None:
            # Count only non-padding tokens
            self.batch_tokens = padding_mask.sum().item()
        else:
            # Count all tokens (including padding)
            self.batch_tokens = batch_size * seq_len
            
        self.total_tokens += self.batch_tokens
    
    def get_total_tokens(self) -> int:
        """Get total processed tokens."""
        return self.total_tokens
    
    def get_batch_tokens(self) -> int:
        """Get tokens in current batch."""
        return self.batch_tokens
    
    @staticmethod
    def format_tokens(tokens: int) -> str:
        """
        Format token count in human-readable format.
        
        Args:
            tokens: number of tokens
            
        Returns:
            Formatted string (e.g., "1.23M tokens")
        """
        if tokens >= 1e9:
            return f"{tokens / 1e9:.2f}B tokens"
        elif tokens >= 1e6:
            return f"{tokens / 1e6:.2f}M tokens"
        elif tokens >= 1e3:
            return f"{tokens / 1e3:.2f}K tokens"
        else:
            return f"{tokens} tokens"