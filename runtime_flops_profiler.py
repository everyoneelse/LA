#!/usr/bin/env python3
"""
Runtime FLOPs Profiler for Actual Training

This module provides tools to measure actual FLOPs during training execution,
not theoretical estimates. It supports multiple measurement methods:
1. PyTorch Profiler with FLOPs tracking
2. Hook-based real-time monitoring
3. Custom operation counters
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from typing import Dict, Any, List, Optional, Callable
import time
import threading
from collections import defaultdict
import numpy as np


class FLOPsHook:
    """Hook to count FLOPs for specific operations during forward/backward pass."""
    
    def __init__(self):
        self.flop_counts = defaultdict(int)
        self.hooks = []
        self.current_step_flops = 0
        self.total_flops = 0
        
    def _linear_flop_count(self, module, input, output):
        """Count FLOPs for linear layers."""
        input_numel = input[0].numel()
        output_numel = output.numel()
        # For linear layer: input_features * output_features * batch_size
        flops = input_numel * module.out_features
        self.flop_counts['linear'] += flops
        self.current_step_flops += flops
        return flops
    
    def _conv2d_flop_count(self, module, input, output):
        """Count FLOPs for 2D convolution layers."""
        batch_size, in_channels, input_height, input_width = input[0].shape
        output_dims = output.shape[2:]
        kernel_dims = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups
        
        filters_per_channel = out_channels // groups
        conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
        
        active_elements_count = batch_size * int(np.prod(output_dims))
        overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
        
        # Add bias flops if bias is used
        if module.bias is not None:
            bias_flops = out_channels * active_elements_count
            overall_conv_flops += bias_flops
            
        self.flop_counts['conv2d'] += overall_conv_flops
        self.current_step_flops += overall_conv_flops
        return overall_conv_flops
    
    def _attention_flop_count(self, module, input, output):
        """Count FLOPs for attention mechanisms (approximate)."""
        # This is for MultiheadAttention or similar
        if hasattr(module, 'num_heads') and hasattr(module, 'embed_dim'):
            batch_size, seq_len = input[0].shape[:2]
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            head_dim = embed_dim // num_heads
            
            # Q, K, V projections
            qkv_flops = 3 * batch_size * seq_len * embed_dim * embed_dim
            
            # Attention computation: Q @ K^T
            attention_flops = batch_size * num_heads * seq_len * seq_len * head_dim
            
            # Attention @ V
            attn_out_flops = batch_size * num_heads * seq_len * head_dim * seq_len
            
            # Output projection
            out_proj_flops = batch_size * seq_len * embed_dim * embed_dim
            
            total_attn_flops = qkv_flops + attention_flops + attn_out_flops + out_proj_flops
            self.flop_counts['attention'] += total_attn_flops
            self.current_step_flops += total_attn_flops
            return total_attn_flops
        
        return 0
    
    def _embedding_flop_count(self, module, input, output):
        """Count FLOPs for embedding layers (usually minimal)."""
        # Embeddings are typically lookups, minimal FLOPs
        flops = 0  # Lookup operations don't count as FLOPs
        return flops
    
    def register_hooks(self, model):
        """Register hooks for all supported layer types."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._linear_flop_count)
                self.hooks.append(hook)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                hook = module.register_forward_hook(self._conv2d_flop_count)
                self.hooks.append(hook)
            elif isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._attention_flop_count)
                self.hooks.append(hook)
    
    def reset_step_count(self):
        """Reset the current step FLOPs counter."""
        self.current_step_flops = 0
    
    def get_step_flops(self):
        """Get FLOPs for current step and add to total."""
        step_flops = self.current_step_flops
        self.total_flops += step_flops
        return step_flops
    
    def get_total_flops(self):
        """Get total accumulated FLOPs."""
        return self.total_flops
    
    def get_flop_breakdown(self):
        """Get breakdown of FLOPs by operation type."""
        return dict(self.flop_counts)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class PyTorchProfilerFLOPs:
    """Use PyTorch's built-in profiler to measure FLOPs."""
    
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda
        self.activities = [ProfilerActivity.CPU]
        if use_cuda and torch.cuda.is_available():
            self.activities.append(ProfilerActivity.CUDA)
    
    def profile_step(self, model, inputs, targets=None, optimizer=None):
        """Profile a single training step and return FLOPs."""
        
        with profile(
            activities=self.activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            use_cuda=self.use_cuda
        ) as prof:
            with record_function("forward_pass"):
                outputs = model(inputs)
            
            if targets is not None and optimizer is not None:
                with record_function("backward_pass"):
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    else:
                        # Assume outputs are logits, compute simple loss
                        loss = nn.functional.cross_entropy(outputs, targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Extract FLOPs from profiler results
        flops_info = self._extract_flops_from_profile(prof)
        return flops_info
    
    def _extract_flops_from_profile(self, prof):
        """Extract FLOPs information from profiler results."""
        events = prof.key_averages()
        total_flops = 0
        flop_breakdown = defaultdict(int)
        
        for event in events:
            if hasattr(event, 'flops') and event.flops > 0:
                total_flops += event.flops
                # Categorize by operation type
                op_name = event.key
                if 'linear' in op_name.lower() or 'addmm' in op_name.lower():
                    flop_breakdown['linear'] += event.flops
                elif 'conv' in op_name.lower():
                    flop_breakdown['convolution'] += event.flops
                elif 'bmm' in op_name.lower() or 'mm' in op_name.lower():
                    flop_breakdown['matrix_mult'] += event.flops
                else:
                    flop_breakdown['other'] += event.flops
        
        return {
            'total_flops': total_flops,
            'breakdown': dict(flop_breakdown),
            'profiler_table': events.table(sort_by="flops", row_limit=20)
        }


class RuntimeFLOPsMonitor:
    """Main class for monitoring FLOPs during actual training."""
    
    def __init__(self, model, method='hooks', use_cuda=True):
        """
        Initialize FLOPs monitor.
        
        Args:
            model: PyTorch model to monitor
            method: 'hooks' or 'profiler'
            use_cuda: Whether to include CUDA profiling
        """
        self.model = model
        self.method = method
        self.use_cuda = use_cuda
        
        if method == 'hooks':
            self.flop_counter = FLOPsHook()
            self.flop_counter.register_hooks(model)
        elif method == 'profiler':
            self.flop_counter = PyTorchProfilerFLOPs(use_cuda)
        else:
            raise ValueError("Method must be 'hooks' or 'profiler'")
        
        # Statistics
        self.step_count = 0
        self.step_flops_history = []
        self.total_flops = 0
        self.start_time = None
    
    def start_monitoring(self):
        """Start monitoring (mainly for timing)."""
        self.start_time = time.time()
        print(f"🚀 Started FLOPs monitoring using method: {self.method}")
    
    def measure_step(self, inputs, targets=None, optimizer=None):
        """
        Measure FLOPs for one training step.
        
        Args:
            inputs: Model inputs
            targets: Training targets (for loss computation)
            optimizer: Optimizer (for backward pass)
            
        Returns:
            Dictionary with FLOPs information
        """
        if self.method == 'hooks':
            return self._measure_step_hooks(inputs, targets, optimizer)
        elif self.method == 'profiler':
            return self._measure_step_profiler(inputs, targets, optimizer)
    
    def _measure_step_hooks(self, inputs, targets=None, optimizer=None):
        """Measure step using hooks method."""
        self.flop_counter.reset_step_count()
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Backward pass if optimizer provided
        if targets is not None and optimizer is not None:
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = nn.functional.cross_entropy(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        step_flops = self.flop_counter.get_step_flops()
        self.step_count += 1
        self.step_flops_history.append(step_flops)
        self.total_flops += step_flops
        
        return {
            'step_flops': step_flops,
            'total_flops': self.total_flops,
            'step_count': self.step_count,
            'breakdown': self.flop_counter.get_flop_breakdown(),
            'avg_flops_per_step': self.total_flops / self.step_count if self.step_count > 0 else 0
        }
    
    def _measure_step_profiler(self, inputs, targets=None, optimizer=None):
        """Measure step using profiler method."""
        flops_info = self.flop_counter.profile_step(inputs, targets, optimizer)
        
        step_flops = flops_info['total_flops']
        self.step_count += 1
        self.step_flops_history.append(step_flops)
        self.total_flops += step_flops
        
        return {
            'step_flops': step_flops,
            'total_flops': self.total_flops,
            'step_count': self.step_count,
            'breakdown': flops_info['breakdown'],
            'avg_flops_per_step': self.total_flops / self.step_count if self.step_count > 0 else 0,
            'profiler_details': flops_info.get('profiler_table', '')
        }
    
    def get_statistics(self):
        """Get comprehensive statistics."""
        if self.step_count == 0:
            return {"error": "No steps measured yet"}
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        stats = {
            'total_steps': self.step_count,
            'total_flops': self.total_flops,
            'avg_flops_per_step': self.total_flops / self.step_count,
            'flops_per_second': self.total_flops / elapsed_time if elapsed_time > 0 else 0,
            'elapsed_time_seconds': elapsed_time,
        }
        
        if len(self.step_flops_history) > 1:
            stats.update({
                'min_step_flops': min(self.step_flops_history),
                'max_step_flops': max(self.step_flops_history),
                'std_step_flops': np.std(self.step_flops_history),
            })
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        if self.method == 'hooks':
            self.flop_counter.remove_hooks()
    
    @staticmethod
    def format_flops(flops):
        """Format FLOPs in human-readable format."""
        if flops >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops / 1e3:.2f} KFLOPs"
        else:
            return f"{flops:.0f} FLOPs"


# Example usage and testing
def demo_flops_monitoring():
    """Demonstrate FLOPs monitoring on a simple model."""
    print("🧪 Demo: Runtime FLOPs Monitoring")
    print("=" * 50)
    
    # Create a simple model for demonstration
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=1000, d_model=512, nhead=8, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead,
                dim_feedforward=2048,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.output_proj = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            seq_len = x.size(1)
            x = self.embedding(x) + self.pos_encoding[:seq_len]
            x = self.transformer(x)
            return self.output_proj(x)
    
    # Initialize model and data
    model = SimpleTransformer()
    batch_size, seq_len = 4, 128
    inputs = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))
    optimizer = torch.optim.Adam(model.parameters())
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {inputs.shape}")
    print()
    
    # Test both methods
    for method in ['hooks', 'profiler']:
        print(f"Testing method: {method}")
        print("-" * 30)
        
        monitor = RuntimeFLOPsMonitor(model, method=method)
        monitor.start_monitoring()
        
        # Run a few training steps
        for step in range(3):
            result = monitor.measure_step(inputs, targets, optimizer)
            print(f"Step {step + 1}: {monitor.format_flops(result['step_flops'])}")
        
        # Get final statistics
        stats = monitor.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total FLOPs: {monitor.format_flops(stats['total_flops'])}")
        print(f"  Avg per step: {monitor.format_flops(stats['avg_flops_per_step'])}")
        print(f"  FLOPs/second: {monitor.format_flops(stats['flops_per_second'])}/s")
        
        monitor.cleanup()
        print()


if __name__ == "__main__":
    demo_flops_monitoring()