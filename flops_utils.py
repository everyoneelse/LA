#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经网络 FLOPs 计算实用工具函数
简化版本，便于直接使用
"""

import torch
import torch.nn as nn
from calflops import calculate_flops
import math
from typing import Dict, Tuple, Optional, Any


def calculate_model_flops(model: nn.Module, 
                         batch_size: int = 1, 
                         seq_size: int = 128,
                         compute_bp_factor: float = 2.0) -> Dict[str, float]:
    """
    计算模型的 FLOPs
    
    Args:
        model: PyTorch 模型
        batch_size: 批次大小
        seq_size: 序列长度（对于 NLP 模型）或其他维度
        compute_bp_factor: 反向传播计算倍数
        
    Returns:
        包含前向、反向、总 FLOPs 的字典
    """
    # 尝试不同的输入格式
    try:
        # 对于 Transformer 模型，输入通常是 token ids
        input_ids = torch.randint(0, 1000, (batch_size, seq_size))
        
        forward_flops, _, params = calculate_flops(
            model=model,
            args=[input_ids],
            include_backPropagation=False,
            print_results=False,
            output_as_string=False
        )
    except:
        try:
            # 对于其他模型，使用浮点输入
            input_shape = (batch_size, seq_size, 768)  # 假设 hidden_size=768
            forward_flops, _, params = calculate_flops(
                model=model,
                input_shape=input_shape,
                include_backPropagation=False,
                print_results=False,
                output_as_string=False
            )
        except:
            # 最后尝试简单的输入形状
            input_shape = (batch_size, seq_size)
            forward_flops, _, params = calculate_flops(
                model=model,
                input_shape=input_shape,
                include_backPropagation=False,
                print_results=False,
                output_as_string=False
            )
    
    backward_flops = forward_flops * compute_bp_factor
    total_flops = forward_flops + backward_flops
    
    return {
        'forward_flops': forward_flops,
        'backward_flops': backward_flops,
        'total_flops': total_flops,
        'params': params,
        'batch_size': batch_size,
        'seq_size': seq_size
    }


def get_flops_per_token(forward_flops: float, batch_size: int, seq_size: int) -> float:
    """
    计算每个 token 的 FLOPs
    
    Args:
        forward_flops: 前向传播 FLOPs
        batch_size: 批次大小
        seq_size: 序列长度
        
    Returns:
        每个 token 的 FLOPs
    """
    return forward_flops / (batch_size * seq_size)


def calculate_epoch_flops(unit_flops_per_token: float,
                         total_samples: int,
                         seq_size: int,
                         compute_bp_factor: float = 2.0) -> Dict[str, float]:
    """
    计算每个 epoch 的 FLOPs
    
    Args:
        unit_flops_per_token: 每个 token 的前向传播 FLOPs
        total_samples: 训练样本总数
        seq_size: 序列长度
        compute_bp_factor: 反向传播计算倍数
        
    Returns:
        epoch FLOPs 计算结果
    """
    epoch_forward_flops = unit_flops_per_token * total_samples * seq_size
    epoch_backward_flops = epoch_forward_flops * compute_bp_factor
    epoch_total_flops = epoch_forward_flops + epoch_backward_flops
    
    return {
        'epoch_forward_flops': epoch_forward_flops,
        'epoch_backward_flops': epoch_backward_flops,
        'epoch_total_flops': epoch_total_flops,
        'total_samples': total_samples,
        'seq_size': seq_size
    }


def format_flops(flops: float) -> str:
    """格式化 FLOPs 显示"""
    if flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPS"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPS"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f} KFLOPS"
    else:
        return f"{flops:.2f} FLOPS"


def print_flops_summary(model_flops: Dict, epoch_flops: Dict, unit_flops: float):
    """打印 FLOPs 计算总结"""
    print("=" * 60)
    print("FLOPs 计算总结")
    print("=" * 60)
    
    print(f"模型参数量: {format_flops(model_flops['params']).replace('FLOPS', 'Params')}")
    print(f"基准配置: batch_size={model_flops['batch_size']}, seq_size={model_flops['seq_size']}")
    print()
    
    print("单次批次 FLOPs:")
    print(f"  前向传播: {format_flops(model_flops['forward_flops'])}")
    print(f"  反向传播: {format_flops(model_flops['backward_flops'])}")
    print(f"  总计: {format_flops(model_flops['total_flops'])}")
    print()
    
    print("每个 epoch FLOPs:")
    print(f"  前向传播: {format_flops(epoch_flops['epoch_forward_flops'])}")
    print(f"  反向传播: {format_flops(epoch_flops['epoch_backward_flops'])}")
    print(f"  总计: {format_flops(epoch_flops['epoch_total_flops'])}")
    print()
    
    print("FLOPs 表达式:")
    print(f"  单位 FLOPs/token: {unit_flops:.2e}")
    print(f"  前向传播 = {unit_flops:.2e} * batch_size * seq_size")
    print(f"  训练总计 = {unit_flops * 3:.2e} * batch_size * seq_size")
    print(f"  epoch 总计 = {unit_flops * 3:.2e} * total_samples * seq_size")


# 快速使用函数
def quick_calculate_epoch_flops(model: nn.Module,
                               total_samples: int,
                               batch_size: int = 32,
                               seq_size: int = 512,
                               base_batch_size: int = 1,
                               base_seq_size: int = 128) -> Dict[str, Any]:
    """
    快速计算 epoch FLOPs
    
    Args:
        model: PyTorch 模型
        total_samples: 训练样本总数
        batch_size: 实际训练批次大小
        seq_size: 实际序列长度
        base_batch_size: 基准测量批次大小
        base_seq_size: 基准测量序列长度
        
    Returns:
        完整的 FLOPs 计算结果
    """
    # 1. 基准测量
    print("正在进行基准测量...")
    model_flops = calculate_model_flops(model, base_batch_size, base_seq_size)
    
    # 2. 计算单位 FLOPs
    unit_flops = get_flops_per_token(
        model_flops['forward_flops'], 
        base_batch_size, 
        base_seq_size
    )
    
    # 3. 计算 epoch FLOPs
    epoch_flops = calculate_epoch_flops(unit_flops, total_samples, seq_size)
    
    # 4. 打印结果
    print_flops_summary(model_flops, epoch_flops, unit_flops)
    
    return {
        'model_flops': model_flops,
        'epoch_flops': epoch_flops,
        'unit_flops_per_token': unit_flops,
        'expressions': {
            'forward': f"{unit_flops:.2e} * batch_size * seq_size",
            'training': f"{unit_flops * 3:.2e} * batch_size * seq_size",
            'epoch': f"{unit_flops * 3:.2e} * total_samples * seq_size"
        }
    }


if __name__ == "__main__":
    print("FLOPs 计算实用工具演示")
    print("=" * 60)
    
    # 创建一个简单模型进行演示
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(768, 3072)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(3072, 768)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    
    # 快速计算示例
    results = quick_calculate_epoch_flops(
        model=model,
        total_samples=1000000,  # 100万样本
        batch_size=32,
        seq_size=512,
        base_batch_size=1,
        base_seq_size=128
    )
    
    print("\n表达式使用示例:")
    print("=" * 60)
    print("# 根据上面的计算结果，你可以使用以下表达式:")
    for name, expr in results['expressions'].items():
        print(f"{name}_flops = {expr}")
    
    print(f"\n# 例如，计算不同配置下的 FLOPs:")
    print(f"# batch_size=64, seq_size=1024 的训练 FLOPs:")
    example_flops = results['unit_flops_per_token'] * 3 * 64 * 1024
    print(f"# = {results['unit_flops_per_token']:.2e} * 3 * 64 * 1024 = {format_flops(example_flops)}")