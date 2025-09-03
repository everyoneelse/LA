#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经网络 FLOPs 计算演示
"""

import torch
import torch.nn as nn
from calflops import calculate_flops
import math


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


def demo_simple_linear_model():
    """演示简单线性模型的 FLOPs 计算"""
    print("演示：简单线性模型 FLOPs 计算")
    print("-" * 50)
    
    # 创建一个简单的线性模型
    class SimpleLinearModel(nn.Module):
        def __init__(self, input_size=768, hidden_size=3072, output_size=768):
            super().__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = SimpleLinearModel()
    
    # 计算不同输入大小的 FLOPs
    batch_sizes = [1, 8, 16, 32]
    seq_sizes = [128, 256, 512, 1024]
    
    print("计算不同 batch_size 和 seq_size 下的 FLOPs:")
    print()
    
    base_results = {}
    
    for batch_size in batch_sizes:
        for seq_size in seq_sizes:
            input_shape = (batch_size, seq_size, 768)  # (batch, seq, hidden)
            
            # 计算前向传播 FLOPs
            forward_flops, forward_macs, params = calculate_flops(
                model=model,
                input_shape=input_shape,
                include_backPropagation=False,
                print_results=False,
                output_as_string=False
            )
            
            # 计算包含反向传播的 FLOPs
            total_flops, total_macs, _ = calculate_flops(
                model=model,
                input_shape=input_shape,
                include_backPropagation=True,
                compute_bp_factor=2.0,
                print_results=False,
                output_as_string=False
            )
            
            backward_flops = total_flops - forward_flops
            
            print(f"batch_size={batch_size:2d}, seq_size={seq_size:4d}: "
                  f"前向={format_flops(forward_flops):>12s}, "
                  f"反向={format_flops(backward_flops):>12s}, "
                  f"总计={format_flops(total_flops):>12s}")
            
            # 保存基准结果用于分析
            if batch_size == 1 and seq_size == 128:
                base_results = {
                    'forward_flops': forward_flops,
                    'backward_flops': backward_flops,
                    'total_flops': total_flops,
                    'params': params
                }
    
    print()
    print("=" * 80)
    print("FLOPs 与 batch_size、seq_size 的关系分析")
    print("=" * 80)
    
    if base_results:
        base_forward = base_results['forward_flops']
        base_backward = base_results['backward_flops']
        
        # 计算每个 token 的单位 FLOPs
        unit_forward_flops = base_forward / (1 * 128)  # 基准是 batch=1, seq=128
        unit_backward_flops = base_backward / (1 * 128)
        unit_total_flops = unit_forward_flops + unit_backward_flops
        
        print(f"模型参数量: {format_flops(base_results['params']).replace('FLOPS', 'Params')}")
        print(f"基准配置 (batch_size=1, seq_size=128):")
        print(f"  前向传播: {format_flops(base_forward)}")
        print(f"  反向传播: {format_flops(base_backward)}")
        print()
        
        print("单位 FLOPs (每个 token):")
        print(f"  前向传播: {unit_forward_flops:.2e} FLOPs/token")
        print(f"  反向传播: {unit_backward_flops:.2e} FLOPs/token")
        print(f"  总计: {unit_total_flops:.2e} FLOPs/token")
        print()
        
        print("FLOPs 计算公式:")
        print(f"  前向传播 = {unit_forward_flops:.2e} * batch_size * seq_size")
        print(f"  反向传播 = {unit_backward_flops:.2e} * batch_size * seq_size")
        print(f"  总计 = {unit_total_flops:.2e} * batch_size * seq_size")
        print()
        
        print("每个 epoch 的 FLOPs 公式:")
        print(f"  epoch_forward_flops = {unit_forward_flops:.2e} * total_samples * seq_size")
        print(f"  epoch_backward_flops = {unit_backward_flops:.2e} * total_samples * seq_size")
        print(f"  epoch_total_flops = {unit_total_flops:.2e} * total_samples * seq_size")
        print()
        
        # 示例计算
        print("示例计算 (total_samples=1,000,000, batch_size=32, seq_size=512):")
        total_samples = 1000000
        batch_size = 32
        seq_size = 512
        num_batches = math.ceil(total_samples / batch_size)
        
        epoch_forward = unit_forward_flops * total_samples * seq_size
        epoch_backward = unit_backward_flops * total_samples * seq_size
        epoch_total = epoch_forward + epoch_backward
        
        print(f"  批次数量: {num_batches:,}")
        print(f"  epoch 前向传播: {format_flops(epoch_forward)}")
        print(f"  epoch 反向传播: {format_flops(epoch_backward)}")
        print(f"  epoch 总计: {format_flops(epoch_total)}")


def demo_attention_flops():
    """演示注意力机制的 FLOPs 计算"""
    print("\n" + "=" * 80)
    print("演示：注意力机制 FLOPs 理论计算")
    print("-" * 50)
    
    def calculate_attention_flops(batch_size, seq_size, hidden_size, num_heads):
        """计算注意力机制的理论 FLOPs"""
        head_dim = hidden_size // num_heads
        
        # Q, K, V 线性变换: 3 * B * S * H * H
        qkv_flops = 3 * batch_size * seq_size * hidden_size * hidden_size
        
        # 注意力分数计算: B * NH * S * S * head_dim
        attention_scores_flops = batch_size * num_heads * seq_size * seq_size * head_dim
        
        # 注意力权重与 V 相乘: B * NH * S * S * head_dim
        attention_output_flops = batch_size * num_heads * seq_size * seq_size * head_dim
        
        # 输出线性变换: B * S * H * H
        output_linear_flops = batch_size * seq_size * hidden_size * hidden_size
        
        total_flops = qkv_flops + attention_scores_flops + attention_output_flops + output_linear_flops
        
        return {
            'qkv_linear': qkv_flops,
            'attention_scores': attention_scores_flops,
            'attention_output': attention_output_flops,
            'output_linear': output_linear_flops,
            'total': total_flops
        }
    
    # 计算不同配置下的注意力 FLOPs
    configs = [
        {'batch_size': 1, 'seq_size': 128, 'hidden_size': 768, 'num_heads': 12},
        {'batch_size': 32, 'seq_size': 512, 'hidden_size': 768, 'num_heads': 12},
        {'batch_size': 16, 'seq_size': 1024, 'hidden_size': 1024, 'num_heads': 16},
    ]
    
    print("不同配置下的注意力机制 FLOPs:")
    print()
    
    for i, config in enumerate(configs, 1):
        result = calculate_attention_flops(**config)
        print(f"配置 {i}: {config}")
        print(f"  QKV 线性变换: {format_flops(result['qkv_linear'])}")
        print(f"  注意力分数计算: {format_flops(result['attention_scores'])}")
        print(f"  注意力输出计算: {format_flops(result['attention_output'])}")
        print(f"  输出线性变换: {format_flops(result['output_linear'])}")
        print(f"  总计: {format_flops(result['total'])}")
        print()
    
    print("注意力机制 FLOPs 公式:")
    print("  QKV 变换: 3 * B * S * H²")
    print("  注意力计算: 2 * B * NH * S² * (H/NH) = 2 * B * S² * H")
    print("  输出变换: B * S * H²")
    print("  总计: B * S * (4 * H² + 2 * S * H)")
    print()
    print("可以看出:")
    print("  - 线性层部分与 seq_size 成正比")
    print("  - 注意力计算部分与 seq_size 的平方成正比")
    print("  - 当 seq_size 较大时，注意力计算占主导")


if __name__ == "__main__":
    print("神经网络 FLOPs 计算工具演示")
    print("=" * 80)
    
    # 演示简单线性模型
    demo_simple_linear_model()
    
    # 演示注意力机制理论计算
    demo_attention_flops()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("1. 对于神经网络预训练，FLOPs 计算公式为:")
    print("   - 前向传播: unit_flops_per_token * batch_size * seq_size")
    print("   - 反向传播: 2 * unit_flops_per_token * batch_size * seq_size")
    print("   - 每个 epoch: unit_flops_per_token * 3 * total_samples * seq_size")
    print()
    print("2. 使用 calflops 库的步骤:")
    print("   - 用基准配置计算单位 FLOPs")
    print("   - 根据实际 batch_size 和 seq_size 进行缩放")
    print("   - 乘以样本数量得到 epoch 总计算量")
    print()
    print("3. 关键参数:")
    print("   - compute_bp_factor: 反向传播计算倍数（默认2.0）")
    print("   - include_backPropagation: 是否包含反向传播")
    print("   - FLOPs 与 batch_size、seq_size 基本成线性关系")