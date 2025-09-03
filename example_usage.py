#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实际使用示例：计算神经网络预训练的 FLOPs
"""

from flops_utils import quick_calculate_epoch_flops, format_flops
import torch
import torch.nn as nn
import math


def example_pretrain_flops_calculation():
    """
    示例：预训练场景的 FLOPs 计算
    """
    print("实际预训练场景 FLOPs 计算示例")
    print("=" * 60)
    
    # 创建一个类似 GPT 的简单模型
    class SimpleGPT(nn.Module):
        def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=12,
                    dim_feedforward=3072,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            self.ln_f = nn.LayerNorm(hidden_size)
            self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                # 注意：这里简化了 causal mask 的处理
                x = layer(x, x)
            x = self.ln_f(x)
            return self.head(x)
    
    print("创建 GPT 类型模型 (hidden_size=768, num_layers=12)...")
    model = SimpleGPT()
    
    # 预训练配置
    pretrain_config = {
        'total_samples': 10_000_000,  # 1000万训练样本
        'batch_size': 32,             # 批次大小
        'seq_size': 1024,            # 序列长度
        'base_batch_size': 1,        # 基准测量批次大小
        'base_seq_size': 128         # 基准测量序列长度
    }
    
    print(f"预训练配置: {pretrain_config}")
    print()
    
    # 计算 FLOPs
    results = quick_calculate_epoch_flops(model=model, **pretrain_config)
    
    # 额外分析
    print("\n" + "=" * 60)
    print("训练资源分析")
    print("=" * 60)
    
    unit_flops = results['unit_flops_per_token']
    
    # 计算不同 epoch 数的总计算量
    epoch_counts = [1, 3, 10, 100]
    print("不同 epoch 数的总训练 FLOPs:")
    for epochs in epoch_counts:
        total_flops = results['epoch_flops']['epoch_total_flops'] * epochs
        print(f"  {epochs:3d} epochs: {format_flops(total_flops)}")
    print()
    
    # 计算不同 seq_size 的影响
    seq_sizes = [256, 512, 1024, 2048]
    print("不同序列长度的 epoch FLOPs:")
    for seq_len in seq_sizes:
        epoch_flops = unit_flops * 3 * pretrain_config['total_samples'] * seq_len
        print(f"  seq_size={seq_len:4d}: {format_flops(epoch_flops)}")
    print()
    
    # 计算不同 batch_size 的单次训练 FLOPs
    batch_sizes = [8, 16, 32, 64, 128]
    print("不同批次大小的单次训练 FLOPs:")
    for bs in batch_sizes:
        training_flops = unit_flops * 3 * bs * pretrain_config['seq_size']
        print(f"  batch_size={bs:3d}: {format_flops(training_flops)}")
    
    return results


def example_model_comparison():
    """
    示例：不同模型大小的 FLOPs 对比
    """
    print("\n" + "=" * 60)
    print("不同模型大小的 FLOPs 对比")
    print("=" * 60)
    
    # 定义不同大小的模型配置
    model_configs = {
        'Small': {'hidden_size': 512, 'num_layers': 6, 'vocab_size': 30000},
        'Base': {'hidden_size': 768, 'num_layers': 12, 'vocab_size': 30000},
        'Large': {'hidden_size': 1024, 'num_layers': 24, 'vocab_size': 30000}
    }
    
    # 训练配置
    train_config = {
        'total_samples': 1_000_000,
        'batch_size': 32,
        'seq_size': 512
    }
    
    print(f"训练配置: {train_config}")
    print()
    
    # 理论计算不同模型的 FLOPs
    for model_name, config in model_configs.items():
        H, L, V = config['hidden_size'], config['num_layers'], config['vocab_size']
        F = H * 4  # FFN 维度通常是 hidden_size 的 4 倍
        
        B, S = train_config['batch_size'], train_config['seq_size']
        
        # 理论前向传播 FLOPs
        forward_flops = B * S * (2 * H * V + L * (4 * H * H + 2 * S * H + 2 * H * F))
        training_flops = forward_flops * 3  # 包含反向传播
        
        # 每个 epoch FLOPs
        num_batches = math.ceil(train_config['total_samples'] / B)
        epoch_flops = training_flops * num_batches
        
        # 估算参数量
        params = V * H + L * (4 * H * H + 2 * H * F) + H * V
        
        print(f"{model_name} 模型 (H={H}, L={L}):")
        print(f"  参数量: {format_flops(params).replace('FLOPS', 'Params')}")
        print(f"  单次训练: {format_flops(training_flops)}")
        print(f"  epoch 总计: {format_flops(epoch_flops)}")
        print()


def print_practical_guidelines():
    """打印实际使用指南"""
    print("=" * 60)
    print("实际使用指南")
    print("=" * 60)
    
    print("1. 快速估算步骤:")
    print("   a) 用小配置测量基准 FLOPs: calculate_model_flops(model, 1, 128)")
    print("   b) 计算单位 FLOPs: unit_flops = forward_flops / (1 * 128)")
    print("   c) 缩放到实际配置: actual_flops = unit_flops * 3 * batch_size * seq_size")
    print("   d) 计算 epoch 总量: epoch_flops = actual_flops * (total_samples / batch_size)")
    print()
    
    print("2. 关键公式:")
    print("   前向传播 FLOPs = unit_flops_per_token * batch_size * seq_size")
    print("   反向传播 FLOPs = 2 * 前向传播 FLOPs")
    print("   训练总 FLOPs = 3 * 前向传播 FLOPs")
    print("   epoch FLOPs = 3 * unit_flops_per_token * total_samples * seq_size")
    print()
    
    print("3. 注意事项:")
    print("   - unit_flops_per_token 是模型特定的常数")
    print("   - FLOPs 与 batch_size 和 seq_size 基本成线性关系")
    print("   - 反向传播倍数通常是 2.0，但可以调整")
    print("   - 使用激活重计算时，倍数可能是 3.0")
    print()
    
    print("4. 实际应用:")
    print("   - 训练资源规划：根据 FLOPs 估算训练时间")
    print("   - 模型对比：比较不同架构的计算效率")
    print("   - 超参数调优：平衡 batch_size 和 seq_size")


if __name__ == "__main__":
    # 运行示例
    results = example_pretrain_flops_calculation()
    
    # 模型对比
    example_model_comparison()
    
    # 使用指南
    print_practical_guidelines()