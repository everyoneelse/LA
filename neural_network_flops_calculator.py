#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经网络预训练 FLOPs 计算工具
基于 calflops 库计算每个 epoch 的前向传播和反向传播计算量
支持与 batch_size 和 seq_size 相关的表达式计算
"""

import torch
import torch.nn as nn
from calflops import calculate_flops
from typing import Union, Tuple, Optional, Dict, Any
import math


class NeuralNetworkFLOPsCalculator:
    """
    神经网络 FLOPs 计算器
    用于计算预训练时每个 epoch 的前向和反向传播计算量
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化计算器
        
        Args:
            model: PyTorch 模型实例
        """
        self.model = model
        self.base_flops = None
        self.base_macs = None
        self.base_params = None
        self.base_batch_size = None
        self.base_seq_size = None
    
    def calculate_base_flops(self, 
                           batch_size: int = 1, 
                           seq_size: int = 128,
                           transformer_tokenizer=None,
                           include_backward: bool = True,
                           compute_bp_factor: float = 2.0) -> Dict[str, Any]:
        """
        计算基准 FLOPs（单次前向传播）
        
        Args:
            batch_size: 批次大小
            seq_size: 序列长度
            transformer_tokenizer: Transformer 模型的 tokenizer
            include_backward: 是否包含反向传播
            compute_bp_factor: 反向传播计算倍数（默认2.0）
            
        Returns:
            包含 FLOPs、MACs、Params 的字典
        """
        input_shape = (batch_size, seq_size)
        
        # 本地模型计算
        flops, macs, params = calculate_flops(
            model=self.model,
            input_shape=input_shape,
            transformer_tokenizer=transformer_tokenizer,
            include_backPropagation=include_backward,
            compute_bp_factor=compute_bp_factor,
            print_results=False,
            output_as_string=False
        )
        
        # 保存基准值
        self.base_flops = flops
        self.base_macs = macs
        self.base_params = params
        self.base_batch_size = batch_size
        self.base_seq_size = seq_size
        
        return {
            'forward_flops': flops / (1 + compute_bp_factor) if include_backward else flops,
            'backward_flops': flops * compute_bp_factor / (1 + compute_bp_factor) if include_backward else flops * compute_bp_factor,
            'total_flops': flops,
            'macs': macs,
            'params': params,
            'batch_size': batch_size,
            'seq_size': seq_size
        }
    
    def calculate_epoch_flops(self, 
                            total_samples: int,
                            batch_size: int,
                            seq_size: int,
                            compute_bp_factor: float = 2.0) -> Dict[str, Any]:
        """
        计算每个 epoch 的总 FLOPs
        
        Args:
            total_samples: 训练数据总样本数
            batch_size: 批次大小
            seq_size: 序列长度
            compute_bp_factor: 反向传播计算倍数
            
        Returns:
            包含每个 epoch 详细计算量的字典
        """
        if self.base_flops is None:
            raise ValueError("请先调用 calculate_base_flops 方法计算基准 FLOPs")
        
        # 计算批次数量
        num_batches = math.ceil(total_samples / batch_size)
        
        # 根据 batch_size 和 seq_size 的变化调整 FLOPs
        batch_scale_factor = batch_size / self.base_batch_size
        seq_scale_factor = seq_size / self.base_seq_size
        scale_factor = batch_scale_factor * seq_scale_factor
        
        # 单次前向传播 FLOPs
        single_forward_flops = self.base_flops / (1 + compute_bp_factor) * scale_factor
        
        # 单次反向传播 FLOPs
        single_backward_flops = single_forward_flops * compute_bp_factor
        
        # 每个 epoch 的总计算量
        epoch_forward_flops = single_forward_flops * num_batches
        epoch_backward_flops = single_backward_flops * num_batches
        epoch_total_flops = epoch_forward_flops + epoch_backward_flops
        
        return {
            'total_samples': total_samples,
            'batch_size': batch_size,
            'seq_size': seq_size,
            'num_batches': num_batches,
            'single_forward_flops': single_forward_flops,
            'single_backward_flops': single_backward_flops,
            'single_total_flops': single_forward_flops + single_backward_flops,
            'epoch_forward_flops': epoch_forward_flops,
            'epoch_backward_flops': epoch_backward_flops,
            'epoch_total_flops': epoch_total_flops,
            'scale_factor': scale_factor,
            'compute_bp_factor': compute_bp_factor
        }
    
    def get_flops_expression(self, compute_bp_factor: float = 2.0) -> Dict[str, str]:
        """
        获取与 batch_size 和 seq_size 相关的 FLOPs 表达式
        
        Args:
            compute_bp_factor: 反向传播计算倍数
            
        Returns:
            包含各种 FLOPs 表达式的字典
        """
        if self.base_flops is None:
            raise ValueError("请先调用 calculate_base_flops 方法计算基准 FLOPs")
        
        # 基准单次前向传播 FLOPs
        base_forward_flops = self.base_flops / (1 + compute_bp_factor)
        
        # 计算单位 FLOPs（每个样本每个 token 的 FLOPs）
        unit_flops_per_token = base_forward_flops / (self.base_batch_size * self.base_seq_size)
        
        expressions = {
            'unit_flops_per_token': f"{unit_flops_per_token:.2e}",
            'single_forward_flops': f"{unit_flops_per_token:.2e} * batch_size * seq_size",
            'single_backward_flops': f"{unit_flops_per_token * compute_bp_factor:.2e} * batch_size * seq_size",
            'single_total_flops': f"{unit_flops_per_token * (1 + compute_bp_factor):.2e} * batch_size * seq_size",
            'epoch_forward_flops': f"{unit_flops_per_token:.2e} * total_samples * seq_size",
            'epoch_backward_flops': f"{unit_flops_per_token * compute_bp_factor:.2e} * total_samples * seq_size",
            'epoch_total_flops': f"{unit_flops_per_token * (1 + compute_bp_factor):.2e} * total_samples * seq_size"
        }
        
        return expressions
    
    def format_flops(self, flops: float) -> str:
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
    
    def print_detailed_results(self, results: Dict[str, Any]):
        """打印详细的计算结果"""
        print("=" * 80)
        print("神经网络预训练 FLOPs 计算结果")
        print("=" * 80)
        
        print(f"模型参数: {self.format_flops(self.base_params).replace('FLOPS', 'Params')}")
        print(f"训练样本总数: {results['total_samples']:,}")
        print(f"批次大小: {results['batch_size']}")
        print(f"序列长度: {results['seq_size']}")
        print(f"每个 epoch 批次数: {results['num_batches']}")
        print(f"反向传播倍数: {results['compute_bp_factor']}")
        print()
        
        print("单次批次计算量:")
        print(f"  前向传播: {self.format_flops(results['single_forward_flops'])}")
        print(f"  反向传播: {self.format_flops(results['single_backward_flops'])}")
        print(f"  总计: {self.format_flops(results['single_total_flops'])}")
        print()
        
        print("每个 epoch 总计算量:")
        print(f"  前向传播: {self.format_flops(results['epoch_forward_flops'])}")
        print(f"  反向传播: {self.format_flops(results['epoch_backward_flops'])}")
        print(f"  总计: {self.format_flops(results['epoch_total_flops'])}")
        print()
        
        print(f"缩放因子 (相对于基准 batch_size={self.base_batch_size}, seq_size={self.base_seq_size}): {results['scale_factor']:.2f}")


def demo_simple_transformer():
    """演示使用简单 Transformer 模型计算 FLOPs"""
    print("演示：计算简单 Transformer 模型的 FLOPs")
    print("-" * 50)
    
    # 创建一个简单的 Transformer 模型
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=10000, hidden_size=768, num_layers=6, num_heads=12):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=3072, batch_first=True)
                for _ in range(num_layers)
            ])
            self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.transformer_layers:
                x = layer(x)
            return self.output_layer(x)
    
    model = SimpleTransformer()
    calculator = NeuralNetworkFLOPsCalculator(model=model)
    
    # 计算基准 FLOPs（batch_size=1, seq_size=128）
    print("1. 计算基准 FLOPs...")
    base_results = calculator.calculate_base_flops(
        batch_size=1, 
        seq_size=128,
        include_backward=True,
        compute_bp_factor=2.0
    )
    
    print(f"基准前向传播 FLOPs: {calculator.format_flops(base_results['forward_flops'])}")
    print(f"基准反向传播 FLOPs: {calculator.format_flops(base_results['backward_flops'])}")
    print(f"基准总 FLOPs: {calculator.format_flops(base_results['total_flops'])}")
    print()
    
    # 获取 FLOPs 表达式
    print("2. FLOPs 计算表达式:")
    expressions = calculator.get_flops_expression()
    print(f"每个 token 单位 FLOPs: {expressions['unit_flops_per_token']}")
    print(f"单次前向传播: {expressions['single_forward_flops']}")
    print(f"单次反向传播: {expressions['single_backward_flops']}")
    print(f"每个 epoch 总计算量: {expressions['epoch_total_flops']}")
    print()
    
    # 计算具体 epoch 的 FLOPs
    print("3. 计算具体训练场景的 FLOPs:")
    epoch_results = calculator.calculate_epoch_flops(
        total_samples=100000,  # 10万个训练样本
        batch_size=32,         # 批次大小32
        seq_size=512          # 序列长度512
    )
    
    calculator.print_detailed_results(epoch_results)
    
    return calculator, expressions, epoch_results


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


def print_general_formulas():
    """打印通用的 FLOPs 计算公式"""
    print("=" * 80)
    print("神经网络 FLOPs 通用计算公式")
    print("=" * 80)
    print("符号说明:")
    print("  B = batch_size (批次大小)")
    print("  S = seq_size (序列长度)")
    print("  H = hidden_size (隐藏层维度)")
    print("  F = ffn_size (FFN 中间层维度)")
    print("  NH = num_heads (注意力头数)")
    print("  L = num_layers (层数)")
    print("  V = vocab_size (词汇表大小)")
    print()
    
    print("1. Transformer 模型单层 FLOPs:")
    print("   注意力机制: B*S*(4*H² + 2*S*H)")
    print("   前馈网络:   2*B*S*H*F")
    print("   单层总计:   B*S*(4*H² + 2*S*H + 2*H*F)")
    print()
    
    print("2. 完整 Transformer 模型 FLOPs:")
    print("   嵌入层:     B*S*H*V")
    print("   L层编码器:  L*B*S*(4*H² + 2*S*H + 2*H*F)")
    print("   输出层:     B*S*H*V")
    print("   总计:       B*S*(2*H*V + L*(4*H² + 2*S*H + 2*H*F))")
    print()
    
    print("3. 训练时的 FLOPs (包含反向传播):")
    print("   前向传播:   上述公式")
    print("   反向传播:   2 * 前向传播 FLOPs")
    print("   训练总计:   3 * 前向传播 FLOPs")
    print()
    
    print("4. 每个 epoch 的 FLOPs:")
    print("   epoch_flops = 单次训练 FLOPs * (total_samples / batch_size)")
    print("   = 3 * 前向传播 FLOPs * (total_samples / batch_size)")
    print()
    
    print("5. 关键关系:")
    print("   - FLOPs 与 batch_size 成正比")
    print("   - FLOPs 与 seq_size 成正比（线性层）或平方关系（注意力）")
    print("   - 实际使用中通常近似为线性关系")


if __name__ == "__main__":
    print("神经网络 FLOPs 计算工具演示")
    print("=" * 80)
    
    # 创建一个简单的 Transformer 模型
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size=10000, hidden_size=768, num_layers=6, num_heads=12):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=3072, batch_first=True)
                for _ in range(num_layers)
            ])
            self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.transformer_layers:
                x = layer(x)
            return self.output_layer(x)
    
    print("创建简单 Transformer 模型...")
    model = SimpleTransformer()
    calculator = NeuralNetworkFLOPsCalculator(model=model)
    
    # 计算基准 FLOPs
    print("计算基准 FLOPs (batch_size=1, seq_size=128)...")
    base_results = calculator.calculate_base_flops(
        batch_size=1, 
        seq_size=128,
        include_backward=True,
        compute_bp_factor=2.0
    )
    
    print(f"前向传播 FLOPs: {calculator.format_flops(base_results['forward_flops'])}")
    print(f"反向传播 FLOPs: {calculator.format_flops(base_results['backward_flops'])}")
    print(f"总 FLOPs: {calculator.format_flops(base_results['total_flops'])}")
    print()
    
    # 获取表达式
    print("FLOPs 计算表达式:")
    expressions = calculator.get_flops_expression()
    for key, value in expressions.items():
        print(f"  {key}: {value}")
    print()
    
    # 计算 epoch FLOPs
    print("计算 epoch FLOPs (100k samples, batch_size=32, seq_size=512)...")
    epoch_results = calculator.calculate_epoch_flops(
        total_samples=100000,
        batch_size=32,
        seq_size=512
    )
    
    calculator.print_detailed_results(epoch_results)
    
    # 显示通用公式
    print_general_formulas()
    
    print("\n使用说明:")
    print("1. 创建计算器: calculator = NeuralNetworkFLOPsCalculator(model)")
    print("2. 计算基准: base_results = calculator.calculate_base_flops()")
    print("3. 计算 epoch: epoch_results = calculator.calculate_epoch_flops(total_samples, batch_size, seq_size)")
    print("4. 获取表达式: expressions = calculator.get_flops_expression()")