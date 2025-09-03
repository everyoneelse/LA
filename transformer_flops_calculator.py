#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer 模型预训练 FLOPs 计算工具
基于 calflops 库和理论公式计算每个 epoch 的计算量
"""

import torch
import torch.nn as nn
from calflops import calculate_flops
import math
from typing import Dict, Any, Optional


class TransformerFLOPsCalculator:
    """
    Transformer 模型 FLOPs 计算器
    支持实际测量和理论计算两种方式
    """
    
    def __init__(self, model: Optional[nn.Module] = None, model_config: Optional[Dict] = None):
        """
        初始化计算器
        
        Args:
            model: PyTorch Transformer 模型实例
            model_config: 模型配置字典，包含 hidden_size, num_layers, num_heads, vocab_size 等
        """
        self.model = model
        self.model_config = model_config or {}
        self.base_measurements = {}
    
    def measure_flops_with_calflops(self, 
                                  batch_size: int = 1, 
                                  seq_size: int = 128,
                                  compute_bp_factor: float = 2.0) -> Dict[str, float]:
        """
        使用 calflops 库实际测量模型的 FLOPs
        
        Args:
            batch_size: 批次大小
            seq_size: 序列长度
            compute_bp_factor: 反向传播计算倍数
            
        Returns:
            测量结果字典
        """
        if self.model is None:
            raise ValueError("需要提供模型实例才能进行实际测量")
        
        # 创建随机输入（长整型，适用于 embedding）
        input_ids = torch.randint(0, self.model_config.get('vocab_size', 10000), 
                                (batch_size, seq_size))
        
        # 计算前向传播 FLOPs
        forward_flops, forward_macs, params = calculate_flops(
            model=self.model,
            args=[input_ids],
            include_backPropagation=False,
            print_results=False,
            output_as_string=False
        )
        
        # 计算包含反向传播的 FLOPs
        total_flops, total_macs, _ = calculate_flops(
            model=self.model,
            args=[input_ids],
            include_backPropagation=True,
            compute_bp_factor=compute_bp_factor,
            print_results=False,
            output_as_string=False
        )
        
        backward_flops = total_flops - forward_flops
        
        # 保存基准测量结果
        self.base_measurements = {
            'batch_size': batch_size,
            'seq_size': seq_size,
            'forward_flops': forward_flops,
            'backward_flops': backward_flops,
            'total_flops': total_flops,
            'params': params,
            'compute_bp_factor': compute_bp_factor
        }
        
        return self.base_measurements
    
    def calculate_theoretical_flops(self, 
                                  batch_size: int, 
                                  seq_size: int,
                                  include_backward: bool = True,
                                  compute_bp_factor: float = 2.0) -> Dict[str, float]:
        """
        基于理论公式计算 Transformer 模型的 FLOPs
        
        Args:
            batch_size: 批量大小
            seq_size: 序列长度
            include_backward: 是否包含反向传播
            compute_bp_factor: 反向传播计算倍数
            
        Returns:
            理论 FLOPs 计算结果
        """
        if not self.model_config:
            raise ValueError("需要提供模型配置才能进行理论计算")
        
        B, S = batch_size, seq_size
        H = self.model_config.get('hidden_size', 768)
        L = self.model_config.get('num_layers', 12)
        V = self.model_config.get('vocab_size', 30522)
        F = self.model_config.get('ffn_size', H * 4)  # 通常是 hidden_size 的 4 倍
        
        # 1. 嵌入层 FLOPs: B * S * H * V (查表操作，实际 FLOPs 可能为 0)
        embedding_flops = 0  # 嵌入层通常是查表操作，不计算 FLOPs
        
        # 2. 单层 Transformer 编码器 FLOPs
        # 注意力机制: B * S * (4 * H² + 2 * S * H)
        attention_flops_per_layer = B * S * (4 * H * H + 2 * S * H)
        
        # 前馈网络: 2 * B * S * H * F
        ffn_flops_per_layer = 2 * B * S * H * F
        
        # 单层总 FLOPs
        single_layer_flops = attention_flops_per_layer + ffn_flops_per_layer
        
        # 所有层的 FLOPs
        all_layers_flops = single_layer_flops * L
        
        # 3. 输出层 FLOPs: B * S * H * V
        output_flops = B * S * H * V
        
        # 总前向传播 FLOPs
        forward_flops = embedding_flops + all_layers_flops + output_flops
        
        # 反向传播 FLOPs
        backward_flops = forward_flops * compute_bp_factor if include_backward else 0
        
        # 总 FLOPs
        total_flops = forward_flops + backward_flops
        
        return {
            'batch_size': batch_size,
            'seq_size': seq_size,
            'embedding_flops': embedding_flops,
            'attention_flops_per_layer': attention_flops_per_layer,
            'ffn_flops_per_layer': ffn_flops_per_layer,
            'single_layer_flops': single_layer_flops,
            'all_layers_flops': all_layers_flops,
            'output_flops': output_flops,
            'forward_flops': forward_flops,
            'backward_flops': backward_flops,
            'total_flops': total_flops,
            'compute_bp_factor': compute_bp_factor
        }
    
    def get_flops_expressions(self, compute_bp_factor: float = 2.0) -> Dict[str, str]:
        """
        获取与 batch_size 和 seq_size 相关的 FLOPs 表达式
        
        Args:
            compute_bp_factor: 反向传播计算倍数
            
        Returns:
            FLOPs 表达式字典
        """
        if self.base_measurements:
            # 基于实际测量的表达式
            base_forward = self.base_measurements['forward_flops']
            base_batch = self.base_measurements['batch_size']
            base_seq = self.base_measurements['seq_size']
            
            unit_flops_per_token = base_forward / (base_batch * base_seq)
            
            return {
                'method': 'measured',
                'unit_flops_per_token': f"{unit_flops_per_token:.2e}",
                'forward_flops': f"{unit_flops_per_token:.2e} * batch_size * seq_size",
                'backward_flops': f"{unit_flops_per_token * compute_bp_factor:.2e} * batch_size * seq_size",
                'total_flops': f"{unit_flops_per_token * (1 + compute_bp_factor):.2e} * batch_size * seq_size",
                'epoch_total_flops': f"{unit_flops_per_token * (1 + compute_bp_factor):.2e} * total_samples * seq_size"
            }
        
        elif self.model_config:
            # 基于理论公式的表达式
            H = self.model_config.get('hidden_size', 768)
            L = self.model_config.get('num_layers', 12)
            V = self.model_config.get('vocab_size', 30522)
            F = self.model_config.get('ffn_size', H * 4)
            
            return {
                'method': 'theoretical',
                'forward_flops': f"B * S * (2 * H * V + L * (4 * H² + 2 * S * H + 2 * H * F))",
                'backward_flops': f"{compute_bp_factor} * B * S * (2 * H * V + L * (4 * H² + 2 * S * H + 2 * H * F))",
                'total_flops': f"{1 + compute_bp_factor} * B * S * (2 * H * V + L * (4 * H² + 2 * S * H + 2 * H * F))",
                'epoch_total_flops': f"{1 + compute_bp_factor} * total_samples * S * (2 * H * V + L * (4 * H² + 2 * S * H + 2 * H * F))",
                'config': self.model_config
            }
        
        else:
            raise ValueError("需要先进行测量或提供模型配置")
    
    def calculate_epoch_flops(self, 
                            total_samples: int,
                            batch_size: int,
                            seq_size: int,
                            method: str = 'auto') -> Dict[str, Any]:
        """
        计算每个 epoch 的 FLOPs
        
        Args:
            total_samples: 训练样本总数
            batch_size: 批次大小
            seq_size: 序列长度
            method: 计算方法 ('measured', 'theoretical', 'auto')
            
        Returns:
            epoch FLOPs 计算结果
        """
        num_batches = math.ceil(total_samples / batch_size)
        
        if method == 'auto':
            method = 'measured' if self.base_measurements else 'theoretical'
        
        if method == 'measured' and self.base_measurements:
            # 基于实际测量进行缩放
            base_forward = self.base_measurements['forward_flops']
            base_backward = self.base_measurements['backward_flops']
            base_batch = self.base_measurements['batch_size']
            base_seq = self.base_measurements['seq_size']
            
            # 计算缩放因子
            scale_factor = (batch_size / base_batch) * (seq_size / base_seq)
            
            single_forward_flops = base_forward * scale_factor
            single_backward_flops = base_backward * scale_factor
            
        elif method == 'theoretical' and self.model_config:
            # 基于理论公式计算
            theoretical_result = self.calculate_theoretical_flops(
                batch_size=batch_size,
                seq_size=seq_size,
                include_backward=True,
                compute_bp_factor=self.base_measurements.get('compute_bp_factor', 2.0)
            )
            
            single_forward_flops = theoretical_result['forward_flops']
            single_backward_flops = theoretical_result['backward_flops']
            
        else:
            raise ValueError(f"无法使用方法 '{method}' 进行计算，请检查是否已提供必要的数据")
        
        # 计算 epoch 总量
        epoch_forward_flops = single_forward_flops * num_batches
        epoch_backward_flops = single_backward_flops * num_batches
        epoch_total_flops = epoch_forward_flops + epoch_backward_flops
        
        return {
            'method': method,
            'total_samples': total_samples,
            'batch_size': batch_size,
            'seq_size': seq_size,
            'num_batches': num_batches,
            'single_forward_flops': single_forward_flops,
            'single_backward_flops': single_backward_flops,
            'single_total_flops': single_forward_flops + single_backward_flops,
            'epoch_forward_flops': epoch_forward_flops,
            'epoch_backward_flops': epoch_backward_flops,
            'epoch_total_flops': epoch_total_flops
        }
    
    @staticmethod
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
    
    def print_epoch_results(self, results: Dict[str, Any]):
        """打印 epoch 计算结果"""
        print("=" * 80)
        print("Transformer 模型预训练 FLOPs 计算结果")
        print("=" * 80)
        
        if self.base_measurements:
            params = self.base_measurements.get('params', 0)
            print(f"模型参数量: {self.format_flops(params).replace('FLOPS', 'Params')}")
        
        print(f"计算方法: {results['method']}")
        print(f"训练样本总数: {results['total_samples']:,}")
        print(f"批次大小: {results['batch_size']}")
        print(f"序列长度: {results['seq_size']}")
        print(f"每个 epoch 批次数: {results['num_batches']:,}")
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


def demo_bert_like_model():
    """演示 BERT 类型模型的 FLOPs 计算"""
    print("演示：BERT 类型模型 FLOPs 计算")
    print("-" * 50)
    
    # BERT-base 配置
    bert_config = {
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ffn_size': 3072
    }
    
    # 创建简单的 BERT 类型模型
    class SimpleBERT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
            self.encoder_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config['hidden_size'],
                    nhead=config['num_heads'],
                    dim_feedforward=config['ffn_size'],
                    batch_first=True
                )
                for _ in range(config['num_layers'])
            ])
            self.output_layer = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.encoder_layers:
                x = layer(x)
            return self.output_layer(x)
    
    model = SimpleBERT(bert_config)
    calculator = TransformerFLOPsCalculator(model=model, model_config=bert_config)
    
    # 1. 实际测量基准 FLOPs
    print("1. 实际测量基准 FLOPs (batch_size=1, seq_size=128)...")
    measured_results = calculator.measure_flops_with_calflops(
        batch_size=1,
        seq_size=128,
        compute_bp_factor=2.0
    )
    
    print(f"测量结果 - 前向: {calculator.format_flops(measured_results['forward_flops'])}")
    print(f"测量结果 - 反向: {calculator.format_flops(measured_results['backward_flops'])}")
    print(f"测量结果 - 总计: {calculator.format_flops(measured_results['total_flops'])}")
    print()
    
    # 2. 理论计算对比
    print("2. 理论计算对比...")
    theoretical_results = calculator.calculate_theoretical_flops(
        batch_size=1,
        seq_size=128,
        include_backward=True,
        compute_bp_factor=2.0
    )
    
    print(f"理论结果 - 前向: {calculator.format_flops(theoretical_results['forward_flops'])}")
    print(f"理论结果 - 反向: {calculator.format_flops(theoretical_results['backward_flops'])}")
    print(f"理论结果 - 总计: {calculator.format_flops(theoretical_results['total_flops'])}")
    print()
    
    # 3. 获取表达式
    print("3. FLOPs 计算表达式:")
    expressions = calculator.get_flops_expressions()
    for key, value in expressions.items():
        if key != 'config':
            print(f"  {key}: {value}")
    print()
    
    # 4. 计算不同训练场景的 epoch FLOPs
    print("4. 不同训练场景的 epoch FLOPs:")
    scenarios = [
        {'total_samples': 100000, 'batch_size': 16, 'seq_size': 256},
        {'total_samples': 1000000, 'batch_size': 32, 'seq_size': 512},
        {'total_samples': 10000000, 'batch_size': 64, 'seq_size': 1024}
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n场景 {i}: {scenario}")
        epoch_results = calculator.calculate_epoch_flops(**scenario)
        
        print(f"  批次数: {epoch_results['num_batches']:,}")
        print(f"  单次批次: {calculator.format_flops(epoch_results['single_total_flops'])}")
        print(f"  epoch 总计: {calculator.format_flops(epoch_results['epoch_total_flops'])}")
    
    return calculator


def demo_llama_like_model():
    """演示 LLaMA 类型模型的理论 FLOPs 计算"""
    print("\n" + "=" * 80)
    print("演示：LLaMA 类型模型理论 FLOPs 计算")
    print("-" * 50)
    
    # LLaMA-7B 配置
    llama_config = {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'num_layers': 32,
        'num_heads': 32,
        'ffn_size': 11008  # LLaMA 使用 SwiGLU，中间维度更大
    }
    
    calculator = TransformerFLOPsCalculator(model_config=llama_config)
    
    print(f"LLaMA-7B 配置: {llama_config}")
    print()
    
    # 计算不同输入大小的理论 FLOPs
    input_configs = [
        {'batch_size': 1, 'seq_size': 128},
        {'batch_size': 1, 'seq_size': 512},
        {'batch_size': 1, 'seq_size': 2048},
        {'batch_size': 32, 'seq_size': 512}
    ]
    
    print("不同输入配置的理论 FLOPs:")
    for config in input_configs:
        result = calculator.calculate_theoretical_flops(**config)
        print(f"batch_size={config['batch_size']}, seq_size={config['seq_size']}:")
        print(f"  前向传播: {calculator.format_flops(result['forward_flops'])}")
        print(f"  训练总计: {calculator.format_flops(result['total_flops'])}")
        print()
    
    # 获取理论表达式
    print("理论 FLOPs 表达式:")
    expressions = calculator.get_flops_expressions()
    for key, value in expressions.items():
        if key != 'config':
            print(f"  {key}: {value}")
    
    return calculator


def print_usage_examples():
    """打印使用示例"""
    print("\n" + "=" * 80)
    print("使用示例和最佳实践")
    print("=" * 80)
    
    print("1. 基本使用流程:")
    print("""
# 方法一：使用实际模型测量
model = YourTransformerModel()
calculator = TransformerFLOPsCalculator(model=model)
base_results = calculator.measure_flops_with_calflops(batch_size=1, seq_size=128)
expressions = calculator.get_flops_expressions()

# 方法二：使用理论公式
model_config = {
    'vocab_size': 30522,
    'hidden_size': 768,
    'num_layers': 12,
    'num_heads': 12,
    'ffn_size': 3072
}
calculator = TransformerFLOPsCalculator(model_config=model_config)
theoretical_results = calculator.calculate_theoretical_flops(batch_size=1, seq_size=128)
    """)
    
    print("2. 计算每个 epoch 的 FLOPs:")
    print("""
epoch_results = calculator.calculate_epoch_flops(
    total_samples=1000000,  # 100万训练样本
    batch_size=32,          # 批次大小
    seq_size=512           # 序列长度
)
calculator.print_epoch_results(epoch_results)
    """)
    
    print("3. 关键公式总结:")
    print("   对于 Transformer 模型:")
    print("   - 前向传播 FLOPs ≈ B * S * (2*H*V + L*(4*H² + 2*S*H + 2*H*F))")
    print("   - 反向传播 FLOPs ≈ 2 * 前向传播 FLOPs")
    print("   - 每个 epoch FLOPs = 单次训练 FLOPs * (total_samples / batch_size)")
    print()
    print("   其中:")
    print("   - B = batch_size, S = seq_size, H = hidden_size")
    print("   - L = num_layers, V = vocab_size, F = ffn_size")
    print("   - 注意力部分与 S² 相关，线性层与 S 相关")


if __name__ == "__main__":
    print("Transformer 模型预训练 FLOPs 计算工具")
    print("=" * 80)
    
    # 演示 BERT 类型模型
    try:
        bert_calculator = demo_bert_like_model()
    except Exception as e:
        print(f"BERT 演示出错: {e}")
    
    # 演示 LLaMA 类型模型理论计算
    try:
        llama_calculator = demo_llama_like_model()
    except Exception as e:
        print(f"LLaMA 演示出错: {e}")
    
    # 打印使用示例
    print_usage_examples()