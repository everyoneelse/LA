#!/usr/bin/env python3
"""
神经网络预训练 FLOPs 计算器

基于 calculate-flops.pytorch 库，专门用于计算神经网络预训练时：
1. 每个 epoch 的前向传播 FLOPs
2. 每个 epoch 的反向传播 FLOPs  
3. 表达为与 batch_size 和 seq_size 相关的计算公式

使用方法:
python pretrain_flops_calculator.py

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import sys
import os
import math
import json
from typing import Dict, Tuple, Optional, Any

# 添加 calculate-flops.pytorch 到路径
sys.path.append('/workspace/calculate-flops.pytorch')

try:
    from calflops import calculate_flops, calculate_flops_hf
    from calflops.utils import flops_to_string, params_to_string
    CALFLOPS_AVAILABLE = True
except ImportError:
    print("⚠️  calflops 库不可用，将使用估算方法")
    CALFLOPS_AVAILABLE = False


class PretrainFLOPsCalculator:
    """预训练 FLOPs 计算器"""
    
    def __init__(self, model: nn.Module, model_name: str = "Neural Network"):
        """
        初始化计算器
        
        Args:
            model: PyTorch 模型
            model_name: 模型名称
        """
        self.model = model
        self.model_name = model_name
        self.backward_factor = 2.0  # 反向传播通常是前向传播的2倍
        
        # 模型信息
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 缓存计算结果
        self._flops_cache = {}
    
    def calculate_forward_flops(self, 
                              batch_size: int, 
                              seq_size: int,
                              transformer_tokenizer=None) -> float:
        """
        计算前向传播的 FLOPs
        
        Args:
            batch_size: 批量大小
            seq_size: 序列长度
            transformer_tokenizer: Transformer 分词器（可选）
            
        Returns:
            前向传播的 FLOPs 数量
        """
        cache_key = f"forward_{batch_size}_{seq_size}"
        
        if cache_key in self._flops_cache:
            return self._flops_cache[cache_key]
        
        if CALFLOPS_AVAILABLE:
            try:
                # 使用 calflops 库计算
                input_shape = (batch_size, seq_size)
                
                # 检查模型是否包含 embedding 层
                has_embedding = any(isinstance(m, nn.Embedding) for m in self.model.modules())
                
                if has_embedding:
                    # 为包含 embedding 的模型创建整数输入
                    device = next(self.model.parameters()).device
                    vocab_size = 10000  # 默认词汇表大小
                    
                    # 尝试从模型中获取实际的词汇表大小
                    for m in self.model.modules():
                        if isinstance(m, nn.Embedding):
                            vocab_size = m.num_embeddings
                            break
                    
                    input_ids = torch.randint(0, vocab_size, input_shape, device=device, dtype=torch.long)
                    
                    forward_flops, macs, params = calculate_flops(
                        model=self.model,
                        args=[input_ids],
                        include_backPropagation=False,
                        print_results=False,
                        output_as_string=False
                    )
                else:
                    # 对于普通模型，使用 input_shape
                    forward_flops, macs, params = calculate_flops(
                        model=self.model,
                        input_shape=input_shape,
                        transformer_tokenizer=transformer_tokenizer,
                        include_backPropagation=False,
                        print_results=False,
                        output_as_string=False
                    )
                
                self._flops_cache[cache_key] = forward_flops
                return forward_flops
                
            except Exception as e:
                print(f"⚠️  calflops 计算失败: {e}")
                print("🔄 使用估算方法...")
        
        # 使用估算方法
        forward_flops = self._estimate_forward_flops(batch_size, seq_size)
        self._flops_cache[cache_key] = forward_flops
        return forward_flops
    
    def _estimate_forward_flops(self, batch_size: int, seq_size: int) -> float:
        """
        估算前向传播的 FLOPs
        
        基于模型参数数量和输入大小的简单估算
        """
        # 简单估算：每个参数大约进行 2 次浮点运算（乘法+加法）
        # 乘以 batch_size 和 seq_size
        return self.total_params * batch_size * seq_size * 2
    
    def calculate_backward_flops(self, forward_flops: float) -> float:
        """
        计算反向传播的 FLOPs
        
        Args:
            forward_flops: 前向传播的 FLOPs
            
        Returns:
            反向传播的 FLOPs
        """
        return forward_flops * self.backward_factor
    
    def calculate_epoch_flops(self, 
                            batch_size: int,
                            seq_size: int, 
                            num_samples_per_epoch: int,
                            transformer_tokenizer=None) -> Dict[str, Any]:
        """
        计算每个 epoch 的 FLOPs
        
        Args:
            batch_size: 批量大小
            seq_size: 序列长度
            num_samples_per_epoch: 每个 epoch 的样本数量
            transformer_tokenizer: Transformer 分词器（可选）
            
        Returns:
            包含详细 FLOPs 信息的字典
        """
        # 1. 计算单个批次的前向传播 FLOPs
        forward_flops_per_batch = self.calculate_forward_flops(
            batch_size, seq_size, transformer_tokenizer
        )
        
        # 2. 计算单个批次的反向传播 FLOPs
        backward_flops_per_batch = self.calculate_backward_flops(forward_flops_per_batch)
        
        # 3. 计算每个 epoch 的批次数
        batches_per_epoch = math.ceil(num_samples_per_epoch / batch_size)
        
        # 4. 计算每个 epoch 的总 FLOPs
        epoch_forward_flops = forward_flops_per_batch * batches_per_epoch
        epoch_backward_flops = backward_flops_per_batch * batches_per_epoch
        epoch_total_flops = epoch_forward_flops + epoch_backward_flops
        
        # 5. 计算相关指标
        total_tokens_per_epoch = num_samples_per_epoch * seq_size
        flops_per_token = epoch_total_flops / total_tokens_per_epoch
        forward_flops_per_token = epoch_forward_flops / total_tokens_per_epoch
        backward_flops_per_token = epoch_backward_flops / total_tokens_per_epoch
        
        return {
            # 基本配置
            'model_name': self.model_name,
            'batch_size': batch_size,
            'seq_size': seq_size,
            'num_samples_per_epoch': num_samples_per_epoch,
            'batches_per_epoch': batches_per_epoch,
            'total_tokens_per_epoch': total_tokens_per_epoch,
            
            # 模型信息
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            
            # 单批次 FLOPs
            'forward_flops_per_batch': forward_flops_per_batch,
            'backward_flops_per_batch': backward_flops_per_batch,
            'total_flops_per_batch': forward_flops_per_batch + backward_flops_per_batch,
            
            # Epoch FLOPs
            'epoch_forward_flops': epoch_forward_flops,
            'epoch_backward_flops': epoch_backward_flops,
            'epoch_total_flops': epoch_total_flops,
            
            # 效率指标
            'flops_per_token': flops_per_token,
            'forward_flops_per_token': forward_flops_per_token,
            'backward_flops_per_token': backward_flops_per_token,
            
            # 公式系数
            'flops_coefficient': flops_per_token,  # 用于生成公式
        }
    
    def generate_formulas(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        生成与 batch_size 和 seq_size 相关的计算公式
        
        Args:
            results: calculate_epoch_flops 的返回结果
            
        Returns:
            包含各种公式的字典
        """
        B = results['batch_size']
        L = results['seq_size']
        N = results['num_samples_per_epoch']
        
        # 计算系数
        C_total = results['flops_coefficient']  # 每个 token 的总 FLOPs
        C_forward = results['forward_flops_per_token']  # 每个 token 的前向 FLOPs
        C_backward = results['backward_flops_per_token']  # 每个 token 的反向 FLOPs
        
        formulas = {
            # 基本公式
            'forward_flops_per_batch': f"Forward_FLOPs(B, L) = {C_forward:.2e} × B × L",
            'backward_flops_per_batch': f"Backward_FLOPs(B, L) = {C_backward:.2e} × B × L ≈ {self.backward_factor} × Forward_FLOPs(B, L)",
            'total_flops_per_batch': f"Total_FLOPs(B, L) = {C_total:.2e} × B × L",
            
            # Epoch 公式
            'epoch_forward_flops': f"Epoch_Forward_FLOPs(B, L, N) = Forward_FLOPs(B, L) × ceil(N / B)",
            'epoch_backward_flops': f"Epoch_Backward_FLOPs(B, L, N) = Backward_FLOPs(B, L) × ceil(N / B)",
            'epoch_total_flops': f"Epoch_Total_FLOPs(B, L, N) = Total_FLOPs(B, L) × ceil(N / B)",
            
            # 简化公式（当 N >> B 时）
            'simplified_epoch_forward': f"Epoch_Forward_FLOPs ≈ {C_forward:.2e} × L × N",
            'simplified_epoch_backward': f"Epoch_Backward_FLOPs ≈ {C_backward:.2e} × L × N",
            'simplified_epoch_total': f"Epoch_Total_FLOPs ≈ {C_total:.2e} × L × N",
            
            # 多 epoch 训练
            'training_total_flops': f"Training_Total_FLOPs(B, L, N, E) = Epoch_Total_FLOPs(B, L, N) × E",
            'simplified_training_total': f"Training_Total_FLOPs ≈ {C_total:.2e} × L × N × E",
            
            # 变量说明
            'variables': "B = batch_size, L = seq_size, N = num_samples_per_epoch, E = num_epochs"
        }
        
        return formulas
    
    def print_comprehensive_report(self, results: Dict[str, Any]) -> None:
        """打印综合报告"""
        print("\n" + "=" * 100)
        print(f"🧮 神经网络预训练 FLOPs 计算报告")
        print(f"模型: {results['model_name']}")
        print("=" * 100)
        
        # 模型信息
        print(f"\n🏗️  模型信息:")
        print(f"  总参数量: {self._format_number(results['total_params'])}")
        print(f"  可训练参数: {self._format_number(results['trainable_params'])}")
        
        # 训练配置
        print(f"\n⚙️  训练配置:")
        print(f"  批量大小 (batch_size): {results['batch_size']}")
        print(f"  序列长度 (seq_size): {results['seq_size']}")
        print(f"  每个 epoch 样本数: {results['num_samples_per_epoch']:,}")
        print(f"  每个 epoch 批次数: {results['batches_per_epoch']:,}")
        print(f"  每个 epoch 总 tokens: {self._format_number(results['total_tokens_per_epoch'])}")
        
        # 单批次 FLOPs
        print(f"\n🔄 单批次 FLOPs:")
        print(f"  前向传播: {self._format_flops(results['forward_flops_per_batch'])}")
        print(f"  反向传播: {self._format_flops(results['backward_flops_per_batch'])} (≈ {self.backward_factor}x 前向)")
        print(f"  总计: {self._format_flops(results['total_flops_per_batch'])}")
        
        # Epoch FLOPs
        print(f"\n📈 每个 Epoch FLOPs:")
        print(f"  前向传播: {self._format_flops(results['epoch_forward_flops'])}")
        print(f"  反向传播: {self._format_flops(results['epoch_backward_flops'])}")
        print(f"  总计: {self._format_flops(results['epoch_total_flops'])}")
        
        # 效率指标
        print(f"\n⚡ 效率指标:")
        print(f"  FLOPs/Token (总): {results['flops_per_token']:.2f}")
        print(f"  FLOPs/Token (前向): {results['forward_flops_per_token']:.2f}")
        print(f"  FLOPs/Token (反向): {results['backward_flops_per_token']:.2f}")
        
        # 公式
        formulas = self.generate_formulas(results)
        print(f"\n📐 计算公式 (与 batch_size 和 seq_size 的关系):")
        print(f"  {formulas['variables']}")
        print(f"")
        print(f"  单批次公式:")
        print(f"    {formulas['forward_flops_per_batch']}")
        print(f"    {formulas['backward_flops_per_batch']}")
        print(f"    {formulas['total_flops_per_batch']}")
        print(f"")
        print(f"  每个 Epoch 公式:")
        print(f"    {formulas['epoch_forward_flops']}")
        print(f"    {formulas['epoch_backward_flops']}")
        print(f"    {formulas['epoch_total_flops']}")
        print(f"")
        print(f"  简化公式 (当 N >> B 时):")
        print(f"    {formulas['simplified_epoch_forward']}")
        print(f"    {formulas['simplified_epoch_backward']}")
        print(f"    {formulas['simplified_epoch_total']}")
        print(f"")
        print(f"  多 Epoch 训练:")
        print(f"    {formulas['training_total_flops']}")
        print(f"    {formulas['simplified_training_total']}")
        
        print("=" * 100)
    
    def analyze_scaling(self, 
                       base_batch_size: int = 8,
                       base_seq_size: int = 512,
                       num_samples: int = 10000) -> None:
        """
        分析 FLOPs 随 batch_size 和 seq_size 的缩放关系
        
        Args:
            base_batch_size: 基准批量大小
            base_seq_size: 基准序列长度
            num_samples: 样本数量
        """
        print(f"\n📊 FLOPs 缩放关系分析")
        print("=" * 80)
        
        # 分析 seq_size 的影响（固定 batch_size）
        print(f"🔍 固定 batch_size={base_batch_size}，变化 seq_size:")
        print(f"{'Seq Size':<10} {'Epoch FLOPs':<15} {'FLOPs/Token':<12} {'缩放倍数':<10}")
        print("-" * 50)
        
        base_flops = None
        seq_sizes = [128, 256, 512, 1024, 2048]
        
        for seq_size in seq_sizes:
            results = self.calculate_epoch_flops(base_batch_size, seq_size, num_samples)
            epoch_flops = results['epoch_total_flops']
            flops_per_token = results['flops_per_token']
            
            if base_flops is None:
                base_flops = epoch_flops
                scaling = 1.0
            else:
                scaling = epoch_flops / base_flops
            
            print(f"{seq_size:<10} {self._format_flops(epoch_flops):<15} {flops_per_token:<12.0f} {scaling:<10.2f}")
        
        # 分析 batch_size 的影响（固定 seq_size）
        print(f"\n🔍 固定 seq_size={base_seq_size}，变化 batch_size:")
        print(f"{'Batch Size':<12} {'Epoch FLOPs':<15} {'FLOPs/Sample':<15} {'缩放倍数':<10}")
        print("-" * 55)
        
        base_flops = None
        batch_sizes = [4, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            results = self.calculate_epoch_flops(batch_size, base_seq_size, num_samples)
            epoch_flops = results['epoch_total_flops']
            flops_per_sample = epoch_flops / num_samples
            
            if base_flops is None:
                base_flops = epoch_flops
                scaling = 1.0
            else:
                scaling = epoch_flops / base_flops
            
            print(f"{batch_size:<12} {self._format_flops(epoch_flops):<15} {self._format_flops(flops_per_sample):<15} {scaling:<10.2f}")
    
    def compare_different_models(self, models_dict: Dict[str, nn.Module], 
                               batch_size: int = 8, 
                               seq_size: int = 512,
                               num_samples: int = 10000) -> None:
        """
        比较不同模型的 FLOPs
        
        Args:
            models_dict: 模型字典 {name: model}
            batch_size: 批量大小
            seq_size: 序列长度
            num_samples: 样本数量
        """
        print(f"\n🔬 模型 FLOPs 对比分析")
        print("=" * 80)
        print(f"配置: batch_size={batch_size}, seq_size={seq_size}, samples={num_samples:,}")
        print("-" * 80)
        
        print(f"{'模型名称':<25} {'参数量':<12} {'Epoch FLOPs':<15} {'FLOPs/Token':<12}")
        print("-" * 70)
        
        for name, model in models_dict.items():
            calc = PretrainFLOPsCalculator(model, name)
            try:
                results = calc.calculate_epoch_flops(batch_size, seq_size, num_samples)
                params_str = calc._format_number(results['total_params'])
                flops_str = calc._format_flops(results['epoch_total_flops'])
                flops_per_token = results['flops_per_token']
                
                print(f"{name:<25} {params_str:<12} {flops_str:<15} {flops_per_token:<12.0f}")
            except Exception as e:
                print(f"{name:<25} {'Error':<12} {'Error':<15} {'Error':<12}")
    
    def _format_number(self, num: float) -> str:
        """格式化数字"""
        if num >= 1e12:
            return f"{num/1e12:.2f}T"
        elif num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return f"{num:.0f}"
    
    def _format_flops(self, flops: float) -> str:
        """格式化 FLOPs"""
        if CALFLOPS_AVAILABLE:
            try:
                return flops_to_string(flops)
            except:
                pass
        
        # 备用格式化方法
        if flops >= 1e15:
            return f"{flops/1e15:.2f} PFLOPs"
        elif flops >= 1e12:
            return f"{flops/1e12:.2f} TFLOPs"
        elif flops >= 1e9:
            return f"{flops/1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops/1e3:.2f} KFLOPs"
        else:
            return f"{flops:.0f} FLOPs"


def create_test_models() -> Dict[str, nn.Module]:
    """创建测试用的模型"""
    
    models = {}
    
    # 1. 简单线性模型
    models['Simple Linear'] = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10000)  # 假设词汇表大小为 10000
    )
    
    # 2. 小型 Transformer
    class SmallTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = 10000
            d_model = 256
            nhead = 4
            num_layers = 2
            
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(d_model, vocab_size)
            
        def forward(self, input_ids):
            seq_len = input_ids.size(1)
            x = self.embedding(input_ids)
            x = x + self.pos_encoding[:, :seq_len, :]
            x = self.transformer(x)
            return self.output_proj(x)
    
    models['Small Transformer'] = SmallTransformer()
    
    # 3. 中型 Transformer
    class MediumTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = 32000
            d_model = 512
            nhead = 8
            num_layers = 6
            
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(d_model, vocab_size)
            
        def forward(self, input_ids):
            seq_len = input_ids.size(1)
            x = self.embedding(input_ids)
            x = x + self.pos_encoding[:, :seq_len, :]
            x = self.transformer(x)
            return self.output_proj(x)
    
    models['Medium Transformer'] = MediumTransformer()
    
    return models


def main():
    """主函数 - 演示完整的预训练 FLOPs 计算流程"""
    
    print("🎯 神经网络预训练 FLOPs 计算器")
    print("基于 calculate-flops.pytorch 库")
    print("=" * 80)
    
    # 创建测试模型
    models = create_test_models()
    
    # 选择一个模型进行详细分析
    model_name = "Medium Transformer"
    model = models[model_name]
    
    calculator = PretrainFLOPsCalculator(model, model_name)
    
    # 计算典型预训练场景的 FLOPs
    batch_size = 16
    seq_size = 1024
    num_samples_per_epoch = 50000
    
    print(f"\n🎯 主要分析 - {model_name}")
    results = calculator.calculate_epoch_flops(
        batch_size=batch_size,
        seq_size=seq_size,
        num_samples_per_epoch=num_samples_per_epoch
    )
    
    # 打印详细报告
    calculator.print_comprehensive_report(results)
    
    # 缩放关系分析
    calculator.analyze_scaling(base_batch_size=16, base_seq_size=1024, num_samples=50000)
    
    # 模型对比
    calculator.compare_different_models(models, batch_size=8, seq_size=512, num_samples=10000)
    
    # 保存结果
    output_file = '/workspace/pretrain_flops_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        # 添加公式到结果中
        results['formulas'] = calculator.generate_formulas(results)
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    # 总结关键信息
    print(f"\n✨ 关键总结:")
    print(f"  📌 每个 epoch 前向传播 FLOPs: {calculator._format_flops(results['epoch_forward_flops'])}")
    print(f"  📌 每个 epoch 反向传播 FLOPs: {calculator._format_flops(results['epoch_backward_flops'])}")
    print(f"  📌 每个 epoch 总 FLOPs: {calculator._format_flops(results['epoch_total_flops'])}")
    print(f"  📌 FLOPs 与参数的关系: 总 FLOPs ≈ {results['flops_coefficient']:.2e} × batch_size × seq_size")
    print(f"  📌 反向传播倍数: {calculator.backward_factor}x")


if __name__ == "__main__":
    main()