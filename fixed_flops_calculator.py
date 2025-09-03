#!/usr/bin/env python3
"""
修复版本的 FLOPs 计算器

解决了输入数据类型和模型结构的问题。
专门用于计算神经网络预训练时的前向传播和反向传播 FLOPs。

Author: AI Assistant
"""

import torch
import torch.nn as nn
import sys
import os
import math

# 添加 calculate-flops.pytorch 到路径
sys.path.append('/workspace/calculate-flops.pytorch')

try:
    from calflops import calculate_flops
    from calflops.utils import flops_to_string, params_to_string
except ImportError:
    print("❌ 无法导入 calflops 库")
    print("尝试直接使用本地实现...")


class FLOPsCalculator:
    """FLOPs 计算器"""
    
    def __init__(self, model, model_name="Neural Network"):
        self.model = model
        self.model_name = model_name
        self.backward_factor = 2.0  # 反向传播是前向传播的2倍
    
    def calculate_flops_for_batch(self, batch_size, seq_size, use_calflops=True):
        """计算单个批次的 FLOPs"""
        
        if use_calflops:
            try:
                return self._calculate_with_calflops(batch_size, seq_size)
            except Exception as e:
                print(f"⚠️  calflops 计算失败: {e}")
                print("🔄 切换到手动计算方法...")
                return self._calculate_manually(batch_size, seq_size)
        else:
            return self._calculate_manually(batch_size, seq_size)
    
    def _calculate_with_calflops(self, batch_size, seq_size):
        """使用 calflops 库计算"""
        # 为 embedding 模型创建正确的输入
        if hasattr(self.model, 'embedding') or any(isinstance(m, nn.Embedding) for m in self.model.modules()):
            # 对于包含 embedding 的模型，需要整数类型的输入
            input_shape = (batch_size, seq_size)
            
            # 创建模拟输入数据而不是依赖 calflops 自动生成
            device = next(self.model.parameters()).device
            input_ids = torch.randint(0, 1000, input_shape, device=device, dtype=torch.long)
            
            # 使用 args 参数传递具体的输入数据
            forward_flops, macs, params = calculate_flops(
                model=self.model,
                args=[input_ids],
                include_backPropagation=False,
                print_results=False,
                output_as_string=False
            )
        else:
            # 对于普通模型，使用 input_shape
            input_shape = (batch_size, seq_size, 512)  # 假设特征维度为 512
            forward_flops, macs, params = calculate_flops(
                model=self.model,
                input_shape=input_shape,
                include_backPropagation=False,
                print_results=False,
                output_as_string=False
            )
        
        backward_flops = forward_flops * self.backward_factor
        total_flops = forward_flops + backward_flops
        
        return {
            'forward_flops': forward_flops,
            'backward_flops': backward_flops,
            'total_flops': total_flops,
            'macs': macs,
            'params': params
        }
    
    def _calculate_manually(self, batch_size, seq_size):
        """手动计算 FLOPs（估算方法）"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # 简单估算：每个参数在前向传播中大约进行一次乘法和一次加法
        # 这是一个粗略的估算
        forward_flops = total_params * batch_size * seq_size * 2
        backward_flops = forward_flops * self.backward_factor
        total_flops = forward_flops + backward_flops
        
        return {
            'forward_flops': forward_flops,
            'backward_flops': backward_flops,
            'total_flops': total_flops,
            'macs': forward_flops / 2,
            'params': total_params,
            'method': 'manual_estimation'
        }
    
    def calculate_epoch_flops(self, batch_size, seq_size, num_samples_per_epoch):
        """计算每个 epoch 的 FLOPs"""
        
        # 计算单个批次的 FLOPs
        batch_results = self.calculate_flops_for_batch(batch_size, seq_size)
        
        # 计算 epoch 的批次数
        batches_per_epoch = math.ceil(num_samples_per_epoch / batch_size)
        
        # 计算 epoch 总 FLOPs
        epoch_forward_flops = batch_results['forward_flops'] * batches_per_epoch
        epoch_backward_flops = batch_results['backward_flops'] * batches_per_epoch
        epoch_total_flops = batch_results['total_flops'] * batches_per_epoch
        
        return {
            'batch_size': batch_size,
            'seq_size': seq_size,
            'num_samples_per_epoch': num_samples_per_epoch,
            'batches_per_epoch': batches_per_epoch,
            
            # 单批次 FLOPs
            'forward_flops_per_batch': batch_results['forward_flops'],
            'backward_flops_per_batch': batch_results['backward_flops'],
            'total_flops_per_batch': batch_results['total_flops'],
            
            # Epoch FLOPs
            'epoch_forward_flops': epoch_forward_flops,
            'epoch_backward_flops': epoch_backward_flops,
            'epoch_total_flops': epoch_total_flops,
            
            # 其他信息
            'model_params': batch_results['params'],
            'method': batch_results.get('method', 'calflops')
        }
    
    def print_detailed_report(self, results):
        """打印详细报告"""
        print("\n" + "=" * 80)
        print(f"📊 {self.model_name} - FLOPs 计算报告")
        print("=" * 80)
        
        print(f"🏗️  模型信息:")
        print(f"  参数量: {self._format_number(results['model_params'])}")
        print(f"  计算方法: {results['method']}")
        
        print(f"\n⚙️  训练配置:")
        print(f"  批量大小 (batch_size): {results['batch_size']}")
        print(f"  序列长度 (seq_size): {results['seq_size']}")
        print(f"  每个 epoch 样本数: {results['num_samples_per_epoch']:,}")
        print(f"  每个 epoch 批次数: {results['batches_per_epoch']:,}")
        
        print(f"\n🔄 单批次 FLOPs:")
        print(f"  前向传播: {self._format_flops(results['forward_flops_per_batch'])}")
        print(f"  反向传播: {self._format_flops(results['backward_flops_per_batch'])} (≈ {self.backward_factor}x 前向)")
        print(f"  总计: {self._format_flops(results['total_flops_per_batch'])}")
        
        print(f"\n📈 每个 Epoch FLOPs:")
        print(f"  前向传播: {self._format_flops(results['epoch_forward_flops'])}")
        print(f"  反向传播: {self._format_flops(results['epoch_backward_flops'])}")
        print(f"  总计: {self._format_flops(results['epoch_total_flops'])}")
        
        # 计算与 batch_size 和 seq_size 的关系
        flops_per_token = results['total_flops_per_batch'] / (results['batch_size'] * results['seq_size'])
        
        print(f"\n📐 FLOPs 公式 (与 batch_size 和 seq_size 的关系):")
        print(f"  设 B = batch_size, L = seq_size, N = num_samples_per_epoch")
        print(f"  FLOPs/Token ≈ {flops_per_token:.2e}")
        print(f"  ")
        print(f"  前向传播 FLOPs/batch = {flops_per_token/3:.2e} * B * L")
        print(f"  反向传播 FLOPs/batch = {self.backward_factor} * 前向传播 FLOPs/batch")
        print(f"  总 FLOPs/batch = {flops_per_token:.2e} * B * L")
        print(f"  ")
        print(f"  每个 epoch FLOPs = 总 FLOPs/batch * ceil(N / B)")
        print(f"                   ≈ {flops_per_token:.2e} * L * N  (当 N >> B 时)")
        
        print("=" * 80)
    
    def _format_number(self, num):
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
    
    def _format_flops(self, flops):
        """格式化 FLOPs"""
        try:
            return flops_to_string(flops)
        except:
            return self._format_number(flops) + " FLOPs"


def create_simple_transformer(vocab_size=10000, d_model=256, nhead=4, num_layers=2):
    """创建简单的 Transformer 模型"""
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))
            
            # Transformer 编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # 输出层
            self.output_proj = nn.Linear(d_model, vocab_size)
            
        def forward(self, input_ids):
            seq_len = input_ids.size(1)
            
            # Token embeddings
            x = self.embedding(input_ids)
            
            # 添加位置编码
            x = x + self.pos_encoding[:, :seq_len, :]
            
            # Transformer 处理
            x = self.transformer(x)
            
            # 输出投影
            logits = self.output_proj(x)
            
            return logits
    
    return SimpleTransformer()


def create_simple_cnn():
    """创建简单的 CNN 模型"""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, 1000)
    )


def demo_transformer_flops():
    """演示 Transformer 模型的 FLOPs 计算"""
    print("🤖 Transformer 模型 FLOPs 计算演示")
    print("=" * 60)
    
    # 创建模型
    model = create_simple_transformer(
        vocab_size=10000,
        d_model=256,
        nhead=4,
        num_layers=2
    )
    
    calculator = FLOPsCalculator(model, "Simple Transformer (256d, 2L, 4H)")
    
    # 计算不同配置下的 FLOPs
    configs = [
        (4, 128),   # 小批次，短序列
        (8, 256),   # 中等批次，中等序列
        (16, 512),  # 大批次，长序列
    ]
    
    print(f"{'Batch Size':<12} {'Seq Size':<10} {'Forward FLOPs':<15} {'Backward FLOPs':<15} {'Total FLOPs':<15}")
    print("-" * 75)
    
    for batch_size, seq_size in configs:
        results = calculator.calculate_epoch_flops(
            batch_size=batch_size,
            seq_size=seq_size,
            num_samples_per_epoch=1000
        )
        
        forward_str = calculator._format_flops(results['forward_flops_per_batch'])
        backward_str = calculator._format_flops(results['backward_flops_per_batch'])
        total_str = calculator._format_flops(results['total_flops_per_batch'])
        
        print(f"{batch_size:<12} {seq_size:<10} {forward_str:<15} {backward_str:<15} {total_str:<15}")
    
    # 详细报告一个配置
    print(f"\n📋 详细报告 (batch_size=8, seq_size=256):")
    detailed_results = calculator.calculate_epoch_flops(8, 256, 10000)
    calculator.print_detailed_report(detailed_results)


def demo_cnn_flops():
    """演示 CNN 模型的 FLOPs 计算"""
    print("\n🖼️  CNN 模型 FLOPs 计算演示")
    print("=" * 60)
    
    model = create_simple_cnn()
    calculator = FLOPsCalculator(model, "Simple CNN")
    
    # 对于 CNN，我们使用图像尺寸作为 "seq_size"
    configs = [
        (8, 224),   # batch_size=8, image_size=224x224
        (16, 224),  # batch_size=16, image_size=224x224
        (32, 224),  # batch_size=32, image_size=224x224
    ]
    
    print(f"{'Batch Size':<12} {'Image Size':<12} {'Forward FLOPs':<15} {'Backward FLOPs':<15} {'Total FLOPs':<15}")
    print("-" * 75)
    
    for batch_size, image_size in configs:
        try:
            # 对于 CNN，直接使用 calflops
            input_shape = (batch_size, 3, image_size, image_size)
            forward_flops, macs, params = calculate_flops(
                model=model,
                input_shape=input_shape,
                include_backPropagation=False,
                print_results=False,
                output_as_string=False
            )
            
            backward_flops = forward_flops * 2.0
            total_flops = forward_flops + backward_flops
            
            forward_str = calculator._format_flops(forward_flops)
            backward_str = calculator._format_flops(backward_flops)
            total_str = calculator._format_flops(total_flops)
            
            print(f"{batch_size:<12} {image_size}x{image_size:<7} {forward_str:<15} {backward_str:<15} {total_str:<15}")
            
        except Exception as e:
            print(f"{batch_size:<12} {image_size}x{image_size:<7} {'Error':<15} {'Error':<15} {'Error':<15}")


def analyze_scaling_relationship():
    """分析 FLOPs 与 batch_size 和 seq_size 的缩放关系"""
    print("\n📈 FLOPs 缩放关系分析")
    print("=" * 60)
    
    # 创建简单模型进行分析
    model = create_simple_transformer(vocab_size=5000, d_model=128, nhead=2, num_layers=1)
    calculator = FLOPsCalculator(model, "Tiny Transformer")
    
    # 固定 batch_size，变化 seq_size
    print("🔍 固定 batch_size=4，变化 seq_size:")
    print(f"{'Seq Size':<10} {'Total FLOPs':<15} {'FLOPs/Token':<12} {'Scaling Factor':<15}")
    print("-" * 55)
    
    base_flops = None
    base_seq_size = None
    
    for seq_size in [64, 128, 256, 512]:
        results = calculator.calculate_flops_for_batch(4, seq_size, use_calflops=False)
        total_flops = results['total_flops']
        flops_per_token = total_flops / (4 * seq_size)
        
        if base_flops is None:
            base_flops = total_flops
            base_seq_size = seq_size
            scaling_factor = 1.0
        else:
            scaling_factor = total_flops / base_flops
        
        print(f"{seq_size:<10} {calculator._format_flops(total_flops):<15} {flops_per_token:<12.2f} {scaling_factor:<15.2f}")
    
    # 固定 seq_size，变化 batch_size
    print(f"\n🔍 固定 seq_size=128，变化 batch_size:")
    print(f"{'Batch Size':<12} {'Total FLOPs':<15} {'FLOPs/Sample':<15} {'Scaling Factor':<15}")
    print("-" * 60)
    
    base_flops = None
    
    for batch_size in [2, 4, 8, 16]:
        results = calculator.calculate_flops_for_batch(batch_size, 128, use_calflops=False)
        total_flops = results['total_flops']
        flops_per_sample = total_flops / batch_size
        
        if base_flops is None:
            base_flops = total_flops
            scaling_factor = 1.0
        else:
            scaling_factor = total_flops / base_flops
        
        print(f"{batch_size:<12} {calculator._format_flops(total_flops):<15} {calculator._format_flops(flops_per_sample):<15} {scaling_factor:<15.2f}")


def practical_usage_example():
    """实际使用示例"""
    print("\n💼 实际使用示例")
    print("=" * 60)
    
    # 模拟一个实际的预训练场景
    model = create_simple_transformer(
        vocab_size=32000,   # 常见的词汇表大小
        d_model=512,        # 中等模型大小
        nhead=8,           # 8个注意力头
        num_layers=6       # 6层
    )
    
    calculator = FLOPsCalculator(model, "实际 Transformer 模型 (512d, 6L, 8H)")
    
    # 实际训练参数
    batch_size = 16
    seq_size = 1024
    samples_per_epoch = 100000  # 10万样本
    num_epochs = 10
    
    print(f"📋 实际训练配置:")
    print(f"  模型: {calculator.model_name}")
    print(f"  批量大小: {batch_size}")
    print(f"  序列长度: {seq_size}")
    print(f"  每个 epoch 样本数: {samples_per_epoch:,}")
    print(f"  总 epochs: {num_epochs}")
    
    # 计算 epoch FLOPs
    epoch_results = calculator.calculate_epoch_flops(batch_size, seq_size, samples_per_epoch)
    
    # 计算总训练 FLOPs
    total_training_flops = epoch_results['epoch_total_flops'] * num_epochs
    total_tokens = batch_size * seq_size * epoch_results['batches_per_epoch'] * num_epochs
    
    print(f"\n📊 计算结果:")
    print(f"  每个 epoch FLOPs: {calculator._format_flops(epoch_results['epoch_total_flops'])}")
    print(f"  总训练 FLOPs: {calculator._format_flops(total_training_flops)}")
    print(f"  总处理 tokens: {calculator._format_number(total_tokens)}")
    print(f"  FLOPs/Token: {total_training_flops/total_tokens:.2f}")
    
    # 显示公式
    flops_per_token = epoch_results['total_flops_per_batch'] / (batch_size * seq_size)
    
    print(f"\n📐 通用公式:")
    print(f"  每批次 FLOPs(B, L) ≈ {flops_per_token:.2e} * B * L")
    print(f"  每个 epoch FLOPs(B, L, N) ≈ {flops_per_token:.2e} * L * N")
    print(f"  总训练 FLOPs(B, L, N, E) ≈ {flops_per_token:.2e} * L * N * E")
    print(f"  ")
    print(f"  其中: B=batch_size, L=seq_size, N=samples_per_epoch, E=num_epochs")


def main():
    """主函数"""
    print("🎯 神经网络预训练 FLOPs 计算工具")
    print("基于 calculate-flops.pytorch 库")
    print("=" * 80)
    
    # 演示 Transformer 模型
    demo_transformer_flops()
    
    # 演示 CNN 模型
    demo_cnn_flops()
    
    # 分析缩放关系
    analyze_scaling_relationship()
    
    # 实际使用示例
    practical_usage_example()
    
    print(f"\n✅ 演示完成！")
    print(f"💡 关键要点:")
    print(f"  1. 前向传播 FLOPs 可以通过 calflops 库精确计算")
    print(f"  2. 反向传播 FLOPs ≈ 2 * 前向传播 FLOPs")
    print(f"  3. 总 FLOPs 与 batch_size 和 seq_size 呈线性关系")
    print(f"  4. 对于 Transformer 模型，注意力机制使 FLOPs ∝ seq_size²")


if __name__ == "__main__":
    main()