#!/usr/bin/env python3
"""
简单的 FLOPs 计算示例

展示如何使用 calculate-flops.pytorch 库计算神经网络预训练时的 FLOPs。
包含前向传播、反向传播计算量，以及与 batch_size 和 seq_size 的关系。

Author: AI Assistant
"""

import torch
import torch.nn as nn
import sys
import os

# 添加 calculate-flops.pytorch 到路径
sys.path.append('/workspace/calculate-flops.pytorch')

try:
    from calflops import calculate_flops
    from calflops.utils import flops_to_string, params_to_string
except ImportError:
    print("❌ 无法导入 calflops 库")
    print("请确保 calculate-flops.pytorch 库在正确位置")
    sys.exit(1)


def calculate_pretrain_flops(model, batch_size, seq_size, num_samples_per_epoch, num_epochs=1):
    """
    计算预训练过程的 FLOPs
    
    Args:
        model: PyTorch 模型
        batch_size: 批量大小
        seq_size: 序列长度
        num_samples_per_epoch: 每个 epoch 的样本数
        num_epochs: epoch 数量
        
    Returns:
        FLOPs 计算结果字典
    """
    print(f"🧮 计算模型 FLOPs...")
    print(f"  批量大小: {batch_size}")
    print(f"  序列长度: {seq_size}")
    print(f"  每个 epoch 样本数: {num_samples_per_epoch:,}")
    print(f"  Epochs: {num_epochs}")
    
    # 1. 计算单个批次的前向传播 FLOPs
    input_shape = (batch_size, seq_size)
    
    try:
        forward_flops, macs, params = calculate_flops(
            model=model,
            input_shape=input_shape,
            include_backPropagation=False,  # 只计算前向传播
            print_results=False,
            output_as_string=False
        )
        print(f"✅ 成功计算前向传播 FLOPs")
    except Exception as e:
        print(f"❌ FLOPs 计算失败: {e}")
        return None
    
    # 2. 估算反向传播 FLOPs（通常是前向传播的2倍）
    backward_factor = 2.0
    backward_flops = forward_flops * backward_factor
    total_flops_per_batch = forward_flops + backward_flops
    
    # 3. 计算每个 epoch 的总 FLOPs
    batches_per_epoch = (num_samples_per_epoch + batch_size - 1) // batch_size  # 向上取整
    
    forward_flops_per_epoch = forward_flops * batches_per_epoch
    backward_flops_per_epoch = backward_flops * batches_per_epoch
    total_flops_per_epoch = total_flops_per_batch * batches_per_epoch
    
    # 4. 计算整个训练过程的 FLOPs
    total_training_flops = total_flops_per_epoch * num_epochs
    
    # 5. 计算单个 token 的平均 FLOPs
    total_tokens = batch_size * seq_size * batches_per_epoch * num_epochs
    flops_per_token = total_training_flops / total_tokens
    
    results = {
        'model_params': params,
        'model_params_str': params_to_string(params),
        
        # 单批次 FLOPs
        'forward_flops_per_batch': forward_flops,
        'backward_flops_per_batch': backward_flops,
        'total_flops_per_batch': total_flops_per_batch,
        
        # 每个 epoch FLOPs
        'batches_per_epoch': batches_per_epoch,
        'forward_flops_per_epoch': forward_flops_per_epoch,
        'backward_flops_per_epoch': backward_flops_per_epoch,
        'total_flops_per_epoch': total_flops_per_epoch,
        
        # 整个训练 FLOPs
        'total_training_flops': total_training_flops,
        'total_tokens': total_tokens,
        'flops_per_token': flops_per_token,
        
        # 格式化字符串
        'forward_flops_per_batch_str': flops_to_string(forward_flops),
        'backward_flops_per_batch_str': flops_to_string(backward_flops),
        'total_flops_per_batch_str': flops_to_string(total_flops_per_batch),
        'forward_flops_per_epoch_str': flops_to_string(forward_flops_per_epoch),
        'backward_flops_per_epoch_str': flops_to_string(backward_flops_per_epoch),
        'total_flops_per_epoch_str': flops_to_string(total_flops_per_epoch),
        'total_training_flops_str': flops_to_string(total_training_flops),
    }
    
    return results


def print_flops_report(results, model_name="Neural Network"):
    """打印 FLOPs 计算报告"""
    print("\n" + "=" * 80)
    print(f"📊 {model_name} FLOPs 计算报告")
    print("=" * 80)
    
    print(f"🏗️  模型信息:")
    print(f"  参数量: {results['model_params_str']}")
    
    print(f"\n🔄 单批次 FLOPs:")
    print(f"  前向传播: {results['forward_flops_per_batch_str']}")
    print(f"  反向传播: {results['backward_flops_per_batch_str']} (≈ 2x 前向)")
    print(f"  总计: {results['total_flops_per_batch_str']}")
    
    print(f"\n📈 每个 Epoch FLOPs:")
    print(f"  批次数: {results['batches_per_epoch']:,}")
    print(f"  前向传播: {results['forward_flops_per_epoch_str']}")
    print(f"  反向传播: {results['backward_flops_per_epoch_str']}")
    print(f"  总计: {results['total_flops_per_epoch_str']}")
    
    print(f"\n🎯 训练总计:")
    print(f"  总 FLOPs: {results['total_training_flops_str']}")
    print(f"  总 tokens: {results['total_tokens']:,}")
    print(f"  FLOPs/Token: {results['flops_per_token']:.2f}")
    
    print("=" * 80)


def generate_flops_formulas(batch_size, seq_size, flops_per_token):
    """生成与 batch_size 和 seq_size 相关的公式"""
    print(f"\n📐 FLOPs 计算公式 (基于 batch_size={batch_size}, seq_size={seq_size}):")
    print("-" * 60)
    
    print(f"设:")
    print(f"  B = batch_size")
    print(f"  L = seq_size") 
    print(f"  N = num_samples_per_epoch")
    print(f"  E = num_epochs")
    print(f"  C = {flops_per_token:.2f} (每个 token 的平均 FLOPs)")
    
    print(f"\n公式:")
    print(f"  前向传播 FLOPs/batch = C * B * L / 3")
    print(f"  反向传播 FLOPs/batch = 2 * (前向传播 FLOPs/batch)")
    print(f"  总 FLOPs/batch = 前向传播 FLOPs/batch + 反向传播 FLOPs/batch = C * B * L")
    print(f"  ")
    print(f"  每个 epoch FLOPs = 总 FLOPs/batch * (N / B) = C * L * N")
    print(f"  整个训练 FLOPs = 每个 epoch FLOPs * E = C * L * N * E")
    
    print(f"\n💡 关键观察:")
    print(f"  - FLOPs 与 batch_size (B) 线性相关")
    print(f"  - FLOPs 与 seq_size (L) 线性相关（对于这个简化模型）")
    print(f"  - 对于 Transformer 模型，实际上 FLOPs ∝ L² （因为注意力机制）")
    print(f"  - 反向传播通常是前向传播的 2 倍计算量")


def test_different_sizes():
    """测试不同 batch_size 和 seq_size 的影响"""
    print("\n🧪 测试不同大小参数的影响")
    print("=" * 60)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.linear1 = nn.Linear(128, 256)
            self.linear2 = nn.Linear(256, 1000)
            
        def forward(self, x):
            x = self.embedding(x)
            x = torch.mean(x, dim=1)  # 简单池化
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    
    # 测试配置
    test_configs = [
        (2, 128),
        (4, 256),
        (8, 512),
        (16, 1024)
    ]
    
    print(f"{'Batch':<8} {'Seq':<8} {'Forward':<12} {'Backward':<12} {'Total':<12} {'Ratio':<8}")
    print("-" * 60)
    
    base_flops = None
    
    for batch_size, seq_size in test_configs:
        results = calculate_pretrain_flops(
            model=model,
            batch_size=batch_size,
            seq_size=seq_size,
            num_samples_per_epoch=1000,
            num_epochs=1
        )
        
        if results:
            total_flops = results['total_flops_per_epoch']
            if base_flops is None:
                base_flops = total_flops
                ratio = 1.0
            else:
                ratio = total_flops / base_flops
            
            forward_str = flops_to_string(results['forward_flops_per_epoch'])
            backward_str = flops_to_string(results['backward_flops_per_epoch'])
            total_str = flops_to_string(total_flops)
            
            print(f"{batch_size:<8} {seq_size:<8} {forward_str:<12} {backward_str:<12} {total_str:<12} {ratio:<8.2f}")


def main():
    """主函数"""
    print("🎯 神经网络预训练 FLOPs 计算工具")
    print("基于 calculate-flops.pytorch 库")
    print("=" * 80)
    
    # 示例1: 简单模型
    print("\n📝 示例 1: 简单线性模型")
    simple_model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 1000)
    )
    
    results1 = calculate_pretrain_flops(
        model=simple_model,
        batch_size=32,
        seq_size=1,  # 对于非序列模型，seq_size 可以设为 1
        num_samples_per_epoch=50000,
        num_epochs=10
    )
    
    if results1:
        print_flops_report(results1, "简单线性模型")
        generate_flops_formulas(32, 1, results1['flops_per_token'])
    
    # 示例2: 测试不同大小的影响
    test_different_sizes()


if __name__ == "__main__":
    main()