#!/usr/bin/env python3
"""
快速 FLOPs 计算器

专门用于快速计算神经网络预训练时每个 epoch 的前向传播和反向传播 FLOPs。
输出与 batch_size 和 seq_size 相关的计算公式。

使用方法:
python quick_flops_calculator.py

Author: AI Assistant
"""

import torch
import torch.nn as nn
import sys
import math

# 添加 calculate-flops.pytorch 到路径
sys.path.append('/workspace/calculate-flops.pytorch')

try:
    from calflops import calculate_flops
    CALFLOPS_AVAILABLE = True
except ImportError:
    CALFLOPS_AVAILABLE = False


def quick_calculate_epoch_flops(model, batch_size, seq_size, num_samples_per_epoch):
    """
    快速计算每个 epoch 的 FLOPs
    
    Args:
        model: PyTorch 模型
        batch_size: 批量大小
        seq_size: 序列长度  
        num_samples_per_epoch: 每个 epoch 样本数
        
    Returns:
        tuple: (前向传播 FLOPs, 反向传播 FLOPs, 总 FLOPs, 公式系数)
    """
    
    # 1. 计算单批次前向传播 FLOPs
    if CALFLOPS_AVAILABLE:
        try:
            # 检查是否有 embedding 层
            has_embedding = any(isinstance(m, nn.Embedding) for m in model.modules())
            
            if has_embedding:
                # 创建整数输入
                device = next(model.parameters()).device
                vocab_size = 10000
                for m in model.modules():
                    if isinstance(m, nn.Embedding):
                        vocab_size = m.num_embeddings
                        break
                
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_size), 
                                        device=device, dtype=torch.long)
                forward_flops_batch, _, _ = calculate_flops(
                    model=model, args=[input_ids], include_backPropagation=False,
                    print_results=False, output_as_string=False
                )
            else:
                input_shape = (batch_size, seq_size, 512)
                forward_flops_batch, _, _ = calculate_flops(
                    model=model, input_shape=input_shape, include_backPropagation=False,
                    print_results=False, output_as_string=False
                )
        except:
            # 备用估算
            total_params = sum(p.numel() for p in model.parameters())
            forward_flops_batch = total_params * batch_size * seq_size * 2
    else:
        # 简单估算
        total_params = sum(p.numel() for p in model.parameters())
        forward_flops_batch = total_params * batch_size * seq_size * 2
    
    # 2. 计算反向传播 FLOPs
    backward_flops_batch = forward_flops_batch * 2.0
    total_flops_batch = forward_flops_batch + backward_flops_batch
    
    # 3. 计算每个 epoch FLOPs
    batches_per_epoch = math.ceil(num_samples_per_epoch / batch_size)
    
    epoch_forward_flops = forward_flops_batch * batches_per_epoch
    epoch_backward_flops = backward_flops_batch * batches_per_epoch
    epoch_total_flops = total_flops_batch * batches_per_epoch
    
    # 4. 计算公式系数
    total_tokens = num_samples_per_epoch * seq_size
    flops_per_token = epoch_total_flops / total_tokens
    
    return epoch_forward_flops, epoch_backward_flops, epoch_total_flops, flops_per_token


def format_flops(flops):
    """格式化 FLOPs"""
    if flops >= 1e15:
        return f"{flops/1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    else:
        return f"{flops/1e6:.2f} MFLOPs"


def print_flops_summary(model_name, batch_size, seq_size, num_samples_per_epoch, 
                       forward_flops, backward_flops, total_flops, flops_per_token):
    """打印 FLOPs 总结"""
    print(f"\n📊 {model_name} - 预训练 FLOPs 计算")
    print("=" * 60)
    print(f"配置: batch_size={batch_size}, seq_size={seq_size}, samples={num_samples_per_epoch:,}")
    print(f"")
    print(f"每个 Epoch FLOPs:")
    print(f"  前向传播: {format_flops(forward_flops)}")
    print(f"  反向传播: {format_flops(backward_flops)} (≈ 2x 前向)")
    print(f"  总计: {format_flops(total_flops)}")
    print(f"")
    print(f"📐 与 batch_size 和 seq_size 的关系:")
    print(f"  FLOPs/Token: {flops_per_token:.0f}")
    print(f"  Epoch_FLOPs(B, L, N) ≈ {flops_per_token:.2e} × L × N")
    print(f"  其中 B=batch_size, L=seq_size, N=num_samples_per_epoch")
    print("=" * 60)


# 创建测试模型
def create_test_transformer():
    """创建测试用 Transformer"""
    class TestTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10000, 256)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(256, 4, 1024, batch_first=True), 2
            )
            self.output = nn.Linear(256, 10000)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            return self.output(x)
    
    return TestTransformer()


def main():
    """主函数 - 演示用法"""
    print("🎯 快速 FLOPs 计算器")
    
    # 创建模型
    model = create_test_transformer()
    
    # 典型配置
    batch_size = 16
    seq_size = 1024  
    num_samples_per_epoch = 100000
    
    # 计算 FLOPs
    forward_flops, backward_flops, total_flops, flops_per_token = quick_calculate_epoch_flops(
        model, batch_size, seq_size, num_samples_per_epoch
    )
    
    # 打印结果
    print_flops_summary(
        "Test Transformer", batch_size, seq_size, num_samples_per_epoch,
        forward_flops, backward_flops, total_flops, flops_per_token
    )
    
    # 多配置对比
    print(f"\n🔍 不同配置对比:")
    print(f"{'Batch':<8} {'Seq':<8} {'Epoch FLOPs':<15} {'FLOPs/Token':<12}")
    print("-" * 45)
    
    test_configs = [(8, 512), (16, 1024), (32, 2048)]
    
    for b, s in test_configs:
        f, _, t, c = quick_calculate_epoch_flops(model, b, s, 10000)
        print(f"{b:<8} {s:<8} {format_flops(t):<15} {c:<12.0f}")


if __name__ == "__main__":
    main()