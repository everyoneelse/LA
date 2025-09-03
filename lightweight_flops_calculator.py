#!/usr/bin/env python3
"""
轻量级预训练 FLOPs 计算器

专门用于计算神经网络预训练时每个 epoch 的前向传播和反向传播 FLOPs。
表达为与 batch_size 和 seq_size 相关的量。

Author: AI Assistant
"""

import torch
import torch.nn as nn
import sys
import math
import json

# 添加 calculate-flops.pytorch 到路径
sys.path.append('/workspace/calculate-flops.pytorch')

try:
    from calflops import calculate_flops
    from calflops.utils import flops_to_string
    CALFLOPS_AVAILABLE = True
    print("✅ calflops 库加载成功")
except ImportError:
    CALFLOPS_AVAILABLE = False
    print("⚠️  calflops 库不可用，使用估算方法")


def format_flops(flops):
    """格式化 FLOPs 显示"""
    if CALFLOPS_AVAILABLE:
        try:
            return flops_to_string(flops)
        except:
            pass
    
    if flops >= 1e15:
        return f"{flops/1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    else:
        return f"{flops/1e3:.2f} KFLOPs"


def format_number(num):
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


def calculate_model_flops(model, batch_size, seq_size):
    """
    计算模型的前向传播 FLOPs
    
    Args:
        model: PyTorch 模型
        batch_size: 批量大小
        seq_size: 序列长度
        
    Returns:
        前向传播 FLOPs
    """
    if CALFLOPS_AVAILABLE:
        try:
            # 检查是否有 embedding 层
            has_embedding = any(isinstance(m, nn.Embedding) for m in model.modules())
            
            if has_embedding:
                # 为 embedding 模型创建整数输入
                device = next(model.parameters()).device
                vocab_size = 10000
                
                # 尝试获取实际词汇表大小
                for m in model.modules():
                    if isinstance(m, nn.Embedding):
                        vocab_size = m.num_embeddings
                        break
                
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_size), 
                                        device=device, dtype=torch.long)
                
                forward_flops, _, _ = calculate_flops(
                    model=model,
                    args=[input_ids],
                    include_backPropagation=False,
                    print_results=False,
                    output_as_string=False
                )
            else:
                # 普通模型
                input_shape = (batch_size, seq_size, 512)  # 假设特征维度
                forward_flops, _, _ = calculate_flops(
                    model=model,
                    input_shape=input_shape,
                    include_backPropagation=False,
                    print_results=False,
                    output_as_string=False
                )
            
            return forward_flops
            
        except Exception as e:
            print(f"⚠️  calflops 计算失败: {e}")
    
    # 使用简单估算
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * batch_size * seq_size * 2


def calculate_pretrain_epoch_flops(model, 
                                 batch_size, 
                                 seq_size, 
                                 num_samples_per_epoch,
                                 model_name="Neural Network"):
    """
    计算预训练每个 epoch 的 FLOPs
    
    Args:
        model: PyTorch 模型
        batch_size: 批量大小
        seq_size: 序列长度
        num_samples_per_epoch: 每个 epoch 的样本数
        model_name: 模型名称
        
    Returns:
        详细的 FLOPs 计算结果
    """
    print(f"🧮 计算 {model_name} 的预训练 FLOPs...")
    
    # 1. 计算前向传播 FLOPs（单批次）
    forward_flops_per_batch = calculate_model_flops(model, batch_size, seq_size)
    
    # 2. 计算反向传播 FLOPs（单批次）
    backward_factor = 2.0
    backward_flops_per_batch = forward_flops_per_batch * backward_factor
    total_flops_per_batch = forward_flops_per_batch + backward_flops_per_batch
    
    # 3. 计算每个 epoch 的批次数
    batches_per_epoch = math.ceil(num_samples_per_epoch / batch_size)
    
    # 4. 计算每个 epoch 的总 FLOPs
    epoch_forward_flops = forward_flops_per_batch * batches_per_epoch
    epoch_backward_flops = backward_flops_per_batch * batches_per_epoch
    epoch_total_flops = total_flops_per_batch * batches_per_epoch
    
    # 5. 计算相关指标
    total_tokens_per_epoch = num_samples_per_epoch * seq_size
    flops_per_token = epoch_total_flops / total_tokens_per_epoch
    
    # 6. 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        'model_name': model_name,
        'total_params': total_params,
        'batch_size': batch_size,
        'seq_size': seq_size,
        'num_samples_per_epoch': num_samples_per_epoch,
        'batches_per_epoch': batches_per_epoch,
        'total_tokens_per_epoch': total_tokens_per_epoch,
        
        # 单批次 FLOPs
        'forward_flops_per_batch': forward_flops_per_batch,
        'backward_flops_per_batch': backward_flops_per_batch,
        'total_flops_per_batch': total_flops_per_batch,
        
        # Epoch FLOPs
        'epoch_forward_flops': epoch_forward_flops,
        'epoch_backward_flops': epoch_backward_flops,
        'epoch_total_flops': epoch_total_flops,
        
        # 效率指标
        'flops_per_token': flops_per_token,
        'backward_factor': backward_factor
    }


def generate_flops_formulas(results):
    """生成 FLOPs 计算公式"""
    B = results['batch_size']
    L = results['seq_size']
    
    # 计算系数
    flops_per_token = results['flops_per_token']
    forward_flops_per_token = flops_per_token / 3  # 前向传播约占 1/3
    backward_flops_per_token = flops_per_token * 2 / 3  # 反向传播约占 2/3
    
    return {
        'variables': "B = batch_size, L = seq_size, N = num_samples_per_epoch, E = num_epochs",
        'forward_batch': f"Forward_FLOPs(B, L) = {forward_flops_per_token:.2e} × B × L",
        'backward_batch': f"Backward_FLOPs(B, L) = {backward_flops_per_token:.2e} × B × L",
        'total_batch': f"Total_FLOPs(B, L) = {flops_per_token:.2e} × B × L",
        'epoch_forward': f"Epoch_Forward_FLOPs(L, N) ≈ {forward_flops_per_token:.2e} × L × N",
        'epoch_backward': f"Epoch_Backward_FLOPs(L, N) ≈ {backward_flops_per_token:.2e} × L × N", 
        'epoch_total': f"Epoch_Total_FLOPs(L, N) ≈ {flops_per_token:.2e} × L × N",
        'training_total': f"Training_Total_FLOPs(L, N, E) ≈ {flops_per_token:.2e} × L × N × E"
    }


def print_results(results):
    """打印计算结果"""
    print("\n" + "=" * 80)
    print(f"📊 {results['model_name']} - 预训练 FLOPs 计算结果")
    print("=" * 80)
    
    print(f"\n🏗️  模型信息:")
    print(f"  参数量: {format_number(results['total_params'])}")
    
    print(f"\n⚙️  训练配置:")
    print(f"  批量大小: {results['batch_size']}")
    print(f"  序列长度: {results['seq_size']}")
    print(f"  每个 epoch 样本数: {results['num_samples_per_epoch']:,}")
    print(f"  每个 epoch 批次数: {results['batches_per_epoch']:,}")
    
    print(f"\n🔄 单批次 FLOPs:")
    print(f"  前向传播: {format_flops(results['forward_flops_per_batch'])}")
    print(f"  反向传播: {format_flops(results['backward_flops_per_batch'])} (≈ {results['backward_factor']}x 前向)")
    print(f"  总计: {format_flops(results['total_flops_per_batch'])}")
    
    print(f"\n📈 每个 Epoch FLOPs:")
    print(f"  前向传播: {format_flops(results['epoch_forward_flops'])}")
    print(f"  反向传播: {format_flops(results['epoch_backward_flops'])}")
    print(f"  总计: {format_flops(results['epoch_total_flops'])}")
    
    print(f"\n⚡ 效率指标:")
    print(f"  FLOPs/Token: {results['flops_per_token']:.0f}")
    
    # 打印公式
    formulas = generate_flops_formulas(results)
    print(f"\n📐 计算公式:")
    print(f"  {formulas['variables']}")
    print(f"")
    for key, formula in formulas.items():
        if key != 'variables':
            print(f"  {formula}")
    
    print("=" * 80)


def create_simple_transformer():
    """创建简单的 Transformer 模型用于测试"""
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = 10000
            d_model = 256
            nhead = 4
            num_layers = 2
            
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, 4*d_model, batch_first=True),
                num_layers
            )
            self.output_proj = nn.Linear(d_model, vocab_size)
            
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            return self.output_proj(x)
    
    return SimpleTransformer()


def demo_scaling_analysis():
    """演示缩放分析"""
    print("📈 FLOPs 缩放关系演示")
    print("=" * 60)
    
    model = create_simple_transformer()
    
    # 测试不同配置
    configs = [
        (4, 128),
        (8, 256), 
        (16, 512),
        (32, 1024)
    ]
    
    print(f"{'Batch':<8} {'Seq':<8} {'Forward FLOPs':<15} {'Backward FLOPs':<15} {'Total FLOPs':<15}")
    print("-" * 70)
    
    for batch_size, seq_size in configs:
        try:
            forward_flops = calculate_model_flops(model, batch_size, seq_size)
            backward_flops = forward_flops * 2.0
            total_flops = forward_flops + backward_flops
            
            print(f"{batch_size:<8} {seq_size:<8} {format_flops(forward_flops):<15} "
                  f"{format_flops(backward_flops):<15} {format_flops(total_flops):<15}")
        except Exception as e:
            print(f"{batch_size:<8} {seq_size:<8} {'Error':<15} {'Error':<15} {'Error':<15}")


def main():
    """主函数"""
    print("🎯 神经网络预训练 FLOPs 计算器")
    print("=" * 60)
    
    # 创建测试模型
    model = create_simple_transformer()
    
    # 典型预训练配置
    batch_size = 8
    seq_size = 512
    num_samples_per_epoch = 10000
    
    print(f"\n📋 计算配置:")
    print(f"  批量大小: {batch_size}")
    print(f"  序列长度: {seq_size}")
    print(f"  每个 epoch 样本数: {num_samples_per_epoch:,}")
    
    # 计算 FLOPs
    results = calculate_pretrain_epoch_flops(
        model=model,
        batch_size=batch_size,
        seq_size=seq_size,
        num_samples_per_epoch=num_samples_per_epoch,
        model_name="Simple Transformer"
    )
    
    # 打印结果
    print_results(results)
    
    # 缩放分析
    demo_scaling_analysis()
    
    # 保存结果
    with open('/workspace/flops_results.json', 'w') as f:
        # 添加公式
        results['formulas'] = generate_flops_formulas(results)
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存到: /workspace/flops_results.json")
    
    # 关键总结
    print(f"\n✨ 关键总结:")
    print(f"  📌 每个 epoch 前向传播 FLOPs: {format_flops(results['epoch_forward_flops'])}")
    print(f"  📌 每个 epoch 反向传播 FLOPs: {format_flops(results['epoch_backward_flops'])}")
    print(f"  📌 每个 epoch 总 FLOPs: {format_flops(results['epoch_total_flops'])}")
    print(f"  📌 简化公式: Epoch_FLOPs ≈ {results['flops_per_token']:.2e} × seq_size × num_samples")


if __name__ == "__main__":
    main()