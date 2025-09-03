#!/usr/bin/env python3
"""
预训练 FLOPs 跟踪器

专门用于神经网络预训练过程中的 FLOPs 计算和跟踪。
支持实时监控每个 epoch 的前向传播和反向传播计算量。

使用方法:
1. 在训练循环开始前初始化 PretrainFLOPsTracker
2. 在每个训练步骤中调用 track_training_step
3. 在每个 epoch 结束时调用 get_epoch_summary

Author: AI Assistant
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import time
import json
import sys
import os
from collections import defaultdict
import math

# 添加 calculate-flops.pytorch 到路径
sys.path.append('/workspace/calculate-flops.pytorch')

try:
    from calflops import calculate_flops
    from calflops.utils import flops_to_string, params_to_string
except ImportError:
    print("请确保 calculate-flops.pytorch 库已正确安装")
    print("运行: pip install calflops")
    sys.exit(1)


class PretrainFLOPsTracker:
    """预训练 FLOPs 跟踪器"""
    
    def __init__(self, 
                 model: nn.Module,
                 model_name: str = "Neural Network",
                 backward_factor: float = 2.0,
                 log_interval: int = 100):
        """
        初始化预训练 FLOPs 跟踪器
        
        Args:
            model: PyTorch 模型
            model_name: 模型名称
            backward_factor: 反向传播相对于前向传播的计算倍数
            log_interval: 日志打印间隔（步数）
        """
        self.model = model
        self.model_name = model_name
        self.backward_factor = backward_factor
        self.log_interval = log_interval
        
        # 统计信息
        self.total_steps = 0
        self.total_epochs = 0
        self.total_forward_flops = 0
        self.total_backward_flops = 0
        self.total_training_flops = 0
        self.total_tokens_processed = 0
        
        # 当前 epoch 统计
        self.current_epoch_steps = 0
        self.current_epoch_forward_flops = 0
        self.current_epoch_backward_flops = 0
        self.current_epoch_tokens = 0
        
        # 缓存的单样本 FLOPs（避免重复计算）
        self.cached_flops = {}
        
        # 时间统计
        self.start_time = None
        self.epoch_start_time = None
        
        # 历史记录
        self.epoch_history = []
        
    def _get_cache_key(self, batch_size: int, seq_size: int) -> str:
        """生成缓存键"""
        return f"{batch_size}_{seq_size}"
    
    def _calculate_batch_flops(self, 
                              batch_size: int, 
                              seq_size: int,
                              transformer_tokenizer=None) -> Dict[str, float]:
        """
        计算单个批次的 FLOPs
        
        Args:
            batch_size: 批量大小
            seq_size: 序列长度
            transformer_tokenizer: 分词器（可选）
            
        Returns:
            包含批次 FLOPs 信息的字典
        """
        cache_key = self._get_cache_key(batch_size, seq_size)
        
        # 检查缓存
        if cache_key in self.cached_flops:
            return self.cached_flops[cache_key]
        
        # 计算前向传播 FLOPs
        input_shape = (batch_size, seq_size)
        
        try:
            forward_flops, macs, params = calculate_flops(
                model=self.model,
                input_shape=input_shape,
                transformer_tokenizer=transformer_tokenizer,
                include_backPropagation=False,
                print_results=False,
                output_as_string=False
            )
        except Exception as e:
            print(f"警告: 无法计算 FLOPs，使用估算值。错误: {e}")
            # 使用简单的估算（基于参数数量）
            total_params = sum(p.numel() for p in self.model.parameters())
            forward_flops = total_params * batch_size * seq_size * 2  # 简单估算
            macs = forward_flops / 2
        
        # 计算反向传播 FLOPs
        backward_flops = forward_flops * self.backward_factor
        total_flops = forward_flops + backward_flops
        
        result = {
            'forward_flops': forward_flops,
            'backward_flops': backward_flops,
            'total_flops': total_flops,
            'macs': macs
        }
        
        # 缓存结果
        self.cached_flops[cache_key] = result
        return result
    
    def start_epoch(self, epoch: int) -> None:
        """开始新的 epoch"""
        self.total_epochs = epoch
        self.current_epoch_steps = 0
        self.current_epoch_forward_flops = 0
        self.current_epoch_backward_flops = 0
        self.current_epoch_tokens = 0
        self.epoch_start_time = time.time()
        
        if self.start_time is None:
            self.start_time = time.time()
        
        print(f"\n🚀 开始 Epoch {epoch} - FLOPs 跟踪已启动")
        print("-" * 60)
    
    def track_training_step(self, 
                          batch_size: int,
                          seq_size: int,
                          transformer_tokenizer=None,
                          actual_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        跟踪单个训练步骤的 FLOPs
        
        Args:
            batch_size: 当前批次的大小
            seq_size: 序列长度
            transformer_tokenizer: 分词器（可选）
            actual_tokens: 实际处理的 token 数量（排除 padding）
            
        Returns:
            当前步骤的统计信息
        """
        # 计算当前批次的 FLOPs
        batch_flops = self._calculate_batch_flops(
            batch_size=batch_size,
            seq_size=seq_size,
            transformer_tokenizer=transformer_tokenizer
        )
        
        # 更新统计
        self.total_steps += 1
        self.current_epoch_steps += 1
        
        step_forward_flops = batch_flops['forward_flops']
        step_backward_flops = batch_flops['backward_flops']
        step_total_flops = batch_flops['total_flops']
        
        self.total_forward_flops += step_forward_flops
        self.total_backward_flops += step_backward_flops
        self.total_training_flops += step_total_flops
        
        self.current_epoch_forward_flops += step_forward_flops
        self.current_epoch_backward_flops += step_backward_flops
        
        # 计算 token 数量
        if actual_tokens is not None:
            step_tokens = actual_tokens
        else:
            step_tokens = batch_size * seq_size  # 假设没有 padding
        
        self.total_tokens_processed += step_tokens
        self.current_epoch_tokens += step_tokens
        
        # 准备返回的统计信息
        step_stats = {
            'step': self.total_steps,
            'epoch_step': self.current_epoch_steps,
            'batch_size': batch_size,
            'seq_size': seq_size,
            'step_tokens': step_tokens,
            'step_forward_flops': step_forward_flops,
            'step_backward_flops': step_backward_flops,
            'step_total_flops': step_total_flops,
            'total_tokens': self.total_tokens_processed,
            'total_flops': self.total_training_flops,
            'avg_flops_per_token': self.total_training_flops / max(self.total_tokens_processed, 1)
        }
        
        # 定期打印日志
        if self.current_epoch_steps % self.log_interval == 0:
            self._print_step_log(step_stats)
        
        return step_stats
    
    def _print_step_log(self, stats: Dict[str, Any]) -> None:
        """打印步骤日志"""
        print(f"[Step {stats['step']:6d}] "
              f"Tokens: {self._format_number(stats['step_tokens'])} (batch), "
              f"{self._format_number(stats['total_tokens'])} (total) | "
              f"FLOPs: {flops_to_string(stats['step_total_flops'])} (batch), "
              f"{flops_to_string(stats['total_flops'])} (total)")
    
    def end_epoch(self) -> Dict[str, Any]:
        """结束当前 epoch 并返回统计信息"""
        epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        epoch_summary = {
            'epoch': self.total_epochs,
            'steps': self.current_epoch_steps,
            'duration_seconds': epoch_duration,
            'tokens': self.current_epoch_tokens,
            'forward_flops': self.current_epoch_forward_flops,
            'backward_flops': self.current_epoch_backward_flops,
            'total_flops': self.current_epoch_forward_flops + self.current_epoch_backward_flops,
            'tokens_per_second': self.current_epoch_tokens / max(epoch_duration, 1),
            'flops_per_second': (self.current_epoch_forward_flops + self.current_epoch_backward_flops) / max(epoch_duration, 1),
            'flops_per_token': (self.current_epoch_forward_flops + self.current_epoch_backward_flops) / max(self.current_epoch_tokens, 1)
        }
        
        # 添加到历史记录
        self.epoch_history.append(epoch_summary)
        
        # 打印 epoch 总结
        self._print_epoch_summary(epoch_summary)
        
        return epoch_summary
    
    def _print_epoch_summary(self, summary: Dict[str, Any]) -> None:
        """打印 epoch 总结"""
        print(f"\n📊 Epoch {summary['epoch']} 总结:")
        print(f"  步数: {summary['steps']:,}")
        print(f"  持续时间: {summary['duration_seconds']:.1f} 秒")
        print(f"  处理 tokens: {self._format_number(summary['tokens'])}")
        print(f"  前向传播 FLOPs: {flops_to_string(summary['forward_flops'])}")
        print(f"  反向传播 FLOPs: {flops_to_string(summary['backward_flops'])}")
        print(f"  总 FLOPs: {flops_to_string(summary['total_flops'])}")
        print(f"  吞吐量: {self._format_number(summary['tokens_per_second']):.1f} tokens/sec")
        print(f"  计算效率: {flops_to_string(summary['flops_per_second'])}/sec")
        print(f"  FLOPs/Token: {summary['flops_per_token']:.2f}")
        print("-" * 60)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取整个训练过程的总结"""
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        return {
            'model_name': self.model_name,
            'total_epochs': self.total_epochs,
            'total_steps': self.total_steps,
            'total_duration_seconds': total_duration,
            'total_tokens': self.total_tokens_processed,
            'total_forward_flops': self.total_forward_flops,
            'total_backward_flops': self.total_backward_flops,
            'total_training_flops': self.total_training_flops,
            'avg_tokens_per_second': self.total_tokens_processed / max(total_duration, 1),
            'avg_flops_per_second': self.total_training_flops / max(total_duration, 1),
            'avg_flops_per_token': self.total_training_flops / max(self.total_tokens_processed, 1),
            'backward_factor': self.backward_factor,
            'epoch_history': self.epoch_history
        }
    
    def get_flops_formulas(self, 
                          reference_batch_size: int = 8,
                          reference_seq_size: int = 512) -> Dict[str, str]:
        """
        获取与 batch_size 和 seq_size 相关的 FLOPs 计算公式
        
        Args:
            reference_batch_size: 参考批量大小
            reference_seq_size: 参考序列长度
            
        Returns:
            包含公式的字典
        """
        # 使用参考值计算基准 FLOPs
        cache_key = self._get_cache_key(reference_batch_size, reference_seq_size)
        
        if cache_key not in self.cached_flops:
            # 计算参考值的 FLOPs
            self._calculate_batch_flops(reference_batch_size, reference_seq_size)
        
        ref_flops = self.cached_flops[cache_key]
        
        # 提取单个 token 的平均 FLOPs
        flops_per_token = ref_flops['total_flops'] / (reference_batch_size * reference_seq_size)
        forward_flops_per_token = ref_flops['forward_flops'] / (reference_batch_size * reference_seq_size)
        
        formulas = {
            'forward_flops_per_batch': f"Forward_FLOPs(B, L) ≈ {forward_flops_per_token:.2e} * B * L",
            'backward_flops_per_batch': f"Backward_FLOPs(B, L) ≈ {self.backward_factor} * Forward_FLOPs(B, L)",
            'total_flops_per_batch': f"Total_FLOPs(B, L) = Forward_FLOPs(B, L) + Backward_FLOPs(B, L) ≈ {flops_per_token:.2e} * B * L",
            'epoch_flops': f"Epoch_FLOPs(B, L, N) = Total_FLOPs(B, L) * (N / B) = {flops_per_token:.2e} * L * N",
            'training_flops': f"Training_FLOPs(B, L, N, E) = Epoch_FLOPs(B, L, N) * E",
            'explanation': f"其中 B=batch_size, L=seq_size, N=epoch_samples, E=num_epochs"
        }
        
        return formulas
    
    def _get_cache_key(self, batch_size: int, seq_size: int) -> str:
        """生成缓存键"""
        return f"{batch_size}_{seq_size}"
    
    def _calculate_batch_flops(self, 
                              batch_size: int, 
                              seq_size: int,
                              transformer_tokenizer=None) -> Dict[str, float]:
        """计算批次 FLOPs（内部方法）"""
        cache_key = self._get_cache_key(batch_size, seq_size)
        
        if cache_key in self.cached_flops:
            return self.cached_flops[cache_key]
        
        input_shape = (batch_size, seq_size)
        
        try:
            forward_flops, macs, params = calculate_flops(
                model=self.model,
                input_shape=input_shape,
                transformer_tokenizer=transformer_tokenizer,
                include_backPropagation=False,
                print_results=False,
                output_as_string=False
            )
        except Exception as e:
            print(f"警告: FLOPs 计算失败，使用估算值。错误: {e}")
            # 简单估算
            total_params = sum(p.numel() for p in self.model.parameters())
            forward_flops = total_params * batch_size * seq_size * 2
            macs = forward_flops / 2
        
        backward_flops = forward_flops * self.backward_factor
        total_flops = forward_flops + backward_flops
        
        result = {
            'forward_flops': forward_flops,
            'backward_flops': backward_flops,
            'total_flops': total_flops,
            'macs': macs
        }
        
        self.cached_flops[cache_key] = result
        return result
    
    def track_training_step(self,
                          batch_size: int,
                          seq_size: int,
                          transformer_tokenizer=None,
                          actual_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        跟踪单个训练步骤
        
        Args:
            batch_size: 批量大小
            seq_size: 序列长度
            transformer_tokenizer: 分词器（可选）
            actual_tokens: 实际处理的 token 数量（排除 padding）
            
        Returns:
            步骤统计信息
        """
        # 计算当前步骤的 FLOPs
        batch_flops = self._calculate_batch_flops(
            batch_size=batch_size,
            seq_size=seq_size,
            transformer_tokenizer=transformer_tokenizer
        )
        
        # 更新计数器
        self.total_steps += 1
        self.current_epoch_steps += 1
        
        step_forward_flops = batch_flops['forward_flops']
        step_backward_flops = batch_flops['backward_flops']
        step_total_flops = batch_flops['total_flops']
        
        self.total_forward_flops += step_forward_flops
        self.total_backward_flops += step_backward_flops
        self.total_training_flops += step_total_flops
        
        self.current_epoch_forward_flops += step_forward_flops
        self.current_epoch_backward_flops += step_backward_flops
        
        # 计算 token 数量
        if actual_tokens is not None:
            step_tokens = actual_tokens
        else:
            step_tokens = batch_size * seq_size
        
        self.total_tokens_processed += step_tokens
        self.current_epoch_tokens += step_tokens
        
        # 准备统计信息
        stats = {
            'global_step': self.total_steps,
            'epoch_step': self.current_epoch_steps,
            'batch_size': batch_size,
            'seq_size': seq_size,
            'step_tokens': step_tokens,
            'step_forward_flops': step_forward_flops,
            'step_backward_flops': step_backward_flops,
            'step_total_flops': step_total_flops,
            'total_tokens': self.total_tokens_processed,
            'total_flops': self.total_training_flops,
            'epoch_tokens': self.current_epoch_tokens,
            'epoch_flops': self.current_epoch_forward_flops + self.current_epoch_backward_flops
        }
        
        # 定期打印日志
        if self.current_epoch_steps % self.log_interval == 0:
            self._print_step_log(stats)
        
        return stats
    
    def _print_step_log(self, stats: Dict[str, Any]) -> None:
        """打印步骤日志"""
        print(f"[Step {stats['global_step']:6d}] "
              f"Tokens: {self._format_number(stats['step_tokens'])} (batch), "
              f"{self._format_number(stats['total_tokens'])} (total) | "
              f"FLOPs: {flops_to_string(stats['step_total_flops'])} (batch), "
              f"{flops_to_string(stats['total_flops'])} (total)")
    
    def end_epoch(self) -> Dict[str, Any]:
        """结束当前 epoch"""
        epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        epoch_summary = {
            'epoch': self.total_epochs,
            'steps': self.current_epoch_steps,
            'duration_seconds': epoch_duration,
            'tokens': self.current_epoch_tokens,
            'forward_flops': self.current_epoch_forward_flops,
            'backward_flops': self.current_epoch_backward_flops,
            'total_flops': self.current_epoch_forward_flops + self.current_epoch_backward_flops,
            'tokens_per_second': self.current_epoch_tokens / max(epoch_duration, 1),
            'flops_per_second': (self.current_epoch_forward_flops + self.current_epoch_backward_flops) / max(epoch_duration, 1),
            'flops_per_token': (self.current_epoch_forward_flops + self.current_epoch_backward_flops) / max(self.current_epoch_tokens, 1)
        }
        
        self.epoch_history.append(epoch_summary)
        self._print_epoch_summary(epoch_summary)
        
        return epoch_summary
    
    def _print_epoch_summary(self, summary: Dict[str, Any]) -> None:
        """打印 epoch 总结"""
        print(f"\n📊 Epoch {summary['epoch']} 总结:")
        print(f"  训练步数: {summary['steps']:,}")
        print(f"  持续时间: {summary['duration_seconds']:.1f} 秒")
        print(f"  处理 tokens: {self._format_number(summary['tokens'])}")
        print(f"  前向传播 FLOPs: {flops_to_string(summary['forward_flops'])}")
        print(f"  反向传播 FLOPs: {flops_to_string(summary['backward_flops'])}")
        print(f"  总 FLOPs: {flops_to_string(summary['total_flops'])}")
        print(f"  吞吐量: {self._format_number(summary['tokens_per_second']):.1f} tokens/sec")
        print(f"  计算效率: {flops_to_string(summary['flops_per_second'])}/sec")
        print(f"  FLOPs/Token: {summary['flops_per_token']:.2f}")
        print("-" * 60)
    
    def print_final_summary(self) -> None:
        """打印最终训练总结"""
        summary = self.get_training_summary()
        
        print("\n" + "=" * 80)
        print(f"🏁 训练总结 - {summary['model_name']}")
        print("=" * 80)
        
        print(f"总 epochs: {summary['total_epochs']}")
        print(f"总步数: {summary['total_steps']:,}")
        print(f"总训练时间: {summary['total_duration_seconds']:.1f} 秒")
        print(f"总处理 tokens: {self._format_number(summary['total_tokens'])}")
        print(f"总前向传播 FLOPs: {flops_to_string(summary['total_forward_flops'])}")
        print(f"总反向传播 FLOPs: {flops_to_string(summary['total_backward_flops'])}")
        print(f"总训练 FLOPs: {flops_to_string(summary['total_training_flops'])}")
        print(f"平均吞吐量: {self._format_number(summary['avg_tokens_per_second']):.1f} tokens/sec")
        print(f"平均计算效率: {flops_to_string(summary['avg_flops_per_second'])}/sec")
        print(f"平均 FLOPs/Token: {summary['avg_flops_per_token']:.2f}")
        
        print(f"\n📐 计算公式:")
        formulas = self.get_flops_formulas()
        for key, formula in formulas.items():
            if key != 'explanation':
                print(f"  {formula}")
        print(f"  {formulas['explanation']}")
        
        print("=" * 80)
    
    def save_results(self, filepath: str) -> None:
        """保存结果到文件"""
        results = {
            'training_summary': self.get_training_summary(),
            'epoch_history': self.epoch_history,
            'formulas': self.get_flops_formulas(),
            'cached_flops': self.cached_flops
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 结果已保存到: {filepath}")
    
    def _format_number(self, num: float) -> str:
        """格式化数字显示"""
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


def create_demo_transformer(vocab_size: int = 32000,
                          hidden_size: int = 512,
                          num_layers: int = 6,
                          num_heads: int = 8) -> nn.Module:
    """创建演示用的 Transformer 模型"""
    
    class DemoTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.pos_embedding = nn.Embedding(2048, hidden_size)  # 最大序列长度 2048
            
            # Transformer 层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=4 * hidden_size,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # 输出投影
            self.output_projection = nn.Linear(hidden_size, vocab_size)
            
            # 添加配置信息（用于理论计算）
            self.config = type('Config', (), {
                'vocab_size': vocab_size,
                'hidden_size': hidden_size,
                'num_hidden_layers': num_layers,
                'num_attention_heads': num_heads,
                'intermediate_size': 4 * hidden_size
            })()
        
        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            
            # Token embeddings
            token_emb = self.embedding(input_ids)
            
            # Position embeddings
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            
            # 组合 embeddings
            embeddings = token_emb + pos_emb
            
            # Transformer 处理
            hidden_states = self.transformer(embeddings)
            
            # 输出投影
            logits = self.output_projection(hidden_states)
            
            return logits
    
    return DemoTransformer()


def demo_pretrain_tracking():
    """演示预训练 FLOPs 跟踪"""
    print("🚀 预训练 FLOPs 跟踪演示")
    print("=" * 60)
    
    # 创建演示模型
    model = create_demo_transformer(
        vocab_size=32000,
        hidden_size=512,
        num_layers=6,
        num_heads=8
    )
    
    # 创建 FLOPs 跟踪器
    tracker = PretrainFLOPsTracker(
        model=model,
        model_name="Demo Transformer (512d, 6L, 8H)",
        backward_factor=2.0,
        log_interval=50
    )
    
    # 模拟训练参数
    batch_size = 8
    seq_size = 512
    samples_per_epoch = 1000
    num_epochs = 3
    
    print(f"📋 训练配置:")
    print(f"  模型: {tracker.model_name}")
    print(f"  批量大小: {batch_size}")
    print(f"  序列长度: {seq_size}")
    print(f"  每个 epoch 样本数: {samples_per_epoch:,}")
    print(f"  总 epochs: {num_epochs}")
    
    # 模拟训练循环
    for epoch in range(1, num_epochs + 1):
        tracker.start_epoch(epoch)
        
        # 计算每个 epoch 的批次数
        batches_per_epoch = math.ceil(samples_per_epoch / batch_size)
        
        # 模拟训练步骤
        for step in range(batches_per_epoch):
            # 模拟可变的 batch_size（最后一个批次可能较小）
            current_batch_size = min(batch_size, samples_per_epoch - step * batch_size)
            if current_batch_size <= 0:
                break
                
            # 跟踪训练步骤
            step_stats = tracker.track_training_step(
                batch_size=current_batch_size,
                seq_size=seq_size,
                transformer_tokenizer=None,
                actual_tokens=current_batch_size * seq_size  # 假设没有 padding
            )
            
            # 模拟训练时间（可选）
            time.sleep(0.01)  # 模拟训练延迟
        
        # 结束 epoch
        epoch_summary = tracker.end_epoch()
    
    # 打印最终总结
    tracker.print_final_summary()
    
    # 保存结果
    tracker.save_results('/workspace/pretrain_flops_results.json')
    
    return tracker


def analyze_flops_scaling():
    """分析 FLOPs 随 batch_size 和 seq_size 的变化"""
    print("\n🔍 FLOPs 缩放分析")
    print("=" * 60)
    
    # 创建小模型用于快速测试
    model = create_demo_transformer(
        vocab_size=10000,
        hidden_size=256,
        num_layers=3,
        num_heads=4
    )
    
    tracker = PretrainFLOPsTracker(model, "Small Transformer")
    
    # 测试不同的配置
    test_configs = [
        (4, 256),    # 小批次，短序列
        (8, 512),    # 中等批次，中等序列
        (16, 1024),  # 大批次，长序列
        (32, 2048),  # 很大批次，很长序列
    ]
    
    print(f"{'Batch Size':<12} {'Seq Size':<10} {'Forward FLOPs':<15} {'Backward FLOPs':<15} {'Total FLOPs':<15} {'FLOPs/Token':<12}")
    print("-" * 90)
    
    for batch_size, seq_size in test_configs:
        batch_flops = tracker._calculate_batch_flops(batch_size, seq_size)
        
        forward_str = flops_to_string(batch_flops['forward_flops'])
        backward_str = flops_to_string(batch_flops['backward_flops'])
        total_str = flops_to_string(batch_flops['total_flops'])
        flops_per_token = batch_flops['total_flops'] / (batch_size * seq_size)
        
        print(f"{batch_size:<12} {seq_size:<10} {forward_str:<15} {backward_str:<15} {total_str:<15} {flops_per_token:<12.2f}")
    
    # 显示公式
    print(f"\n📐 计算公式:")
    formulas = tracker.get_flops_formulas(reference_batch_size=8, reference_seq_size=512)
    for key, formula in formulas.items():
        print(f"  {formula}")


if __name__ == "__main__":
    # 运行演示
    demo_pretrain_tracking()
    
    # 运行缩放分析
    analyze_flops_scaling()