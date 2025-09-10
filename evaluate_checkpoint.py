#!/usr/bin/env python3
"""
独立的checkpoint评估脚本
用于加载已保存的模型checkpoint并在新的验证数据集上评估loss

使用方法:
python evaluate_checkpoint.py \
    --checkpoint_path /path/to/checkpoint/dir \
    --val_data_meta_path /path/to/validation/meta.json \
    --val_data_root /path/to/validation/data \
    --tokenizer_path /path/to/tokenizer.model \
    --batch_size 4 \
    --max_words 2048 \
    --packed_data
"""

import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 1)[0])

import argparse
import contextlib
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import accessory.util.misc as misc
from accessory.util.tensor_type import default_tensor_type
from accessory.model.meta import MetaModel
from accessory.data import falcon, falcon_packed


def get_args_parser():
    parser = argparse.ArgumentParser('Checkpoint Evaluator', add_help=False)
    
    # Model and checkpoint parameters
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='Path to checkpoint directory or file')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='Path to tokenizer.model')
    parser.add_argument('--llama_type', default=None, type=str,
                        help='LLaMA model type (optional, will be auto-detected from checkpoint)')
    parser.add_argument('--llama_config', default=[], nargs="*",
                        help='Path to llama model config (optional, will be auto-detected from checkpoint)')
    
    # Validation dataset parameters
    parser.add_argument('--val_data_meta_path', required=True, type=str,
                        help='Path to validation data meta file')
    parser.add_argument('--val_data_root', default=None, type=str,
                        help='Root path for validation data')
    parser.add_argument('--packed_data', action="store_true",
                        help='Use packed dataset format')
    parser.add_argument('--max_words', default=2048, type=int,
                        help='Max token length')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', default=5, type=int,
                        help='Number of data loader workers')
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'tf32'], default='bf16',
                        help='Evaluation precision')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for evaluation')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to evaluate (for quick testing)')
    
    # Output parameters
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save evaluation results (JSON format)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print detailed progress information')
    
    return parser


def load_model_from_checkpoint(checkpoint_path: str, tokenizer_path: str, 
                               llama_type: Optional[str] = None, 
                               llama_config: Optional[list] = None,
                               max_words: int = 2048, 
                               precision: str = 'bf16',
                               device: str = 'cuda') -> MetaModel:
    """从checkpoint加载模型"""
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # 设置数据类型
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "tf32": torch.float32,
    }
    dtype = dtype_map[precision]
    
    try:
        # 尝试使用MetaModel.from_pretrained方法加载
        model = MetaModel.from_pretrained(
            pretrained_path=[checkpoint_path],
            llama_type=llama_type,
            llama_config=llama_config,
            tokenizer_path=tokenizer_path,
            with_visual=False,
            max_seq_len=max_words,
            mp_group=None,
            dtype=dtype,
            device=device,
            quant=False,
        )
        print("Model loaded successfully using MetaModel.from_pretrained")
        
    except Exception as e:
        print(f"Failed to load with from_pretrained: {e}")
        print("Trying alternative loading method...")
        
        # 备用加载方法
        with default_tensor_type(dtype=dtype, device=device):
            model = MetaModel(
                llama_type or 'llama', 
                llama_config or [],
                tokenizer_path, 
                with_visual=False,
                max_seq_len=max_words
            )
        
        # 手动加载权重
        if os.path.isdir(checkpoint_path):
            # 查找checkpoint文件
            ckpt_files = []
            for file in os.listdir(checkpoint_path):
                if file.endswith('.pth') or file.endswith('.pt'):
                    ckpt_files.append(os.path.join(checkpoint_path, file))
            
            if not ckpt_files:
                raise ValueError(f"No checkpoint files found in {checkpoint_path}")
            
            # 加载第一个找到的checkpoint文件
            ckpt_path = ckpt_files[0]
            print(f"Loading checkpoint from: {ckpt_path}")
            
        else:
            ckpt_path = checkpoint_path
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 加载状态字典
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    model.eval()
    return model


def create_validation_dataset(val_data_meta_path: str, val_data_root: str,
                              tokenizer_path: str, max_words: int, 
                              packed_data: bool):
    """创建验证数据集"""
    
    if packed_data:
        dataset = falcon_packed.FalconVal(
            data_meta_path=None,  # 不使用训练数据
            data_root=None,
            tokenizer_path=tokenizer_path,
            max_words=max_words,
            val_data_meta_path=val_data_meta_path,
            val_data_root=val_data_root
        )
    else:
        dataset = falcon.FalconVal(
            data_meta_path=val_data_meta_path,
            data_root=val_data_root,
            tokenizer_path=tokenizer_path,
            max_words=max_words
        )
    
    return dataset


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, 
                   precision: str = 'bf16', max_batches: Optional[int] = None,
                   verbose: bool = False):
    """评估模型在验证集上的表现"""
    
    model.eval()
    
    # 设置自动混合精度上下文
    autocast_ctx = {
        "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
        "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
        "tf32": contextlib.nullcontext(),
    }[precision]
    
    total_loss = 0.0
    total_batches = 0
    losses = []
    
    print(f"Starting evaluation on {len(data_loader)} batches...")
    if max_batches is not None:
        print(f"Limited to {max_batches} batches for quick testing")
    
    for batch_idx, batch_data in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        # 处理不同的数据格式
        if len(batch_data) == 2:
            examples, labels = batch_data
        elif len(batch_data) == 3:
            examples, labels, _ = batch_data  # 忽略item_states
        else:
            raise ValueError(f"Unexpected batch data format: {len(batch_data)} elements")
        
        with autocast_ctx:
            c_loss, additional_loss_dict = model(examples, labels)
        
        # 计算总损失
        loss = c_loss
        for (add_loss, weight) in additional_loss_dict.values():
            loss = loss + add_loss * weight
        
        loss_value = loss.item()
        losses.append(loss_value)
        total_loss += loss_value
        total_batches += 1
        
        if verbose and (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / total_batches
            print(f"Batch {batch_idx + 1}/{len(data_loader)}: "
                  f"Current Loss: {loss_value:.6f}, Avg Loss: {avg_loss:.6f}")
    
    # 计算统计信息
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    std_loss = np.std(losses) if len(losses) > 1 else 0.0
    min_loss = min(losses) if losses else 0.0
    max_loss = max(losses) if losses else 0.0
    
    results = {
        'average_loss': avg_loss,
        'std_loss': std_loss,
        'min_loss': min_loss,
        'max_loss': max_loss,
        'total_batches': total_batches,
        'total_samples': total_batches * data_loader.batch_size,
        'losses': losses
    }
    
    return results


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("Checkpoint Evaluation Script")
    print("=" * 60)
    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Validation data: {args.val_data_meta_path}")
    print(f"Packed data: {args.packed_data}")
    print(f"Device: {args.device}")
    print(f"Precision: {args.precision}")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    cudnn.benchmark = True
    
    # 设置设备
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'
        else:
            torch.cuda.set_device(args.device)
    
    # 加载模型
    try:
        model = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            tokenizer_path=args.tokenizer_path,
            llama_type=args.llama_type,
            llama_config=args.llama_config,
            max_words=args.max_words,
            precision=args.precision,
            device=args.device
        )
        print(f"Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # 创建验证数据集
    try:
        dataset = create_validation_dataset(
            val_data_meta_path=args.val_data_meta_path,
            val_data_root=args.val_data_root,
            tokenizer_path=args.tokenizer_path,
            max_words=args.max_words,
            packed_data=args.packed_data
        )
        print(f"Validation dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"Error creating validation dataset: {e}")
        sys.exit(1)
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if args.device.startswith('cuda') else False,
        shuffle=False,
        drop_last=False,
    )
    
    # 评估模型
    try:
        results = evaluate_model(
            model=model,
            data_loader=data_loader,
            precision=args.precision,
            max_batches=args.max_batches,
            verbose=args.verbose
        )
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Average Loss: {results['average_loss']:.6f}")
        print(f"Loss Std Dev: {results['std_loss']:.6f}")
        print(f"Min Loss: {results['min_loss']:.6f}")
        print(f"Max Loss: {results['max_loss']:.6f}")
        print(f"Total Batches: {results['total_batches']}")
        print(f"Total Samples: {results['total_samples']}")
        print("=" * 60)
        
        # 保存结果到文件
        if args.output_file:
            output_data = {
                'checkpoint_path': args.checkpoint_path,
                'val_data_meta_path': args.val_data_meta_path,
                'val_data_root': args.val_data_root,
                'evaluation_params': {
                    'batch_size': args.batch_size,
                    'max_words': args.max_words,
                    'precision': args.precision,
                    'packed_data': args.packed_data,
                },
                'results': results
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
    
    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()