#!/usr/bin/env python3
"""
批量checkpoint评估脚本
用于评估输出目录中的所有checkpoint在新验证数据集上的性能

使用方法:
python batch_evaluate_checkpoints.py \
    --output_dir /path/to/training/output \
    --val_data_meta_path /path/to/validation/meta.json \
    --val_data_root /path/to/validation/data \
    --tokenizer_path /path/to/tokenizer.model \
    --batch_size 4 \
    --packed_data
"""

import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 1)[0])

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import subprocess

import pandas as pd
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('Batch Checkpoint Evaluator', add_help=False)
    
    # Input parameters
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Training output directory containing checkpoints')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='Path to tokenizer.model')
    
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
                        help='Maximum number of batches to evaluate per checkpoint')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory to save evaluation results (defaults to output_dir/evaluation_results)')
    parser.add_argument('--plot_results', action='store_true', default=False,
                        help='Generate loss curve plots')
    
    # Checkpoint filtering
    parser.add_argument('--min_iter', type=int, default=None,
                        help='Minimum iteration number to evaluate')
    parser.add_argument('--max_iter', type=int, default=None,
                        help='Maximum iteration number to evaluate')
    parser.add_argument('--iter_step', type=int, default=None,
                        help='Only evaluate checkpoints at specific iteration intervals')
    
    return parser


def parse_iter_from_ckpt_name(name: str) -> int:
    """从checkpoint名称中解析迭代次数"""
    match = re.search(r"iter(\d+)", name)
    if match:
        return int(match.group(1))
    # 尝试从epoch名称解析
    match = re.search(r"epoch(\d+)", name)
    if match:
        return int(match.group(1)) * 1000  # 粗略估计
    return 0


def find_checkpoint_dirs(output_dir: str) -> List[Dict[str, Any]]:
    """查找所有checkpoint目录"""
    checkpoint_dirs = []
    
    # 查找主输出目录中的checkpoint
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and (item.startswith("epoch") or "iter" in item):
            iter_num = parse_iter_from_ckpt_name(item)
            checkpoint_dirs.append({
                'path': item_path,
                'name': item,
                'iteration': iter_num,
                'type': 'main'
            })
    
    # 查找eval_snapshots目录中的checkpoint
    eval_snapshots_dir = os.path.join(output_dir, "eval_snapshots")
    if os.path.exists(eval_snapshots_dir):
        for item in os.listdir(eval_snapshots_dir):
            item_path = os.path.join(eval_snapshots_dir, item)
            if os.path.isdir(item_path) and (item.startswith("epoch") or "iter" in item):
                iter_num = parse_iter_from_ckpt_name(item)
                checkpoint_dirs.append({
                    'path': item_path,
                    'name': item,
                    'iteration': iter_num,
                    'type': 'snapshot'
                })
    
    # 按迭代次数排序
    checkpoint_dirs.sort(key=lambda x: x['iteration'])
    
    return checkpoint_dirs


def filter_checkpoints(checkpoint_dirs: List[Dict[str, Any]], 
                       min_iter: int = None, max_iter: int = None, 
                       iter_step: int = None) -> List[Dict[str, Any]]:
    """根据条件过滤checkpoint"""
    filtered = checkpoint_dirs
    
    if min_iter is not None:
        filtered = [ckpt for ckpt in filtered if ckpt['iteration'] >= min_iter]
    
    if max_iter is not None:
        filtered = [ckpt for ckpt in filtered if ckpt['iteration'] <= max_iter]
    
    if iter_step is not None:
        # 选择符合步长的checkpoint
        filtered = [ckpt for ckpt in filtered if ckpt['iteration'] % iter_step == 0]
    
    return filtered


def evaluate_single_checkpoint(checkpoint_info: Dict[str, Any], args) -> Dict[str, Any]:
    """评估单个checkpoint"""
    checkpoint_path = checkpoint_info['path']
    checkpoint_name = checkpoint_info['name']
    
    print(f"\nEvaluating checkpoint: {checkpoint_name}")
    print(f"Path: {checkpoint_path}")
    
    # 构建评估命令
    cmd = [
        sys.executable, "evaluate_checkpoint.py",
        "--checkpoint_path", checkpoint_path,
        "--val_data_meta_path", args.val_data_meta_path,
        "--tokenizer_path", args.tokenizer_path,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--precision", args.precision,
        "--device", args.device,
        "--max_words", str(args.max_words),
    ]
    
    if args.val_data_root:
        cmd.extend(["--val_data_root", args.val_data_root])
    
    if args.packed_data:
        cmd.append("--packed_data")
    
    if args.max_batches:
        cmd.extend(["--max_batches", str(args.max_batches)])
    
    # 设置输出文件
    results_file = os.path.join(args.results_dir, f"{checkpoint_name}_results.json")
    cmd.extend(["--output_file", results_file])
    
    try:
        # 运行评估
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            # 读取结果
            with open(results_file, 'r') as f:
                eval_results = json.load(f)
            
            # 添加checkpoint信息
            eval_results['checkpoint_info'] = checkpoint_info
            
            print(f"✓ Evaluation completed. Average loss: {eval_results['results']['average_loss']:.6f}")
            return eval_results
        else:
            print(f"✗ Evaluation failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"✗ Evaluation failed with exception: {e}")
        return None


def save_summary_results(all_results: List[Dict[str, Any]], results_dir: str):
    """保存汇总结果"""
    
    # 创建汇总数据
    summary_data = []
    for result in all_results:
        if result is not None:
            checkpoint_info = result['checkpoint_info']
            eval_results = result['results']
            
            summary_data.append({
                'checkpoint_name': checkpoint_info['name'],
                'iteration': checkpoint_info['iteration'],
                'checkpoint_type': checkpoint_info['type'],
                'average_loss': eval_results['average_loss'],
                'std_loss': eval_results['std_loss'],
                'min_loss': eval_results['min_loss'],
                'max_loss': eval_results['max_loss'],
                'total_batches': eval_results['total_batches'],
                'total_samples': eval_results['total_samples'],
            })
    
    # 保存为CSV
    df = pd.DataFrame(summary_data)
    csv_file = os.path.join(results_dir, "evaluation_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"Summary results saved to: {csv_file}")
    
    # 保存为JSON
    json_file = os.path.join(results_dir, "evaluation_summary.json")
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary results saved to: {json_file}")
    
    return df


def plot_loss_curves(df: pd.DataFrame, results_dir: str):
    """绘制loss曲线"""
    try:
        import matplotlib.pyplot as plt
        
        # 按迭代次数排序
        df_sorted = df.sort_values('iteration')
        
        plt.figure(figsize=(12, 8))
        
        # 绘制平均loss
        plt.subplot(2, 1, 1)
        plt.plot(df_sorted['iteration'], df_sorted['average_loss'], 'b-o', label='Average Loss')
        plt.fill_between(df_sorted['iteration'], 
                         df_sorted['average_loss'] - df_sorted['std_loss'],
                         df_sorted['average_loss'] + df_sorted['std_loss'],
                         alpha=0.3, color='blue', label='±1 Std Dev')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Validation Loss vs Iteration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制min/max loss
        plt.subplot(2, 1, 2)
        plt.plot(df_sorted['iteration'], df_sorted['min_loss'], 'g-o', label='Min Loss')
        plt.plot(df_sorted['iteration'], df_sorted['max_loss'], 'r-o', label='Max Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Min/Max Validation Loss vs Iteration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(results_dir, "loss_curves.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to: {plot_file}")
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"Error generating plots: {e}")


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("=" * 80)
    print("Batch Checkpoint Evaluation Script")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Validation data: {args.val_data_meta_path}")
    print(f"Packed data: {args.packed_data}")
    
    # 设置结果目录
    if args.results_dir is None:
        args.results_dir = os.path.join(args.output_dir, "evaluation_results")
    
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Results directory: {args.results_dir}")
    
    # 查找checkpoint
    print("\nSearching for checkpoints...")
    checkpoint_dirs = find_checkpoint_dirs(args.output_dir)
    
    if not checkpoint_dirs:
        print("No checkpoints found!")
        sys.exit(1)
    
    print(f"Found {len(checkpoint_dirs)} checkpoints")
    
    # 过滤checkpoint
    filtered_checkpoints = filter_checkpoints(
        checkpoint_dirs, 
        args.min_iter, 
        args.max_iter, 
        args.iter_step
    )
    
    print(f"After filtering: {len(filtered_checkpoints)} checkpoints to evaluate")
    
    if not filtered_checkpoints:
        print("No checkpoints match the filtering criteria!")
        sys.exit(1)
    
    # 显示将要评估的checkpoint
    print("\nCheckpoints to evaluate:")
    for ckpt in filtered_checkpoints:
        print(f"  - {ckpt['name']} (iter {ckpt['iteration']}, type: {ckpt['type']})")
    
    # 开始批量评估
    print("\n" + "=" * 80)
    print("Starting batch evaluation...")
    print("=" * 80)
    
    all_results = []
    
    for i, checkpoint_info in enumerate(filtered_checkpoints, 1):
        print(f"\n[{i}/{len(filtered_checkpoints)}] ", end="")
        result = evaluate_single_checkpoint(checkpoint_info, args)
        all_results.append(result)
    
    # 过滤成功的结果
    successful_results = [r for r in all_results if r is not None]
    failed_count = len(all_results) - len(successful_results)
    
    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Total checkpoints: {len(filtered_checkpoints)}")
    print(f"Successful evaluations: {len(successful_results)}")
    print(f"Failed evaluations: {failed_count}")
    
    if successful_results:
        # 保存汇总结果
        print("\nSaving summary results...")
        df = save_summary_results(successful_results, args.results_dir)
        
        # 显示最佳结果
        best_result = df.loc[df['average_loss'].idxmin()]
        print(f"\nBest checkpoint:")
        print(f"  Name: {best_result['checkpoint_name']}")
        print(f"  Iteration: {best_result['iteration']}")
        print(f"  Average Loss: {best_result['average_loss']:.6f}")
        
        # 生成图表
        if args.plot_results:
            print("\nGenerating loss curve plots...")
            plot_loss_curves(df, args.results_dir)
    
    print(f"\nAll results saved in: {args.results_dir}")


if __name__ == '__main__':
    main()