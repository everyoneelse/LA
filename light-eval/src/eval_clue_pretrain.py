#!/usr/bin/env python3
"""
CLUE Benchmark预训练模型评估脚本
专门针对预训练模型优化，只评估适合的任务
支持zero-shot和few-shot评估
"""

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import sys
import random
from typing import List, Dict, Optional

# 添加accessory路径
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel
from util import misc
from fairscale.nn.model_parallel import initialize as fs_init
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize

# 导入CLUE预训练任务
from clue.clue_pretrain_tasks import (
    get_pretrain_task, 
    PRETRAIN_TASK_REGISTRY,
    TASK_RECOMMENDATIONS,
    get_recommended_tasks
)


def get_args_parser():
    parser = argparse.ArgumentParser('CLUE Pretrain Evaluation', add_help=False)
    
    # 数据集参数
    parser.add_argument('--data_dir', type=str, default='data/clue',
                        help='CLUE数据集目录')
    parser.add_argument('--tasks', type=str, nargs='+', default=['recommended'],
                        help='要评估的任务列表，可选: recommended, all, easy, medium, hard, 或具体任务名')
    parser.add_argument('--evaluation_mode', type=str, default='few-shot',
                        choices=['zero-shot', 'few-shot', 'both'],
                        help='评估模式')
    parser.add_argument('--num_shots', type=int, default=None,
                        help='Few-shot示例数量，None表示使用推荐值')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help='每个任务最大评估样本数（用于快速测试）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="覆盖已有结果")
    
    # 模型参数
    parser.add_argument('--llama_type', default='llama', type=str,
                        help='LLaMA模型类型')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                        help='LLaMA模型配置文件路径')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='分词器路径')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='预训练模型路径')
    parser.add_argument('--pretrained_type', type=str, default="consolidated",
                        choices=['consolidated', 'meta_ori'],
                        help='预训练模型格式')
    parser.add_argument('--max_seq_len', default=2048, type=int,
                        help='最大输入序列长度')
    parser.add_argument('--quant', action='store_true', default=False,
                        help='是否使用4bit量化')
    
    # 生成参数
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p采样参数')
    parser.add_argument('--max_gen_len', type=int, default=128,
                        help='最大生成长度')
    
    # 并行参数
    parser.add_argument('--device', default='cuda',
                        help='推理设备')
    parser.add_argument('--model_parallel_size', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='分布式进程数')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='分布式训练URL')
    
    return parser


def load_model(args):
    """加载模型和分词器"""
    # 初始化分布式
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    
    # 创建模型
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=False)
    print(f"加载预训练模型: {args.pretrained_path}")
    load_tensor_parallel_model_list(model, args.pretrained_path)
    
    # 量化
    if args.quant:
        print("量化模型到4bit...")
        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            },
            return_unused_kwargs=False,
        )
        quantize(model, quantization_config)
    
    model.bfloat16().cuda()
    model.eval()
    
    return model


def select_demonstrations(task, num_shots: int, seed: int = 42) -> List[Dict]:
    """选择few-shot示例"""
    if not task.train_data or num_shots == 0:
        return []
    
    random.seed(seed)
    
    # 尝试平衡选择各类别的示例
    demonstrations = []
    
    # 按标签分组
    label_groups = {}
    for example in task.train_data:
        label = example.get('label', 'unknown')
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(example)
    
    # 每个类别选择示例
    shots_per_label = max(1, num_shots // len(label_groups))
    remaining_shots = num_shots
    
    for label, examples in label_groups.items():
        n = min(shots_per_label, len(examples), remaining_shots)
        if n > 0:
            selected = random.sample(examples, n)
            demonstrations.extend(selected)
            remaining_shots -= n
    
    # 如果还需要更多示例，随机补充
    if remaining_shots > 0:
        all_examples = [e for e in task.train_data if e not in demonstrations]
        if all_examples:
            additional = random.sample(all_examples, min(remaining_shots, len(all_examples)))
            demonstrations.extend(additional)
    
    # 打乱顺序
    random.shuffle(demonstrations)
    
    return demonstrations[:num_shots]


def evaluate_task(model, task, args, mode: str = 'few-shot', num_shots: Optional[int] = None):
    """评估单个任务"""
    task_name = task.task_name
    
    # 确定shot数量
    if mode == 'zero-shot':
        actual_shots = 0
    elif num_shots is not None:
        actual_shots = num_shots
    else:
        # 使用推荐值
        actual_shots = TASK_RECOMMENDATIONS.get(task_name, {}).get('recommended_shots', 3)
    
    print(f"\n评估任务: {task_name} (模式: {mode}, shots: {actual_shots})")
    
    # 加载数据
    task.load_data()
    
    if not task.dev_data:
        print(f"警告: {task_name} 没有验证集数据")
        return None
    
    # 选择示例
    demonstrations = select_demonstrations(task, actual_shots, args.seed) if actual_shots > 0 else []
    
    # 评估样本
    eval_data = task.dev_data
    if args.max_eval_samples:
        eval_data = eval_data[:args.max_eval_samples]
    
    predictions = []
    references = []
    
    for example in tqdm(eval_data, desc=f"评估 {task_name}"):
        # 构建输入
        if mode == 'zero-shot':
            prompt = task.format_zero_shot_prompt(example)
        else:
            prompt = task.format_few_shot_prompt(example, demonstrations)
        
        # 截断到最大长度
        tokens = model.tokenizer.encode(prompt, bos=True, eos=False)
        if len(tokens) > args.max_seq_len - args.max_gen_len:
            # 保留开头的任务描述和结尾的问题
            keep_start = 100
            keep_end = args.max_seq_len - args.max_gen_len - keep_start
            tokens = tokens[:keep_start] + tokens[-keep_end:]
            prompt = model.tokenizer.decode(tokens)
        
        # 生成预测
        with torch.no_grad():
            output = model.generate(
                prompts=[prompt],
                images=None,
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p,
            )[0]
        
        # 提取答案
        generated_text = output[len(prompt):]
        prediction = task.extract_answer(generated_text)
        
        # 获取真实标签
        if 'label' in example:
            reference = str(example['label'])
        else:
            continue
        
        predictions.append(prediction)
        references.append(reference)
    
    # 计算指标
    metrics = task.compute_metric(predictions, references)
    metrics['num_samples'] = len(predictions)
    metrics['mode'] = mode
    metrics['num_shots'] = actual_shots
    
    return metrics


def determine_tasks(args) -> List[str]:
    """确定要评估的任务列表"""
    if 'recommended' in args.tasks:
        return get_recommended_tasks()
    elif 'all' in args.tasks:
        return list(PRETRAIN_TASK_REGISTRY.keys())
    elif 'easy' in args.tasks:
        return get_recommended_tasks('easy')
    elif 'medium' in args.tasks:
        return get_recommended_tasks('medium')
    elif 'hard' in args.tasks:
        return get_recommended_tasks('hard')
    else:
        # 验证任务名称
        valid_tasks = []
        for task in args.tasks:
            if task in PRETRAIN_TASK_REGISTRY:
                valid_tasks.append(task)
            else:
                print(f"警告: 未知任务 {task}")
        return valid_tasks


def evaluate_clue_pretrain(model, args):
    """评估预训练模型在CLUE任务上的表现"""
    results = {}
    
    # 确定要评估的任务
    tasks_to_eval = determine_tasks(args)
    
    if not tasks_to_eval:
        print("错误: 没有有效的任务可评估")
        return results
    
    print(f"将评估以下任务: {tasks_to_eval}")
    
    # 评估每个任务
    for task_name in tasks_to_eval:
        try:
            # 创建任务实例
            task = get_pretrain_task(task_name, args.data_dir)
            
            task_results = {}
            
            # Zero-shot评估
            if args.evaluation_mode in ['zero-shot', 'both']:
                # 检查任务是否支持zero-shot
                if TASK_RECOMMENDATIONS[task_name]['zero_shot_capable']:
                    metrics = evaluate_task(model, task, args, mode='zero-shot')
                    if metrics:
                        task_results['zero_shot'] = metrics
                        print(f"{task_name} Zero-shot结果: {metrics['accuracy']:.4f}")
                else:
                    print(f"{task_name} 不推荐zero-shot评估")
            
            # Few-shot评估
            if args.evaluation_mode in ['few-shot', 'both']:
                metrics = evaluate_task(model, task, args, mode='few-shot', num_shots=args.num_shots)
                if metrics:
                    task_results['few_shot'] = metrics
                    print(f"{task_name} Few-shot结果: {metrics['accuracy']:.4f}")
            
            if task_results:
                results[task_name] = task_results
        
        except Exception as e:
            print(f"评估 {task_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def compute_summary(results: Dict) -> Dict:
    """计算汇总统计"""
    summary = {
        'task_scores': {},
        'average_scores': {},
        'task_count': len(results)
    }
    
    # 收集各模式的分数
    zero_shot_scores = []
    few_shot_scores = []
    
    for task_name, task_results in results.items():
        task_summary = {}
        
        if 'zero_shot' in task_results:
            score = task_results['zero_shot']['accuracy']
            task_summary['zero_shot'] = score
            zero_shot_scores.append(score)
        
        if 'few_shot' in task_results:
            score = task_results['few_shot']['accuracy']
            task_summary['few_shot'] = score
            few_shot_scores.append(score)
        
        # 最佳分数
        task_summary['best'] = max(task_summary.values())
        summary['task_scores'][task_name] = task_summary
    
    # 计算平均分
    if zero_shot_scores:
        summary['average_scores']['zero_shot'] = np.mean(zero_shot_scores)
    
    if few_shot_scores:
        summary['average_scores']['few_shot'] = np.mean(few_shot_scores)
    
    if summary['average_scores']:
        summary['average_scores']['best'] = max(summary['average_scores'].values())
    
    return summary


def print_results(results: Dict, summary: Dict):
    """打印评估结果"""
    print("\n" + "="*60)
    print("CLUE预训练模型评估结果")
    print("="*60)
    
    # 打印每个任务的结果
    for task_name, task_results in results.items():
        print(f"\n{task_name}:")
        print(f"  任务描述: {TASK_RECOMMENDATIONS[task_name]['description']}")
        print(f"  难度: {TASK_RECOMMENDATIONS[task_name]['difficulty']}")
        
        if 'zero_shot' in task_results:
            print(f"  Zero-shot准确率: {task_results['zero_shot']['accuracy']:.4f}")
        
        if 'few_shot' in task_results:
            shots = task_results['few_shot']['num_shots']
            print(f"  Few-shot准确率 ({shots} shots): {task_results['few_shot']['accuracy']:.4f}")
    
    # 打印汇总
    print("\n" + "-"*60)
    print("汇总统计:")
    
    if 'zero_shot' in summary['average_scores']:
        print(f"  Zero-shot平均准确率: {summary['average_scores']['zero_shot']:.4f}")
    
    if 'few_shot' in summary['average_scores']:
        print(f"  Few-shot平均准确率: {summary['average_scores']['few_shot']:.4f}")
    
    if 'best' in summary['average_scores']:
        print(f"  最佳平均准确率: {summary['average_scores']['best']:.4f}")
    
    print("="*60)


def main(args):
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建结果目录
    path_split = args.pretrained_path.split('/')
    if path_split[-1] == '':
        path_split.pop(-1)
    model_name = path_split[-1]
    
    result_dir = Path('results') / model_name / 'clue_pretrain'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = result_dir / f'evaluation_results_{args.evaluation_mode}.json'
    
    # 检查是否已有结果
    if not args.overwrite and result_path.exists():
        print(f"结果文件已存在: {result_path}")
        with open(result_path, 'r') as f:
            saved_results = json.load(f)
        
        if 'summary' in saved_results:
            print_results(saved_results['results'], saved_results['summary'])
        return
    
    # 加载模型
    model = load_model(args)
    
    # 评估
    results = evaluate_clue_pretrain(model, args)
    
    # 计算汇总
    summary = compute_summary(results)
    
    # 保存结果
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        save_data = {
            'results': results,
            'summary': summary,
            'config': {
                'tasks': determine_tasks(args),
                'evaluation_mode': args.evaluation_mode,
                'num_shots': args.num_shots,
                'seed': args.seed,
                'model': model_name
            }
        }
        
        with open(result_path, 'w') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print_results(results, summary)
        print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)