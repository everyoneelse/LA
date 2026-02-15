#!/usr/bin/env python3
"""
CLUE Benchmark评估脚本
用于评估预训练模型在CLUE各项任务上的表现
"""

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import sys

# 添加accessory路径
sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel
from util import misc
from fairscale.nn.model_parallel import initialize as fs_init
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize

# 导入CLUE任务
from clue.clue_tasks import get_task, TASK_REGISTRY


def get_args_parser():
    parser = argparse.ArgumentParser('CLUE Evaluation', add_help=False)
    
    # 数据集参数
    parser.add_argument('--data_dir', type=str, default='data/clue',
                        help='CLUE数据集目录')
    parser.add_argument('--tasks', type=str, nargs='+', default=['all'],
                        help='要评估的任务列表，默认评估所有任务')
    parser.add_argument('--ntrain', type=int, default=5,
                        help='Few-shot示例数量')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                        help='每个任务最大评估样本数（用于快速测试）')
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
    parser.add_argument('--max_gen_len', type=int, default=256,
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


def generate_few_shot_prompt(task, ntrain=5):
    """生成few-shot prompt"""
    if not task.train_data or ntrain == 0:
        return ""
    
    prompt = ""
    # 随机选择ntrain个样本
    indices = np.random.choice(len(task.train_data), min(ntrain, len(task.train_data)), replace=False)
    
    for idx in indices:
        example = task.train_data[idx]
        prompt += task.format_example(example, include_answer=True) + "\n"
    
    return prompt


def evaluate_task(model, task, args):
    """评估单个任务"""
    print(f"\n评估任务: {task.task_name}")
    
    # 加载数据
    task.load_data()
    
    if not task.dev_data:
        print(f"警告: {task.task_name} 没有验证集数据")
        return None
    
    # 生成few-shot prompt
    few_shot_prompt = generate_few_shot_prompt(task, args.ntrain)
    
    # 评估样本
    eval_data = task.dev_data
    if args.max_eval_samples:
        eval_data = eval_data[:args.max_eval_samples]
    
    predictions = []
    references = []
    
    for example in tqdm(eval_data, desc=f"评估 {task.task_name}"):
        # 构建输入
        prompt = task.format_example(example, include_answer=False)
        full_prompt = few_shot_prompt + prompt
        
        # 截断到最大长度
        tokens = model.tokenizer.encode(full_prompt, bos=True, eos=False)
        if len(tokens) > args.max_seq_len - args.max_gen_len:
            tokens = tokens[-(args.max_seq_len - args.max_gen_len):]
            full_prompt = model.tokenizer.decode(tokens)
        
        # 生成预测
        with torch.no_grad():
            output = model.generate(
                prompts=[full_prompt],
                images=None,
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p,
            )[0]
        
        # 提取答案
        generated_text = output[len(full_prompt):]
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
    
    return metrics


def evaluate_clue(model, args):
    """评估所有CLUE任务"""
    results = {}
    
    # 确定要评估的任务
    if 'all' in args.tasks:
        tasks_to_eval = list(TASK_REGISTRY.keys())
    else:
        tasks_to_eval = args.tasks
    
    # 评估每个任务
    for task_name in tasks_to_eval:
        if task_name not in TASK_REGISTRY:
            print(f"警告: 未知任务 {task_name}，跳过")
            continue
        
        try:
            # 创建任务实例
            task = get_task(task_name, args.data_dir)
            
            # 评估任务
            metrics = evaluate_task(model, task, args)
            
            if metrics:
                results[task_name] = metrics
                print(f"{task_name} 结果: {metrics}")
        
        except Exception as e:
            print(f"评估 {task_name} 时出错: {e}")
            continue
    
    return results


def compute_average_score(results):
    """计算平均分数"""
    scores = []
    
    for task_name, metrics in results.items():
        if 'accuracy' in metrics:
            scores.append(metrics['accuracy'])
    
    if scores:
        avg_score = np.mean(scores)
        return avg_score
    else:
        return 0.0


def main(args):
    # 创建结果目录
    path_split = args.pretrained_path.split('/')
    if path_split[-1] == '':
        path_split.pop(-1)
    model_name = path_split[-1]
    
    result_dir = Path('results') / model_name / 'clue'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = result_dir / 'evaluation_results.json'
    
    # 检查是否已有结果
    if not args.overwrite and result_path.exists():
        print(f"结果文件已存在: {result_path}")
        with open(result_path, 'r') as f:
            results = json.load(f)
        print("已有结果:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    
    # 加载模型
    model = load_model(args)
    
    # 评估CLUE
    results = evaluate_clue(model, args)
    
    # 计算平均分
    avg_score = compute_average_score(results)
    results['average'] = avg_score
    
    # 保存结果
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(result_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*50)
        print("CLUE评估结果:")
        print("="*50)
        
        for task_name, metrics in results.items():
            if task_name != 'average':
                print(f"{task_name}: {metrics}")
        
        print(f"\n平均分数: {avg_score:.4f}")
        print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)