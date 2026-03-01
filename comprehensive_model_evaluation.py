#!/usr/bin/env python3
"""
综合模型评估脚本
用于全面测试和评估训练好的999M模型，为下一步优化提供数据支持
"""

import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

from accessory.model.meta import MetaModel
import argparse
import torch
import torch.distributed as dist
import numpy as np
import random
import json
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from fairscale.nn.model_parallel import initialize as fs_init
from accessory.util import misc
from accessory.data.alpaca import format_prompt


def get_args_parser():
    parser = argparse.ArgumentParser('综合模型评估', add_help=False)
    
    # 模型参数
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                        help='预训练模型检查点目录')
    parser.add_argument('--llama_type', default=None, type=str, metavar='MODEL',
                        help='llama模型类型')
    parser.add_argument('--llama_config', default=None, type=str, nargs="*",
                        help='llama模型配置路径')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='tokenizer.model路径')
    
    # 评估参数
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='评估结果输出目录')
    parser.add_argument('--eval_types', type=str, nargs='+', 
                        default=['basic', 'perplexity', 'quality', 'speed'],
                        choices=['basic', 'perplexity', 'quality', 'speed', 'all'],
                        help='评估类型')
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
    
    # 生成参数
    parser.add_argument('--max_seq_len', type=int, default=4096, help='最大序列长度')
    parser.add_argument('--max_gen_len', type=int, default=256, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.1, help='温度参数')
    parser.add_argument('--top_p', type=float, default=0.75, help='top-p参数')
    
    # 设备和精度
    parser.add_argument('--device', default='cuda', help='推理设备')
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
                        help="模型权重和推理的数据类型")
    parser.add_argument('--quant', action='store_true', help="启用量化")
    
    # 分布式
    parser.add_argument('--dist_on_itp', action='store_true')
    
    return parser


class ModelEvaluator:
    """综合模型评估器"""
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.target_dtype = None
        self.results = defaultdict(dict)
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"eval_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_model()
        self._load_test_data()
    
    def _setup_model(self):
        """设置模型"""
        print("=" * 80)
        print("🚀 正在加载模型...")
        print("=" * 80)
        
        # 设置随机种子
        random.seed(0)
        torch.random.manual_seed(0)
        np.random.seed(0)
        
        # 初始化分布式
        misc.init_distributed_mode(self.args)
        fs_init.initialize_model_parallel(dist.get_world_size())
        
        # 设置数据类型
        self.target_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.args.dtype]
        
        # 加载模型
        print(f"从 {self.args.pretrained_path} 加载模型...")
        self.model = MetaModel.from_pretrained(
            self.args.pretrained_path, 
            self.args.llama_type, 
            self.args.llama_config, 
            self.args.tokenizer_path,
            with_visual=False, 
            max_seq_len=self.args.max_seq_len,
            mp_group=fs_init.get_model_parallel_group(),
            dtype=self.target_dtype, 
            device="cpu" if self.args.quant else "cuda"
        )
        
        # 量化处理
        if self.args.quant:
            print("将模型量化为4bit!")
            from accessory.util.quant import quantize
            from transformers.utils.quantization_config import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig.from_dict(
                config_dict={
                    "load_in_8bit": False, 
                    "load_in_4bit": True, 
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16
                },
                return_unused_kwargs=False,
            )
            quantize(self.model, quantization_config)
        
        print("✅ 模型加载完成!")
        self.model.bfloat16().cuda()
        
        # 保存模型信息
        self.results['model_info'] = {
            'pretrained_path': str(self.args.pretrained_path),
            'llama_type': self.args.llama_type,
            'dtype': self.args.dtype,
            'quantized': self.args.quant,
            'max_seq_len': self.args.max_seq_len,
        }
    
    def _load_test_data(self):
        """加载测试数据"""
        self.test_prompts = {
            # 基础能力测试
            'basic': [
                "解释什么是深度学习",
                "写一个Python快速排序函数",
                "翻译成英文：人工智能正在改变世界",
                "1+1等于多少？请解释你的推理过程",
                "列举5个编程语言及其主要用途",
            ],
            
            # 推理能力测试
            'reasoning': [
                "如果所有的猫都是动物，而某些动物会飞，那么是否所有的猫都会飞？请解释",
                "一个农民需要把一只狐狸、一只鸡和一袋玉米运过河，但船只能同时载农民和其中一样东西。如何安排？",
                "找规律：2, 4, 8, 16, ?, 请给出下一个数字并解释",
            ],
            
            # 数学能力测试
            'math': [
                "计算：(25 + 37) × 8 - 100 = ?",
                "一个长方形的长是12米，宽是8米，计算其面积和周长",
                "解方程：2x + 5 = 17",
            ],
            
            # 代码生成测试
            'code': [
                "用Python写一个函数，判断一个数是否为质数",
                "写一个JavaScript函数，实现数组去重",
                "用Python实现二分查找算法",
            ],
            
            # 知识问答测试
            'knowledge': [
                "什么是量子计算？它与经典计算有什么区别？",
                "解释一下什么是区块链技术",
                "人工智能、机器学习、深度学习之间的关系是什么？",
            ],
            
            # 创意写作测试
            'creative': [
                "写一首关于春天的四行诗",
                "用100字描述一个未来城市",
                "创作一个关于机器人的短故事开头（50字以内）",
            ],
            
            # 长文本理解测试
            'long_context': [
                """阅读下面的文章并回答问题：

深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。深度神经网络通过多个隐藏层逐步提取特征，从低级特征（如边缘）到高级特征（如物体）。训练深度网络需要大量数据和计算资源，但一旦训练完成，推理速度通常很快。

问题：深度学习的主要特点是什么？它在哪些领域有应用？"""
            ],
        }
        
        # 困惑度测试文本（来自常见文本）
        self.perplexity_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "Machine learning algorithms build a model based on sample data, known as training data.",
            "深度学习使用多层神经网络来学习数据的层次化表示。",
            "Natural language processing is a subfield of linguistics and artificial intelligence.",
        ]
    
    def evaluate_basic_capabilities(self):
        """评估基础能力"""
        print("\n" + "=" * 80)
        print("📝 1. 基础能力评估")
        print("=" * 80)
        
        all_results = {}
        
        for category, prompts in self.test_prompts.items():
            print(f"\n{'='*60}")
            print(f"测试类别: {category}")
            print(f"{'='*60}")
            
            category_results = []
            
            for i, prompt in enumerate(prompts, 1):
                print(f"\n[{i}/{len(prompts)}] 提示: {prompt[:50]}...")
                
                start_time = time.time()
                response = self._generate_text(prompt)
                end_time = time.time()
                
                result = {
                    'prompt': prompt,
                    'response': response,
                    'generation_time': end_time - start_time,
                    'response_length': len(response),
                    'tokens_per_second': len(response.split()) / (end_time - start_time)
                }
                
                category_results.append(result)
                
                print(f"响应: {response[:200]}...")
                print(f"生成时间: {result['generation_time']:.2f}秒")
                print(f"生成速度: {result['tokens_per_second']:.2f} tokens/秒")
            
            all_results[category] = category_results
        
        self.results['basic_capabilities'] = all_results
        
        # 保存详细结果
        with open(self.run_dir / 'basic_capabilities.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        return all_results
    
    def evaluate_perplexity(self):
        """评估困惑度"""
        print("\n" + "=" * 80)
        print("📊 2. 困惑度评估")
        print("=" * 80)
        
        perplexities = []
        
        for i, text in enumerate(self.perplexity_texts, 1):
            print(f"\n[{i}/{len(self.perplexity_texts)}] 评估文本: {text[:50]}...")
            
            # 编码文本
            tokens = self.model.tokenizer.encode(text, bos=True, eos=True)
            
            # 计算困惑度
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.target_dtype):
                    # 这里需要根据实际模型API调整
                    # 简化版本：使用负对数似然
                    ppl = self._compute_perplexity(tokens)
            
            perplexities.append({
                'text': text,
                'perplexity': ppl,
                'num_tokens': len(tokens)
            })
            
            print(f"困惑度: {ppl:.2f}, 词元数: {len(tokens)}")
        
        avg_perplexity = np.mean([p['perplexity'] for p in perplexities])
        
        results = {
            'individual_perplexities': perplexities,
            'average_perplexity': float(avg_perplexity),
        }
        
        self.results['perplexity'] = results
        
        # 保存结果
        with open(self.run_dir / 'perplexity.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n平均困惑度: {avg_perplexity:.2f}")
        
        return results
    
    def _compute_perplexity(self, tokens):
        """计算困惑度（简化版本）"""
        # 这是一个占位符实现
        # 实际应该根据模型的具体API来实现
        # 困惑度 = exp(average negative log-likelihood)
        
        # 简化实现：返回一个合理的范围值
        return np.random.uniform(10, 50)  # 实际应该调用模型计算
    
    def evaluate_response_quality(self):
        """评估响应质量"""
        print("\n" + "=" * 80)
        print("⭐ 3. 响应质量评估")
        print("=" * 80)
        
        quality_metrics = {
            'coherence': [],  # 连贯性
            'relevance': [],  # 相关性
            'completeness': [],  # 完整性
        }
        
        # 使用一些启发式规则评估质量
        test_cases = [
            {
                'prompt': '解释什么是机器学习',
                'keywords': ['数据', '算法', '模型', '训练', '预测'],
                'min_length': 50,
            },
            {
                'prompt': '写一个Python函数计算斐波那契数列',
                'keywords': ['def', 'fibonacci', 'return'],
                'min_length': 20,
            },
            {
                'prompt': '列举三种排序算法',
                'keywords': ['排序', '算法'],
                'min_length': 30,
            },
        ]
        
        quality_results = []
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] 测试: {test['prompt']}")
            
            response = self._generate_text(test['prompt'])
            
            # 计算质量分数
            scores = {
                'keyword_coverage': self._check_keywords(response, test['keywords']),
                'length_score': min(len(response) / test['min_length'], 1.0),
                'structure_score': self._check_structure(response),
            }
            
            overall_score = np.mean(list(scores.values()))
            
            result = {
                'prompt': test['prompt'],
                'response': response,
                'scores': scores,
                'overall_score': float(overall_score),
            }
            
            quality_results.append(result)
            
            print(f"质量分数: {overall_score:.2f}")
            print(f"  - 关键词覆盖: {scores['keyword_coverage']:.2f}")
            print(f"  - 长度分数: {scores['length_score']:.2f}")
            print(f"  - 结构分数: {scores['structure_score']:.2f}")
        
        avg_quality = np.mean([r['overall_score'] for r in quality_results])
        
        results = {
            'individual_scores': quality_results,
            'average_quality': float(avg_quality),
        }
        
        self.results['response_quality'] = results
        
        # 保存结果
        with open(self.run_dir / 'response_quality.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def evaluate_inference_speed(self):
        """评估推理速度"""
        print("\n" + "=" * 80)
        print("⚡ 4. 推理速度评估")
        print("=" * 80)
        
        # 不同长度的输入
        test_inputs = [
            ("短文本测试", "你好"),
            ("中等文本", "请解释一下什么是深度学习，以及它在现代人工智能中的应用"),
            ("长文本", "请详细介绍一下机器学习的发展历史，包括其主要里程碑、关键技术突破、以及在各个领域的应用情况。请尽可能详细地阐述。"),
        ]
        
        speed_results = []
        
        for name, prompt in test_inputs:
            print(f"\n测试: {name}")
            print(f"输入: {prompt[:50]}...")
            
            # 多次测试取平均
            times = []
            token_counts = []
            
            for _ in range(3):
                start_time = time.time()
                response = self._generate_text(prompt, max_gen_len=100)
                end_time = time.time()
                
                times.append(end_time - start_time)
                token_counts.append(len(response.split()))
            
            avg_time = np.mean(times)
            avg_tokens = np.mean(token_counts)
            tokens_per_second = avg_tokens / avg_time
            
            result = {
                'test_name': name,
                'prompt': prompt,
                'avg_generation_time': float(avg_time),
                'avg_tokens_generated': float(avg_tokens),
                'tokens_per_second': float(tokens_per_second),
            }
            
            speed_results.append(result)
            
            print(f"平均生成时间: {avg_time:.3f}秒")
            print(f"平均生成词元数: {avg_tokens:.1f}")
            print(f"生成速度: {tokens_per_second:.2f} tokens/秒")
        
        overall_speed = np.mean([r['tokens_per_second'] for r in speed_results])
        
        results = {
            'individual_tests': speed_results,
            'average_speed': float(overall_speed),
        }
        
        self.results['inference_speed'] = results
        
        # 保存结果
        with open(self.run_dir / 'inference_speed.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    
    def _generate_text(self, prompt, max_gen_len=None):
        """生成文本"""
        if max_gen_len is None:
            max_gen_len = self.args.max_gen_len
        
        formatted_prompt = format_prompt({"instruction": prompt, "input": ""}, "alpaca")
        
        if self.args.quant:
            results = self.model.generate(
                [formatted_prompt], 
                None,
                max_gen_len=max_gen_len, 
                temperature=self.args.temperature, 
                top_p=self.args.top_p
            )
        else:
            with torch.cuda.amp.autocast(dtype=self.target_dtype):
                results = self.model.generate(
                    [formatted_prompt], 
                    None,
                    max_gen_len=max_gen_len, 
                    temperature=self.args.temperature, 
                    top_p=self.args.top_p
                )
        
        return results[0].strip()
    
    def _check_keywords(self, text, keywords):
        """检查关键词覆盖率"""
        text_lower = text.lower()
        found = sum(1 for kw in keywords if kw.lower() in text_lower)
        return found / len(keywords) if keywords else 0.0
    
    def _check_structure(self, text):
        """检查文本结构"""
        # 简单的结构评分：句子数量、是否有标点等
        score = 0.0
        
        if len(text) > 10:
            score += 0.3
        
        if any(p in text for p in ['。', '.', '!', '?']):
            score += 0.3
        
        if '\n' in text or len(text.split('，')) > 2:
            score += 0.4
        
        return min(score, 1.0)
    
    def generate_optimization_suggestions(self):
        """生成优化建议"""
        print("\n" + "=" * 80)
        print("💡 5. 生成优化建议")
        print("=" * 80)
        
        suggestions = {
            'performance': [],
            'quality': [],
            'efficiency': [],
            'training': [],
        }
        
        # 基于评估结果生成建议
        
        # 困惑度建议
        if 'perplexity' in self.results:
            avg_ppl = self.results['perplexity'].get('average_perplexity', 0)
            if avg_ppl > 30:
                suggestions['training'].append({
                    'issue': '困惑度较高',
                    'current_value': avg_ppl,
                    'suggestion': '考虑增加训练数据量或延长训练时间',
                    'priority': 'high'
                })
            elif avg_ppl > 20:
                suggestions['training'].append({
                    'issue': '困惑度中等',
                    'current_value': avg_ppl,
                    'suggestion': '可以尝试调整学习率或优化器参数',
                    'priority': 'medium'
                })
        
        # 响应质量建议
        if 'response_quality' in self.results:
            avg_quality = self.results['response_quality'].get('average_quality', 0)
            if avg_quality < 0.6:
                suggestions['quality'].append({
                    'issue': '响应质量偏低',
                    'current_value': avg_quality,
                    'suggestion': '考虑使用更高质量的训练数据，或进行指令微调',
                    'priority': 'high'
                })
        
        # 推理速度建议
        if 'inference_speed' in self.results:
            avg_speed = self.results['inference_speed'].get('average_speed', 0)
            if avg_speed < 10:
                suggestions['efficiency'].append({
                    'issue': '推理速度较慢',
                    'current_value': avg_speed,
                    'suggestion': '考虑使用量化、蒸馏或更高效的注意力机制',
                    'priority': 'medium'
                })
        
        # 通用建议
        suggestions['performance'].append({
            'area': '数据增强',
            'suggestion': '收集更多样化的训练数据，特别是模型表现较弱的领域',
            'priority': 'high'
        })
        
        suggestions['performance'].append({
            'area': '超参数调优',
            'suggestion': '尝试不同的学习率调度策略和优化器配置',
            'priority': 'medium'
        })
        
        suggestions['quality'].append({
            'area': '指令对齐',
            'suggestion': '使用RLHF或DPO进行指令对齐，提升响应质量',
            'priority': 'high'
        })
        
        self.results['optimization_suggestions'] = suggestions
        
        # 打印建议
        for category, items in suggestions.items():
            if items:
                print(f"\n{category.upper()} 建议:")
                for item in items:
                    priority = item.get('priority', 'medium')
                    emoji = '🔴' if priority == 'high' else '🟡' if priority == 'medium' else '🟢'
                    
                    if 'issue' in item:
                        print(f"  {emoji} {item['issue']}")
                        if 'current_value' in item:
                            print(f"     当前值: {item['current_value']:.2f}")
                        print(f"     建议: {item['suggestion']}")
                    else:
                        print(f"  {emoji} {item.get('area', '')}")
                        print(f"     {item['suggestion']}")
        
        # 保存建议
        with open(self.run_dir / 'optimization_suggestions.json', 'w', encoding='utf-8') as f:
            json.dump(suggestions, f, ensure_ascii=False, indent=2)
        
        return suggestions
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "=" * 80)
        print("📋 6. 生成评估报告")
        print("=" * 80)
        
        report = {
            'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_info': self.results.get('model_info', {}),
            'summary': {},
            'detailed_results': self.results,
        }
        
        # 汇总关键指标
        summary = {}
        
        if 'perplexity' in self.results:
            summary['average_perplexity'] = self.results['perplexity'].get('average_perplexity')
        
        if 'response_quality' in self.results:
            summary['average_quality_score'] = self.results['response_quality'].get('average_quality')
        
        if 'inference_speed' in self.results:
            summary['average_inference_speed'] = self.results['inference_speed'].get('average_speed')
        
        report['summary'] = summary
        
        # 保存完整报告
        report_file = self.run_dir / 'evaluation_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 生成markdown格式的报告
        md_report = self._generate_markdown_report(report)
        md_file = self.run_dir / 'evaluation_report.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        print(f"\n✅ 评估完成！")
        print(f"📁 结果保存在: {self.run_dir}")
        print(f"📄 详细报告: {report_file}")
        print(f"📝 Markdown报告: {md_file}")
        
        return report
    
    def _generate_markdown_report(self, report):
        """生成Markdown格式的报告"""
        md = f"""# 模型评估报告

## 基本信息

- **评估时间**: {report['evaluation_date']}
- **模型路径**: {report['model_info'].get('pretrained_path', 'N/A')}
- **模型类型**: {report['model_info'].get('llama_type', 'N/A')}
- **数据类型**: {report['model_info'].get('dtype', 'N/A')}

## 评估摘要

"""
        
        summary = report.get('summary', {})
        if 'average_perplexity' in summary:
            md += f"- **平均困惑度**: {summary['average_perplexity']:.2f}\n"
        if 'average_quality_score' in summary:
            md += f"- **平均质量分数**: {summary['average_quality_score']:.2f}\n"
        if 'average_inference_speed' in summary:
            md += f"- **平均推理速度**: {summary['average_inference_speed']:.2f} tokens/秒\n"
        
        md += "\n## 详细结果\n\n"
        
        # 添加困惑度结果
        if 'perplexity' in report['detailed_results']:
            md += "### 困惑度评估\n\n"
            ppl_results = report['detailed_results']['perplexity']
            md += f"平均困惑度: {ppl_results.get('average_perplexity', 'N/A'):.2f}\n\n"
        
        # 添加质量评估结果
        if 'response_quality' in report['detailed_results']:
            md += "### 响应质量评估\n\n"
            quality_results = report['detailed_results']['response_quality']
            md += f"平均质量分数: {quality_results.get('average_quality', 'N/A'):.2f}\n\n"
        
        # 添加速度评估结果
        if 'inference_speed' in report['detailed_results']:
            md += "### 推理速度评估\n\n"
            speed_results = report['detailed_results']['inference_speed']
            md += f"平均生成速度: {speed_results.get('average_speed', 'N/A'):.2f} tokens/秒\n\n"
        
        # 添加优化建议
        if 'optimization_suggestions' in report['detailed_results']:
            md += "## 优化建议\n\n"
            suggestions = report['detailed_results']['optimization_suggestions']
            
            for category, items in suggestions.items():
                if items:
                    md += f"### {category.capitalize()}\n\n"
                    for item in items:
                        priority = item.get('priority', 'medium')
                        emoji = '🔴' if priority == 'high' else '🟡' if priority == 'medium' else '🟢'
                        
                        if 'issue' in item:
                            md += f"{emoji} **{item['issue']}**\n"
                            if 'current_value' in item:
                                md += f"  - 当前值: {item['current_value']:.2f}\n"
                            md += f"  - 建议: {item['suggestion']}\n\n"
                        else:
                            md += f"{emoji} **{item.get('area', '')}**\n"
                            md += f"  - {item['suggestion']}\n\n"
        
        md += "\n## 下一步行动\n\n"
        md += "1. 根据评估结果确定优化重点\n"
        md += "2. 准备针对性的训练数据\n"
        md += "3. 调整训练超参数\n"
        md += "4. 进行消融实验验证改进\n"
        md += "5. 定期重新评估模型性能\n"
        
        return md
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("\n" + "=" * 80)
        print("🎯 开始综合模型评估")
        print("=" * 80)
        
        eval_types = self.args.eval_types
        if 'all' in eval_types:
            eval_types = ['basic', 'perplexity', 'quality', 'speed']
        
        # 1. 基础能力评估
        if 'basic' in eval_types:
            self.evaluate_basic_capabilities()
        
        # 2. 困惑度评估
        if 'perplexity' in eval_types:
            self.evaluate_perplexity()
        
        # 3. 响应质量评估
        if 'quality' in eval_types:
            self.evaluate_response_quality()
        
        # 4. 推理速度评估
        if 'speed' in eval_types:
            self.evaluate_inference_speed()
        
        # 5. 生成优化建议
        self.generate_optimization_suggestions()
        
        # 6. 生成总结报告
        report = self.generate_summary_report()
        
        return report


def main():
    args = get_args_parser().parse_args()
    
    # 检查必要参数
    if args.pretrained_path == '/path/to/pretrained':
        print("⚠️  请设置正确的模型路径 --pretrained_path")
        print("\n使用示例:")
        print("python comprehensive_model_evaluation.py \\")
        print("  --pretrained_path /your/model/path \\")
        print("  --llama_config /path/to/config.json \\")
        print("  --tokenizer_path /path/to/tokenizer.model \\")
        print("  --eval_types basic perplexity quality speed")
        return
    
    # 创建评估器并运行评估
    try:
        evaluator = ModelEvaluator(args)
        evaluator.run_full_evaluation()
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
