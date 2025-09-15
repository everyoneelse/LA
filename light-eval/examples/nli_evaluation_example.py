#!/usr/bin/env python3
"""
NLI任务评估示例
展示预训练模型如何在OCNLI/CMNLI上进行评估
"""

import json
from typing import List, Dict

class NLIEvaluationExample:
    """NLI评估示例类"""
    
    def __init__(self):
        self.label_map = {
            'entailment': '蕴含',
            'neutral': '中立',
            'contradiction': '矛盾'
        }
        
    def format_zero_shot_prompt(self, example: Dict) -> str:
        """构建Zero-shot prompt"""
        prompt = (
            "请判断两个句子之间的逻辑关系。\n"
            "如果第二个句子是第一个句子的必然结果，回答'蕴含'；\n"
            "如果两个句子相互矛盾，回答'矛盾'；\n"
            "如果两个句子既不蕴含也不矛盾，回答'中立'。\n\n"
            f"句子1：{example['sentence1']}\n"
            f"句子2：{example['sentence2']}\n"
            "关系："
        )
        return prompt
    
    def format_few_shot_prompt(self, example: Dict, demonstrations: List[Dict]) -> str:
        """构建Few-shot prompt"""
        prompt = "请判断两个句子之间的逻辑关系。\n\n"
        prompt += "以下是一些示例：\n\n"
        
        # 添加示例
        for demo in demonstrations:
            prompt += f"句子1：{demo['sentence1']}\n"
            prompt += f"句子2：{demo['sentence2']}\n"
            prompt += f"关系：{self.label_map[demo['label']]}\n\n"
        
        # 添加待预测样本
        prompt += "现在请回答：\n\n"
        prompt += f"句子1：{example['sentence1']}\n"
        prompt += f"句子2：{example['sentence2']}\n"
        prompt += "关系："
        
        return prompt
    
    def extract_prediction(self, model_output: str) -> str:
        """从模型输出中提取预测标签"""
        output = model_output.strip()
        
        # 匹配中文标签
        if '蕴含' in output[:20]:
            return 'entailment'
        elif '矛盾' in output[:20]:
            return 'contradiction'
        elif '中立' in output[:20]:
            return 'neutral'
        
        # 匹配英文标签
        output_lower = output.lower()
        if 'entail' in output_lower[:30]:
            return 'entailment'
        elif 'contradict' in output_lower[:30]:
            return 'contradiction'
        else:
            # 默认返回中立
            return 'neutral'
    
    def evaluate_single_sample(self, model, example: Dict, mode: str = 'zero-shot', 
                              demonstrations: List[Dict] = None) -> Dict:
        """评估单个样本"""
        
        # 1. 构建prompt
        if mode == 'zero-shot':
            prompt = self.format_zero_shot_prompt(example)
        else:
            prompt = self.format_few_shot_prompt(example, demonstrations or [])
        
        # 2. 模型生成
        # 这里是伪代码，实际需要调用模型
        model_output = self.mock_model_generate(prompt)
        
        # 3. 提取预测
        prediction = self.extract_prediction(model_output)
        
        # 4. 计算是否正确
        is_correct = prediction == example['label']
        
        return {
            'prediction': prediction,
            'ground_truth': example['label'],
            'correct': is_correct,
            'prompt': prompt,
            'model_output': model_output
        }
    
    def mock_model_generate(self, prompt: str) -> str:
        """模拟模型生成（实际使用时替换为真实模型）"""
        # 这里只是示例，实际需要调用真实的预训练模型
        if "棉大衣" in prompt and "至少一件衣服" in prompt:
            return "蕴含"
        elif "矛盾" in prompt:
            return "矛盾"
        else:
            return "中立"
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """计算评估指标"""
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        # 计算每个类别的准确率
        label_stats = {}
        for label in ['entailment', 'neutral', 'contradiction']:
            label_results = [r for r in results if r['ground_truth'] == label]
            if label_results:
                label_correct = sum(1 for r in label_results if r['correct'])
                label_stats[label] = {
                    'accuracy': label_correct / len(label_results),
                    'count': len(label_results)
                }
        
        return {
            'overall_accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_label_accuracy': label_stats
        }


def demonstrate_evaluation():
    """演示评估流程"""
    
    # 示例数据
    test_example = {
        "sentence1": "身上裹一件工厂发的棉大衣,手插在袖筒里",
        "sentence2": "身上至少一件衣服",
        "label": "entailment"
    }
    
    # Few-shot示例
    demonstrations = [
        {
            "sentence1": "今天下雨了，地面都湿了",
            "sentence2": "地面是湿的",
            "label": "entailment"
        },
        {
            "sentence1": "小明在北京工作",
            "sentence2": "小明在上海工作",
            "label": "contradiction"
        },
        {
            "sentence1": "这个苹果很甜",
            "sentence2": "这个苹果是红色的",
            "label": "neutral"
        }
    ]
    
    evaluator = NLIEvaluationExample()
    
    print("="*60)
    print("NLI任务评估示例")
    print("="*60)
    
    # Zero-shot评估
    print("\n1. Zero-shot评估")
    print("-"*40)
    zero_shot_prompt = evaluator.format_zero_shot_prompt(test_example)
    print("Prompt:")
    print(zero_shot_prompt)
    print("\n模型输出: 蕴含")
    print("真实标签: entailment")
    print("预测正确: ✓")
    
    # Few-shot评估
    print("\n2. Few-shot评估")
    print("-"*40)
    few_shot_prompt = evaluator.format_few_shot_prompt(test_example, demonstrations)
    print("Prompt (截取):")
    print(few_shot_prompt[:500] + "...")
    print("\n模型输出: 蕴含")
    print("真实标签: entailment")
    print("预测正确: ✓")
    
    # 评估指标
    print("\n3. 评估指标计算")
    print("-"*40)
    
    # 模拟一批评估结果
    mock_results = [
        {'prediction': 'entailment', 'ground_truth': 'entailment', 'correct': True},
        {'prediction': 'neutral', 'ground_truth': 'neutral', 'correct': True},
        {'prediction': 'contradiction', 'ground_truth': 'contradiction', 'correct': True},
        {'prediction': 'neutral', 'ground_truth': 'entailment', 'correct': False},
        {'prediction': 'entailment', 'ground_truth': 'neutral', 'correct': False},
    ]
    
    metrics = evaluator.calculate_metrics(mock_results)
    print(f"总体准确率: {metrics['overall_accuracy']:.2%}")
    print(f"正确数/总数: {metrics['correct']}/{metrics['total']}")
    print("\n各类别准确率:")
    for label, stats in metrics['per_label_accuracy'].items():
        print(f"  {label}: {stats['accuracy']:.2%} (样本数: {stats['count']})")


if __name__ == "__main__":
    demonstrate_evaluation()