"""
CLUE任务定义 - 专门针对预训练模型评估优化版本
只包含适合zero-shot/few-shot评估的任务
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import re


class CLUEPretrainTask:
    """CLUE预训练评估任务基类"""
    
    def __init__(self, task_name: str, data_dir: str):
        self.task_name = task_name
        self.data_dir = Path(data_dir) / task_name
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        
    def load_data(self):
        """加载数据集"""
        raise NotImplementedError
        
    def format_zero_shot_prompt(self, example: Dict) -> str:
        """格式化zero-shot prompt"""
        raise NotImplementedError
        
    def format_few_shot_prompt(self, example: Dict, demonstrations: List[Dict]) -> str:
        """格式化few-shot prompt"""
        prompt = ""
        
        # 添加任务描述
        prompt += self.get_task_description() + "\n\n"
        
        # 添加示例
        if demonstrations:
            prompt += "以下是一些示例：\n\n"
            for demo in demonstrations:
                prompt += self.format_example(demo, include_answer=True) + "\n"
            prompt += "现在请回答：\n\n"
        
        # 添加待预测样本
        prompt += self.format_example(example, include_answer=False)
        
        return prompt
        
    def format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化单个样本"""
        raise NotImplementedError
        
    def get_task_description(self) -> str:
        """获取任务描述"""
        raise NotImplementedError
        
    def extract_answer(self, text: str) -> str:
        """从模型输出中提取答案"""
        raise NotImplementedError
        
    def compute_metric(self, predictions: List, references: List) -> Dict:
        """计算评估指标"""
        correct = sum(1 for p, r in zip(predictions, references) if str(p) == str(r))
        accuracy = correct / len(predictions) if predictions else 0
        return {'accuracy': accuracy}


class CMNLITask(CLUEPretrainTask):
    """中文自然语言推理任务 - 最适合预训练模型"""
    
    def __init__(self, data_dir: str):
        super().__init__('cmnli', data_dir)
        self.labels = ['entailment', 'neutral', 'contradiction']
        self.label_map = {
            'entailment': '蕴含',
            'neutral': '中立', 
            'contradiction': '矛盾'
        }
        
    def get_task_description(self) -> str:
        return ("请判断两个句子之间的逻辑关系。\n"
                "如果第二个句子是第一个句子的必然结果，选择'蕴含'；\n"
                "如果两个句子相互矛盾，选择'矛盾'；\n"
                "如果两个句子既不蕴含也不矛盾，选择'中立'。")
        
    def load_data(self):
        """加载CMNLI数据"""
        train_path = self.data_dir / 'cmnli_public' / 'train.json'
        dev_path = self.data_dir / 'cmnli_public' / 'dev.json'
        
        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                self.train_data = [json.loads(line) for line in f]
        
        if dev_path.exists():
            with open(dev_path, 'r', encoding='utf-8') as f:
                self.dev_data = [json.loads(line) for line in f]
                
    def format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化CMNLI样本"""
        prompt = f"前提：{example['sentence1']}\n"
        prompt += f"假设：{example['sentence2']}\n"
        prompt += "关系："
        
        if include_answer and 'label' in example:
            prompt += f"{self.label_map[example['label']]}"
        
        return prompt
    
    def format_zero_shot_prompt(self, example: Dict) -> str:
        """Zero-shot prompt"""
        prompt = self.get_task_description() + "\n\n"
        prompt += self.format_example(example, include_answer=False)
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip()
        
        # 直接匹配中文标签
        if '蕴含' in text[:20]:
            return 'entailment'
        elif '矛盾' in text[:20]:
            return 'contradiction'
        elif '中立' in text[:20]:
            return 'neutral'
        
        # 尝试英文
        text_lower = text.lower()
        if 'entail' in text_lower[:30]:
            return 'entailment'
        elif 'contradict' in text_lower[:30]:
            return 'contradiction'
        else:
            return 'neutral'


class AFQMCTask(CLUEPretrainTask):
    """蚂蚁金融语义相似度任务 - 适合预训练模型"""
    
    def __init__(self, data_dir: str):
        super().__init__('afqmc', data_dir)
        self.labels = ['0', '1']  # 0: 不相似, 1: 相似
        
    def get_task_description(self) -> str:
        return "请判断下面两个句子的语义是否相似。如果意思基本相同，回答'相似'；如果意思不同，回答'不相似'。"
        
    def load_data(self):
        """加载AFQMC数据"""
        train_path = self.data_dir / 'afqmc_public' / 'train.json'
        dev_path = self.data_dir / 'afqmc_public' / 'dev.json'
        
        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                self.train_data = [json.loads(line) for line in f]
        
        if dev_path.exists():
            with open(dev_path, 'r', encoding='utf-8') as f:
                self.dev_data = [json.loads(line) for line in f]
                
    def format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化AFQMC样本"""
        prompt = f"句子1：{example['sentence1']}\n"
        prompt += f"句子2：{example['sentence2']}\n"
        prompt += "判断："
        
        if include_answer and 'label' in example:
            answer = '相似' if example['label'] == '1' else '不相似'
            prompt += answer
            
        return prompt
    
    def format_zero_shot_prompt(self, example: Dict) -> str:
        """Zero-shot prompt"""
        prompt = self.get_task_description() + "\n\n"
        prompt += self.format_example(example, include_answer=False)
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip()
        
        # 检查前20个字符
        if '不' in text[:20] and '相似' in text[:20]:
            return '0'
        elif '相似' in text[:20]:
            return '1'
        
        # 检查其他否定词
        if any(neg in text[:20] for neg in ['不同', '不一样', '无关', '没有关系']):
            return '0'
        elif any(pos in text[:20] for pos in ['相同', '一样', '一致', '相关']):
            return '1'
        
        # 默认
        return '0'


class CSLTask(CLUEPretrainTask):
    """中文科学文献关键词识别任务 - 适合预训练模型"""
    
    def __init__(self, data_dir: str):
        super().__init__('csl', data_dir)
        
    def get_task_description(self) -> str:
        return ("请判断给定的关键词是否真的是这篇论文的关键词。\n"
                "真实的关键词应该准确概括论文的核心内容。")
        
    def load_data(self):
        """加载CSL数据"""
        train_path = self.data_dir / 'csl_public' / 'train.json'
        dev_path = self.data_dir / 'csl_public' / 'dev.json'
        
        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                self.train_data = [json.loads(line) for line in f]
        
        if dev_path.exists():
            with open(dev_path, 'r', encoding='utf-8') as f:
                self.dev_data = [json.loads(line) for line in f]
                
    def format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化CSL样本"""
        # 限制摘要长度
        abstract = example['abst']
        if len(abstract) > 200:
            abstract = abstract[:200] + "..."
            
        prompt = f"标题：{example['title']}\n"
        prompt += f"摘要：{abstract}\n"
        prompt += f"关键词：{', '.join(example['keyword'])}\n"
        prompt += "这些是真实关键词吗？"
        
        if include_answer and 'label' in example:
            answer = '是' if example['label'] == '1' else '否'
            prompt += answer
            
        return prompt
    
    def format_zero_shot_prompt(self, example: Dict) -> str:
        """Zero-shot prompt"""
        prompt = self.get_task_description() + "\n\n"
        prompt += self.format_example(example, include_answer=False)
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip()
        
        # 检查肯定回答
        if any(yes in text[:20] for yes in ['是', '对', '正确', '真实']):
            if '不' not in text[:10] and '否' not in text[:10]:
                return '1'
        
        # 检查否定回答
        if any(no in text[:20] for no in ['否', '不是', '不对', '错误', '假的']):
            return '0'
            
        # 默认
        return '0'


class WSCTask(CLUEPretrainTask):
    """中文指代消解任务 - 适合预训练模型"""
    
    def __init__(self, data_dir: str):
        super().__init__('wsc', data_dir)
        
    def get_task_description(self) -> str:
        return "请判断句子中的代词是否指代特定的实体。"
        
    def load_data(self):
        """加载WSC数据"""
        train_path = self.data_dir / 'cluewsc2020_public' / 'train.json'
        dev_path = self.data_dir / 'cluewsc2020_public' / 'dev.json'
        
        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                self.train_data = [json.loads(line) for line in f]
        
        if dev_path.exists():
            with open(dev_path, 'r', encoding='utf-8') as f:
                self.dev_data = [json.loads(line) for line in f]
                
    def format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化WSC样本"""
        text = example['text']
        pronoun = example['target']['span2_text']
        span1_text = example['target']['span1_text']
        
        prompt = f"句子：{text}\n"
        prompt += f"问题：在这个句子中，"{pronoun}"指的是"{span1_text}"吗？\n"
        prompt += "回答："
        
        if include_answer and 'label' in example:
            answer = '是' if example['label'] == 'true' else '否'
            prompt += answer
            
        return prompt
    
    def format_zero_shot_prompt(self, example: Dict) -> str:
        """Zero-shot prompt"""
        return self.format_example(example, include_answer=False)
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip()
        
        # 检查肯定回答
        if any(yes in text[:20] for yes in ['是', '对', '正确', '指的是']):
            if '不' not in text[:10] and '否' not in text[:10]:
                return 'true'
        
        # 检查否定回答  
        if any(no in text[:20] for no in ['否', '不是', '不对', '错误', '不指']):
            return 'false'
            
        return 'false'


class OCNLITask(CLUEPretrainTask):
    """原生中文自然语言推理 - 适合预训练模型"""
    
    def __init__(self, data_dir: str):
        super().__init__('ocnli', data_dir)
        self.labels = ['entailment', 'neutral', 'contradiction']
        self.label_map = {
            'entailment': '蕴含',
            'neutral': '中立',
            'contradiction': '矛盾'
        }
        
    def get_task_description(self) -> str:
        return ("请判断两个句子之间的逻辑关系。\n"
                "蕴含：第二个句子是第一个句子的必然结果\n"
                "矛盾：两个句子相互矛盾\n"
                "中立：两个句子既不蕴含也不矛盾")
        
    def load_data(self):
        """加载OCNLI数据"""
        train_path = self.data_dir / 'ocnli_public' / 'train.json'
        dev_path = self.data_dir / 'ocnli_public' / 'dev.json'
        
        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                self.train_data = [json.loads(line) for line in f]
        
        if dev_path.exists():
            with open(dev_path, 'r', encoding='utf-8') as f:
                self.dev_data = [json.loads(line) for line in f]
                
    def format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化OCNLI样本"""
        prompt = f"句子1：{example['sentence1']}\n"
        prompt += f"句子2：{example['sentence2']}\n"
        prompt += "关系："
        
        if include_answer and 'label' in example:
            prompt += f"{self.label_map[example['label']]}"
        
        return prompt
    
    def format_zero_shot_prompt(self, example: Dict) -> str:
        """Zero-shot prompt"""
        prompt = self.get_task_description() + "\n\n"
        prompt += self.format_example(example, include_answer=False)
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip()
        
        # 直接匹配中文标签
        if '蕴含' in text[:20]:
            return 'entailment'
        elif '矛盾' in text[:20]:
            return 'contradiction'
        elif '中立' in text[:20]:
            return 'neutral'
        
        # 默认
        return 'neutral'


# 预训练模型适用的任务注册表
PRETRAIN_TASK_REGISTRY = {
    'cmnli': CMNLITask,      # 自然语言推理 - 最适合
    'afqmc': AFQMCTask,      # 语义相似度 - 很适合
    'csl': CSLTask,          # 关键词识别 - 适合
    'wsc': WSCTask,          # 指代消解 - 适合
    'ocnli': OCNLITask,      # 原生中文推理 - 很适合
}

# 任务难度和推荐设置
TASK_RECOMMENDATIONS = {
    'cmnli': {
        'difficulty': 'medium',
        'recommended_shots': 3,
        'zero_shot_capable': True,
        'description': '中文自然语言推理，评估逻辑推理能力'
    },
    'afqmc': {
        'difficulty': 'easy',
        'recommended_shots': 2,
        'zero_shot_capable': True,
        'description': '语义相似度判断，评估语义理解能力'
    },
    'csl': {
        'difficulty': 'medium',
        'recommended_shots': 3,
        'zero_shot_capable': True,
        'description': '关键词识别，评估文本理解和总结能力'
    },
    'wsc': {
        'difficulty': 'hard',
        'recommended_shots': 5,
        'zero_shot_capable': False,
        'description': '指代消解，评估上下文理解能力'
    },
    'ocnli': {
        'difficulty': 'medium',
        'recommended_shots': 3,
        'zero_shot_capable': True,
        'description': '原生中文推理，更自然的中文表达'
    }
}


def get_pretrain_task(task_name: str, data_dir: str) -> CLUEPretrainTask:
    """获取预训练评估任务实例"""
    if task_name not in PRETRAIN_TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available tasks for pretrain: {list(PRETRAIN_TASK_REGISTRY.keys())}")
    
    task_class = PRETRAIN_TASK_REGISTRY[task_name]
    return task_class(data_dir)


def get_recommended_tasks(difficulty: Optional[str] = None) -> List[str]:
    """获取推荐的任务列表"""
    if difficulty:
        return [task for task, info in TASK_RECOMMENDATIONS.items() 
                if info['difficulty'] == difficulty]
    return list(PRETRAIN_TASK_REGISTRY.keys())