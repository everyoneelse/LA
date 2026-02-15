"""
CLUE任务定义和数据处理
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

class CLUETask:
    """CLUE任务基类"""
    
    def __init__(self, task_name: str, data_dir: str):
        self.task_name = task_name
        self.data_dir = Path(data_dir) / task_name
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        
    def load_data(self):
        """加载数据集"""
        raise NotImplementedError
        
    def format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化单个样本用于模型输入"""
        raise NotImplementedError
        
    def extract_answer(self, text: str) -> str:
        """从模型输出中提取答案"""
        raise NotImplementedError
        
    def compute_metric(self, predictions: List, references: List) -> Dict:
        """计算评估指标"""
        raise NotImplementedError


class AFQMCTask(CLUETask):
    """蚂蚁金融语义相似度任务"""
    
    def __init__(self, data_dir: str):
        super().__init__('afqmc', data_dir)
        self.labels = ['0', '1']  # 0: 不相似, 1: 相似
        
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
        prompt = f"判断下面两个句子的语义是否相似。\n"
        prompt += f"句子1: {example['sentence1']}\n"
        prompt += f"句子2: {example['sentence2']}\n"
        prompt += "选项:\nA. 不相似\nB. 相似\n"
        
        if include_answer and 'label' in example:
            answer = 'B' if example['label'] == '1' else 'A'
            prompt += f"答案: {answer}\n"
        else:
            prompt += "答案: "
            
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip().upper()
        if 'A' in text[:10]:
            return '0'
        elif 'B' in text[:10]:
            return '1'
        else:
            # 尝试直接匹配相似/不相似
            if '相似' in text and '不' not in text[:text.index('相似')]:
                return '1'
            else:
                return '0'
                
    def compute_metric(self, predictions: List, references: List) -> Dict:
        """计算准确率"""
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        accuracy = correct / len(predictions) if predictions else 0
        return {'accuracy': accuracy}


class TNEWSTask(CLUETask):
    """今日头条新闻分类任务"""
    
    def __init__(self, data_dir: str):
        super().__init__('tnews', data_dir)
        self.label_map = {
            '100': '故事',
            '101': '文化',
            '102': '娱乐',
            '103': '体育',
            '104': '财经',
            '106': '房产',
            '107': '汽车',
            '108': '教育',
            '109': '科技',
            '110': '军事',
            '112': '旅游',
            '113': '国际',
            '114': '股票',
            '115': '农业',
            '116': '游戏'
        }
        
    def load_data(self):
        """加载TNEWS数据"""
        train_path = self.data_dir / 'tnews_public' / 'train.json'
        dev_path = self.data_dir / 'tnews_public' / 'dev.json'
        
        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                self.train_data = [json.loads(line) for line in f]
        
        if dev_path.exists():
            with open(dev_path, 'r', encoding='utf-8') as f:
                self.dev_data = [json.loads(line) for line in f]
                
    def format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化TNEWS样本"""
        prompt = f"请对下面的新闻进行分类。\n"
        prompt += f"新闻: {example['sentence']}\n"
        prompt += "类别选项:\n"
        
        # 生成选项
        options = list(self.label_map.keys())
        for i, label_id in enumerate(options):
            option_letter = chr(65 + i)  # A, B, C, ...
            prompt += f"{option_letter}. {self.label_map[label_id]}\n"
        
        if include_answer and 'label' in example:
            idx = options.index(example['label'])
            answer = chr(65 + idx)
            prompt += f"答案: {answer}\n"
        else:
            prompt += "答案: "
            
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip().upper()
        
        # 尝试提取字母选项
        for i, label_id in enumerate(self.label_map.keys()):
            option_letter = chr(65 + i)
            if option_letter in text[:10]:
                return label_id
        
        # 尝试匹配类别名称
        for label_id, label_name in self.label_map.items():
            if label_name in text:
                return label_id
                
        return list(self.label_map.keys())[0]  # 默认返回第一个
        
    def compute_metric(self, predictions: List, references: List) -> Dict:
        """计算准确率"""
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        accuracy = correct / len(predictions) if predictions else 0
        return {'accuracy': accuracy}


class CMNLITask(CLUETask):
    """中文自然语言推理任务"""
    
    def __init__(self, data_dir: str):
        super().__init__('cmnli', data_dir)
        self.labels = ['entailment', 'neutral', 'contradiction']
        self.label_map = {
            'entailment': '蕴含',
            'neutral': '中立',
            'contradiction': '矛盾'
        }
        
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
        prompt = f"判断下面两个句子之间的逻辑关系。\n"
        prompt += f"前提: {example['sentence1']}\n"
        prompt += f"假设: {example['sentence2']}\n"
        prompt += "关系:\nA. 蕴含\nB. 中立\nC. 矛盾\n"
        
        if include_answer and 'label' in example:
            label_to_option = {'entailment': 'A', 'neutral': 'B', 'contradiction': 'C'}
            answer = label_to_option.get(example['label'], 'B')
            prompt += f"答案: {answer}\n"
        else:
            prompt += "答案: "
            
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip().upper()
        
        option_to_label = {'A': 'entailment', 'B': 'neutral', 'C': 'contradiction'}
        
        # 尝试提取字母选项
        for option, label in option_to_label.items():
            if option in text[:10]:
                return label
        
        # 尝试匹配中文标签
        if '蕴含' in text:
            return 'entailment'
        elif '矛盾' in text:
            return 'contradiction'
        else:
            return 'neutral'
            
    def compute_metric(self, predictions: List, references: List) -> Dict:
        """计算准确率"""
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        accuracy = correct / len(predictions) if predictions else 0
        return {'accuracy': accuracy}


class IFLYTEKTask(CLUETask):
    """科大讯飞长文本分类任务"""
    
    def __init__(self, data_dir: str):
        super().__init__('iflytek', data_dir)
        # 119个应用类别，这里只列举部分
        self.label_map = {}  # 将在load_data时从标签文件加载
        
    def load_data(self):
        """加载IFLYTEK数据"""
        train_path = self.data_dir / 'iflytek_public' / 'train.json'
        dev_path = self.data_dir / 'iflytek_public' / 'dev.json'
        labels_path = self.data_dir / 'iflytek_public' / 'labels.json'
        
        # 加载标签映射
        if labels_path.exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.label_map = json.load(f)
        
        if train_path.exists():
            with open(train_path, 'r', encoding='utf-8') as f:
                self.train_data = [json.loads(line) for line in f]
        
        if dev_path.exists():
            with open(dev_path, 'r', encoding='utf-8') as f:
                self.dev_data = [json.loads(line) for line in f]
                
    def format_example(self, example: Dict, include_answer: bool = True) -> str:
        """格式化IFLYTEK样本 - 使用简化版本"""
        prompt = f"请对下面的应用描述进行分类。\n"
        prompt += f"应用描述: {example['sentence']}\n"
        
        # 由于类别太多，使用直接预测的方式
        if include_answer and 'label' in example:
            label_name = self.label_map.get(str(example['label']), '未知')
            prompt += f"应用类别: {label_name}\n"
        else:
            prompt += "应用类别: "
            
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        # 尝试匹配标签名称
        for label_id, label_name in self.label_map.items():
            if label_name in text:
                return label_id
                
        return '0'  # 默认返回第一个类别
        
    def compute_metric(self, predictions: List, references: List) -> Dict:
        """计算准确率"""
        correct = sum(1 for p, r in zip(predictions, references) if str(p) == str(r))
        accuracy = correct / len(predictions) if predictions else 0
        return {'accuracy': accuracy}


class CSLTask(CLUETask):
    """中文科学文献关键词识别任务"""
    
    def __init__(self, data_dir: str):
        super().__init__('csl', data_dir)
        
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
        prompt = f"判断关键词是否为论文的真实关键词。\n"
        prompt += f"论文标题: {example['title']}\n"
        prompt += f"论文摘要: {example['abst']}\n"
        prompt += f"关键词列表: {', '.join(example['keyword'])}\n"
        prompt += "这些关键词是否为真实关键词？\nA. 否\nB. 是\n"
        
        if include_answer and 'label' in example:
            answer = 'B' if example['label'] == '1' else 'A'
            prompt += f"答案: {answer}\n"
        else:
            prompt += "答案: "
            
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip().upper()
        if 'A' in text[:10] or '否' in text[:10]:
            return '0'
        elif 'B' in text[:10] or '是' in text[:10]:
            return '1'
        else:
            return '0'
            
    def compute_metric(self, predictions: List, references: List) -> Dict:
        """计算准确率"""
        correct = sum(1 for p, r in zip(predictions, references) if str(p) == str(r))
        accuracy = correct / len(predictions) if predictions else 0
        return {'accuracy': accuracy}


class WSCTask(CLUETask):
    """中文指代消解任务"""
    
    def __init__(self, data_dir: str):
        super().__init__('wsc', data_dir)
        
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
        
        prompt = f"在下面的句子中，代词"{pronoun}"指代的是"{span1_text}"吗？\n"
        prompt += f"句子: {text}\n"
        prompt += "选项:\nA. 否\nB. 是\n"
        
        if include_answer and 'label' in example:
            answer = 'B' if example['label'] == 'true' else 'A'
            prompt += f"答案: {answer}\n"
        else:
            prompt += "答案: "
            
        return prompt
    
    def extract_answer(self, text: str) -> str:
        """提取答案"""
        text = text.strip().upper()
        if 'A' in text[:10] or '否' in text[:10]:
            return 'false'
        elif 'B' in text[:10] or '是' in text[:10]:
            return 'true'
        else:
            return 'false'
            
    def compute_metric(self, predictions: List, references: List) -> Dict:
        """计算准确率"""
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        accuracy = correct / len(predictions) if predictions else 0
        return {'accuracy': accuracy}


# 任务注册表
TASK_REGISTRY = {
    'afqmc': AFQMCTask,
    'tnews': TNEWSTask,
    'cmnli': CMNLITask,
    'iflytek': IFLYTEKTask,
    'csl': CSLTask,
    'wsc': WSCTask,
}

def get_task(task_name: str, data_dir: str) -> CLUETask:
    """获取任务实例"""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_REGISTRY.keys())}")
    
    task_class = TASK_REGISTRY[task_name]
    return task_class(data_dir)