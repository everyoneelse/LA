#!/usr/bin/env python3
"""
Packed Data加载器
用于高效加载和使用打包后的训练数据
"""

import os
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Union
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PackedDataset(Dataset):
    """打包数据集"""
    
    def __init__(self, 
                 data_dir: str,
                 tokenizer: Union[str, AutoTokenizer],
                 max_length: int = 2048,
                 shuffle_files: bool = True):
        """
        初始化打包数据集
        
        Args:
            data_dir: 打包数据目录
            tokenizer: 分词器或分词器名称
            max_length: 最大序列长度
            shuffle_files: 是否打乱文件顺序
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        
        # 加载分词器
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        
        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载元数据
        self.metadata = self._load_metadata()
        
        # 获取数据文件列表
        self.data_files = self._get_data_files()
        if shuffle_files:
            random.shuffle(self.data_files)
        
        # 加载所有数据到内存（适用于中小规模数据）
        self.sequences = self._load_all_sequences()
        
        logger.info(f"加载了 {len(self.sequences)} 个序列")
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_data_files(self) -> List[Path]:
        """获取数据文件列表"""
        return list(self.data_dir.glob("packed_data_*.pkl"))
    
    def _load_all_sequences(self) -> List[Dict]:
        """加载所有序列到内存"""
        all_sequences = []
        
        for file_path in self.data_files:
            try:
                with open(file_path, 'rb') as f:
                    sequences = pickle.load(f)
                    all_sequences.extend(sequences)
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {e}")
                continue
        
        return all_sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sequence = self.sequences[idx]
        text = sequence['text']
        
        # 分词
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 准备输入和标签
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 对于语言模型，标签就是input_ids向右偏移一位
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # 忽略padding位置的损失
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'domains': sequence['domains'],
            'length': sequence['length']
        }

class StreamingPackedDataset(IterableDataset):
    """流式打包数据集（适用于大规模数据）"""
    
    def __init__(self,
                 data_dir: str,
                 tokenizer: Union[str, AutoTokenizer],
                 max_length: int = 2048,
                 shuffle_files: bool = True,
                 buffer_size: int = 10000):
        """
        初始化流式打包数据集
        
        Args:
            data_dir: 打包数据目录
            tokenizer: 分词器或分词器名称
            max_length: 最大序列长度
            shuffle_files: 是否打乱文件顺序
            buffer_size: 缓冲区大小
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.buffer_size = buffer_size
        
        # 加载分词器
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载元数据
        self.metadata = self._load_metadata()
        
        # 获取数据文件列表
        self.data_files = self._get_data_files()
        if shuffle_files:
            random.shuffle(self.data_files)
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_data_files(self) -> List[Path]:
        """获取数据文件列表"""
        return list(self.data_dir.glob("packed_data_*.pkl"))
    
    def _load_file_sequences(self, file_path: Path) -> List[Dict]:
        """加载单个文件的序列"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """迭代器"""
        buffer = []
        
        for file_path in self.data_files:
            try:
                sequences = self._load_file_sequences(file_path)
                random.shuffle(sequences)  # 打乱文件内序列
                
                for sequence in sequences:
                    buffer.append(sequence)
                    
                    # 当缓冲区满时，打乱并输出
                    if len(buffer) >= self.buffer_size:
                        random.shuffle(buffer)
                        for seq in buffer:
                            yield self._process_sequence(seq)
                        buffer = []
                
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        # 处理剩余的缓冲区数据
        if buffer:
            random.shuffle(buffer)
            for seq in buffer:
                yield self._process_sequence(seq)
    
    def _process_sequence(self, sequence: Dict) -> Dict[str, torch.Tensor]:
        """处理单个序列"""
        text = sequence['text']
        
        # 分词
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'domains': sequence['domains'],
            'length': sequence['length']
        }

class DomainAwareDataLoader:
    """领域感知的数据加载器"""
    
    def __init__(self,
                 dataset: Union[PackedDataset, StreamingPackedDataset],
                 batch_size: int = 8,
                 domain_balance: bool = True,
                 target_ratios: Optional[Dict[str, float]] = None):
        """
        初始化领域感知数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            domain_balance: 是否进行领域平衡
            target_ratios: 目标领域比例
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.domain_balance = domain_balance
        self.target_ratios = target_ratios or {'news': 0.6, 'code': 0.25, 'math': 0.15}
        
        if domain_balance and isinstance(dataset, PackedDataset):
            self.domain_indices = self._build_domain_indices()
    
    def _build_domain_indices(self) -> Dict[str, List[int]]:
        """构建领域索引"""
        domain_indices = {'news': [], 'code': [], 'math': []}
        
        for idx, sequence in enumerate(self.dataset.sequences):
            for domain in sequence['domains']:
                if domain in domain_indices:
                    domain_indices[domain].append(idx)
        
        return domain_indices
    
    def get_balanced_batch_indices(self) -> List[int]:
        """获取平衡的批次索引"""
        batch_indices = []
        
        for domain, ratio in self.target_ratios.items():
            if domain in self.domain_indices:
                domain_size = int(self.batch_size * ratio)
                if domain_size > 0:
                    available_indices = self.domain_indices[domain]
                    selected = random.sample(available_indices, 
                                           min(domain_size, len(available_indices)))
                    batch_indices.extend(selected)
        
        # 填充到目标批次大小
        while len(batch_indices) < self.batch_size:
            all_indices = []
            for indices in self.domain_indices.values():
                all_indices.extend(indices)
            if all_indices:
                batch_indices.append(random.choice(all_indices))
            else:
                break
        
        return batch_indices[:self.batch_size]
    
    def create_dataloader(self, **kwargs) -> DataLoader:
        """创建DataLoader"""
        if isinstance(self.dataset, StreamingPackedDataset):
            # 流式数据集
            return DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                **kwargs
            )
        else:
            # 普通数据集
            if self.domain_balance:
                # 自定义采样器实现领域平衡
                sampler = DomainBalancedSampler(
                    self.domain_indices, 
                    self.target_ratios,
                    self.batch_size
                )
                return DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    sampler=sampler,
                    **kwargs
                )
            else:
                return DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    **kwargs
                )

class DomainBalancedSampler(torch.utils.data.Sampler):
    """领域平衡采样器"""
    
    def __init__(self, 
                 domain_indices: Dict[str, List[int]],
                 target_ratios: Dict[str, float],
                 batch_size: int):
        self.domain_indices = domain_indices
        self.target_ratios = target_ratios
        self.batch_size = batch_size
        
        # 计算总长度
        self.total_samples = sum(len(indices) for indices in domain_indices.values())
    
    def __iter__(self):
        # 为每个epoch重新打乱域内数据
        shuffled_indices = {}
        for domain, indices in self.domain_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            shuffled_indices[domain] = shuffled
        
        # 生成平衡的批次
        domain_pointers = {domain: 0 for domain in shuffled_indices.keys()}
        
        while any(pointer < len(shuffled_indices[domain]) 
                 for domain, pointer in domain_pointers.items()):
            
            batch_indices = []
            
            # 按比例从各领域采样
            for domain, ratio in self.target_ratios.items():
                if domain in shuffled_indices:
                    domain_size = int(self.batch_size * ratio)
                    pointer = domain_pointers[domain]
                    available = shuffled_indices[domain][pointer:]
                    
                    selected = available[:domain_size]
                    batch_indices.extend(selected)
                    domain_pointers[domain] += len(selected)
            
            # 如果批次不满，随机填充
            while len(batch_indices) < self.batch_size:
                available_domains = [d for d, p in domain_pointers.items() 
                                   if p < len(shuffled_indices[d])]
                if not available_domains:
                    break
                
                domain = random.choice(available_domains)
                pointer = domain_pointers[domain]
                if pointer < len(shuffled_indices[domain]):
                    batch_indices.append(shuffled_indices[domain][pointer])
                    domain_pointers[domain] += 1
            
            # 打乱批次内的顺序
            random.shuffle(batch_indices)
            
            for idx in batch_indices:
                yield idx
    
    def __len__(self):
        return self.total_samples

def create_data_loader(data_dir: str,
                      tokenizer: Union[str, AutoTokenizer],
                      batch_size: int = 8,
                      max_length: int = 2048,
                      streaming: bool = False,
                      domain_balance: bool = True,
                      **kwargs) -> DataLoader:
    """
    便捷函数：创建数据加载器
    
    Args:
        data_dir: 数据目录
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        streaming: 是否使用流式数据集
        domain_balance: 是否进行领域平衡
        **kwargs: 其他DataLoader参数
    
    Returns:
        DataLoader实例
    """
    
    # 创建数据集
    if streaming:
        dataset = StreamingPackedDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=max_length
        )
    else:
        dataset = PackedDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_length=max_length
        )
    
    # 创建数据加载器
    loader_wrapper = DomainAwareDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        domain_balance=domain_balance
    )
    
    return loader_wrapper.create_dataloader(**kwargs)

# 使用示例
def example_usage():
    """使用示例"""
    
    # 创建数据加载器
    data_loader = create_data_loader(
        data_dir="./packed_data/",
        tokenizer="gpt2",
        batch_size=4,
        max_length=1024,
        streaming=False,
        domain_balance=True,
        num_workers=2
    )
    
    # 遍历数据
    for batch_idx, batch in enumerate(data_loader):
        print(f"批次 {batch_idx}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  domains: {batch['domains']}")
        print(f"  lengths: {batch['length']}")
        
        if batch_idx >= 2:  # 只显示前3个批次
            break

if __name__ == "__main__":
    example_usage()