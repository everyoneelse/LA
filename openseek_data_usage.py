#!/usr/bin/env python3
"""
OpenSeek-Pretrain-100B 数据使用指南和示例

该脚本提供了完整的数据加载、处理和使用示例
"""

import os
import numpy as np
from typing import List, Optional, Iterator, Tuple


def install_dependencies():
    """
    安装必要的依赖包
    """
    import subprocess
    import sys
    
    packages = [
        'numpy',
        'torch',  # 如果需要 PyTorch 集成
        'transformers',  # 如果需要使用 tokenizer
    ]
    
    print("正在安装依赖包...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 安装失败")


def create_data_iterator(dataset, batch_size: int = 1, shuffle: bool = False) -> Iterator[List]:
    """
    创建数据迭代器
    
    Args:
        dataset: 数据集对象
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
    Yields:
        批次数据
    """
    indices = list(range(len(dataset)))
    
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = []
        
        for idx in batch_indices:
            try:
                data = dataset[idx]
                batch_data.append(data)
            except Exception as e:
                print(f"读取索引 {idx} 时出错: {e}")
                continue
        
        if batch_data:
            yield batch_data


def analyze_dataset(dataset):
    """
    分析数据集的统计信息
    
    Args:
        dataset: 数据集对象
    """
    print("=== 数据集分析 ===")
    print(f"文档总数: {len(dataset)}")
    
    if hasattr(dataset, 'doc_lengths'):
        lengths = dataset.doc_lengths
        print(f"文档长度统计:")
        print(f"  平均长度: {np.mean(lengths):.2f}")
        print(f"  中位数长度: {np.median(lengths):.2f}")
        print(f"  最小长度: {np.min(lengths)}")
        print(f"  最大长度: {np.max(lengths)}")
        print(f"  标准差: {np.std(lengths):.2f}")
        
        # 长度分布
        print(f"\n长度分布:")
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(lengths, p)
            print(f"  {p}%: {value:.0f}")
    
    # 采样一些数据进行分析
    print(f"\n=== 数据采样分析 ===")
    sample_size = min(10, len(dataset))
    
    for i in range(sample_size):
        try:
            data = dataset[i]
            print(f"文档 {i}:")
            print(f"  类型: {type(data)}")
            print(f"  形状: {data.shape if hasattr(data, 'shape') else len(data)}")
            print(f"  数据类型: {data.dtype if hasattr(data, 'dtype') else 'unknown'}")
            
            if hasattr(data, '__len__') and len(data) > 0:
                print(f"  前10个值: {data[:10]}")
                print(f"  值范围: {np.min(data)} - {np.max(data)}")
            print()
            
        except Exception as e:
            print(f"  读取文档 {i} 时出错: {e}")


def convert_to_pytorch_dataset(dataset):
    """
    将数据集转换为 PyTorch Dataset
    """
    try:
        import torch
        from torch.utils.data import Dataset
        
        class OpenSeekDataset(Dataset):
            def __init__(self, indexed_dataset):
                self.dataset = indexed_dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                data = self.dataset[idx]
                # 转换为 PyTorch tensor
                return torch.tensor(data, dtype=torch.long)
        
        pytorch_dataset = OpenSeekDataset(dataset)
        print("✓ 成功转换为 PyTorch Dataset")
        return pytorch_dataset
        
    except ImportError:
        print("✗ 未安装 PyTorch，跳过转换")
        return None
    except Exception as e:
        print(f"✗ 转换为 PyTorch Dataset 时出错: {e}")
        return None


def create_dataloader_example(dataset):
    """
    创建 DataLoader 示例
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        
        pytorch_dataset = convert_to_pytorch_dataset(dataset)
        if pytorch_dataset is None:
            return None
        
        # 创建 DataLoader
        dataloader = DataLoader(
            pytorch_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            collate_fn=lambda batch: torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=0
            )
        )
        
        print("✓ 成功创建 PyTorch DataLoader")
        
        # 测试加载一个批次
        try:
            batch = next(iter(dataloader))
            print(f"批次形状: {batch.shape}")
            print(f"批次数据类型: {batch.dtype}")
            return dataloader
        except Exception as e:
            print(f"测试 DataLoader 时出错: {e}")
            return dataloader
            
    except ImportError:
        print("✗ 未安装 PyTorch，无法创建 DataLoader")
        return None
    except Exception as e:
        print(f"✗ 创建 DataLoader 时出错: {e}")
        return None


def main():
    """
    主函数：完整的使用示例
    """
    print("=== OpenSeek-Pretrain-100B 数据使用指南 ===\n")
    
    # 1. 检查并安装依赖
    print("1. 检查依赖...")
    try:
        import numpy as np
        print("✓ numpy 已安装")
    except ImportError:
        print("✗ numpy 未安装，正在安装...")
        install_dependencies()
    
    # 2. 加载数据集
    print("\n2. 加载数据集...")
    try:
        # 首先尝试使用简化加载器
        from simple_data_loader import load_with_custom_loader
        dataset = load_with_custom_loader()
        
        if dataset is None:
            print("数据加载失败，请检查文件路径")
            return
            
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return
    
    # 3. 分析数据集
    print("\n3. 分析数据集...")
    analyze_dataset(dataset)
    
    # 4. 创建数据迭代器示例
    print("\n4. 创建数据迭代器...")
    iterator = create_data_iterator(dataset, batch_size=2, shuffle=True)
    
    print("迭代器创建成功，获取前3个批次:")
    for i, batch in enumerate(iterator):
        if i >= 3:  # 只显示前3个批次
            break
        print(f"批次 {i+1}: {len(batch)} 个样本")
        for j, sample in enumerate(batch):
            print(f"  样本 {j+1}: 长度={len(sample)}, 前5个值={sample[:5]}")
    
    # 5. PyTorch 集成示例
    print("\n5. PyTorch 集成...")
    dataloader = create_dataloader_example(dataset)
    
    # 6. 使用建议
    print("\n=== 使用建议 ===")
    print("1. 数据加载:")
    print("   - 使用 load_openseek_data.py 中的 IndexedDataset 类")
    print("   - 或者使用 simple_data_loader.py 中的简化版本")
    print()
    print("2. 数据处理:")
    print("   - 数据已经是 token 格式，可以直接用于训练")
    print("   - 如需转换为文本，需要对应的 tokenizer")
    print()
    print("3. 训练集成:")
    print("   - 可以转换为 PyTorch Dataset 进行训练")
    print("   - 支持批量加载和数据打乱")
    print()
    print("4. 性能优化:")
    print("   - 使用内存映射模式处理大文件")
    print("   - 合理设置批次大小和工作进程数")
    
    # 清理资源
    if dataset:
        dataset.close()


if __name__ == "__main__":
    main()