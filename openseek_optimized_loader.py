#!/usr/bin/env python3
"""
OpenSeek 优化数据加载器

专门针对检测到的 Megatron-LM 格式进行优化
"""

import os
import struct
import numpy as np
from typing import List, Optional, Union, Dict, Any


class OpenSeekOptimizedDataset:
    """
    针对 OpenSeek 数据格式优化的数据集类
    
    根据分析结果：
    - 格式: Megatron-LM 标准格式
    - 文档数量: 9,405,444
    - dtype_code: 0 (需要特殊处理)
    """
    
    def __init__(self, path_prefix: str):
        """
        初始化数据集
        
        Args:
            path_prefix: 数据文件路径前缀（不包含 .bin 或 .idx 扩展名）
        """
        self.path_prefix = path_prefix
        self.bin_path = path_prefix + '.bin'
        self.idx_path = path_prefix + '.idx'
        
        # 检查文件是否存在
        if not os.path.exists(self.bin_path):
            raise FileNotFoundError(f"Binary file not found: {self.bin_path}")
        if not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"Index file not found: {self.idx_path}")
        
        print(f"正在加载数据集: {path_prefix}")
        print(f"二进制文件大小: {os.path.getsize(self.bin_path) / (1024**3):.2f} GB")
        print(f"索引文件大小: {os.path.getsize(self.idx_path) / (1024**2):.2f} MB")
        
        # 解析索引文件
        self._parse_index_file()
        
        # 打开二进制文件（使用内存映射模式）
        self.bin_file = open(self.bin_path, 'rb')
        
        print(f"✓ 数据集加载成功")
        print(f"  文档数量: {self.doc_count:,}")
        print(f"  数据类型: {self.dtype}")
        print(f"  平均文档长度: {np.mean(self.doc_lengths):.1f} tokens")
    
    def _parse_index_file(self):
        """解析索引文件"""
        with open(self.idx_path, 'rb') as f:
            # 读取头部信息
            magic = struct.unpack('<Q', f.read(8))[0]
            version = struct.unpack('<Q', f.read(8))[0]
            dtype_code = struct.unpack('<B', f.read(1))[0]
            doc_count = struct.unpack('<Q', f.read(8))[0]
            
            print(f"索引文件头部信息:")
            print(f"  Magic: {magic}")
            print(f"  Version: {version}")
            print(f"  Data type code: {dtype_code}")
            print(f"  Document count: {doc_count:,}")
            
            # 处理特殊的 dtype_code = 0
            if dtype_code == 0:
                # 默认使用 int32，这是最常见的 token 格式
                self.dtype = np.int32
                print(f"  注意: dtype_code=0，使用默认类型 int32")
            else:
                self.dtype = self._get_dtype(dtype_code)
            
            self.doc_count = doc_count
            
            # 读取文档长度数组
            print("正在读取文档长度信息...")
            self.doc_lengths = []
            
            # 分批读取以节省内存
            batch_size = 100000
            for i in range(0, doc_count, batch_size):
                current_batch_size = min(batch_size, doc_count - i)
                length_data = f.read(current_batch_size * 4)  # 4 bytes per length
                lengths = struct.unpack(f'<{current_batch_size}I', length_data)
                self.doc_lengths.extend(lengths)
                
                if (i + current_batch_size) % 500000 == 0:
                    print(f"  已读取 {i + current_batch_size:,} / {doc_count:,} 文档长度")
            
            # 读取文档偏移量数组
            print("正在读取文档偏移量信息...")
            self.doc_offsets = []
            
            for i in range(0, doc_count, batch_size):
                current_batch_size = min(batch_size, doc_count - i)
                offset_data = f.read(current_batch_size * 8)  # 8 bytes per offset
                offsets = struct.unpack(f'<{current_batch_size}Q', offset_data)
                self.doc_offsets.extend(offsets)
                
                if (i + current_batch_size) % 500000 == 0:
                    print(f"  已读取 {i + current_batch_size:,} / {doc_count:,} 文档偏移量")
            
            print("✓ 索引信息加载完成")
    
    def _get_dtype(self, dtype_code: int):
        """根据类型码获取数据类型"""
        dtype_map = {
            1: np.uint8,
            2: np.int8,
            3: np.int16,
            4: np.int32,
            5: np.int64,
            6: np.float32,
            7: np.float64,
            8: np.uint16
        }
        return dtype_map.get(dtype_code, np.int32)
    
    def __len__(self):
        """返回数据集中文档的数量"""
        return self.doc_count
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """
        获取指定索引的文档数据
        
        Args:
            idx: 文档索引
            
        Returns:
            numpy数组，包含文档的token数据
        """
        if idx >= self.doc_count:
            raise IndexError(f"Index {idx} out of range for dataset with {self.doc_count} documents")
        
        # 获取文档的偏移量和长度
        offset = self.doc_offsets[idx]
        length = self.doc_lengths[idx]
        
        # 读取数据
        self.bin_file.seek(offset)
        bytes_to_read = length * np.dtype(self.dtype).itemsize
        data_bytes = self.bin_file.read(bytes_to_read)
        
        if len(data_bytes) < bytes_to_read:
            print(f"警告: 文档 {idx} 期望读取 {bytes_to_read} 字节，实际读取 {len(data_bytes)} 字节")
        
        # 转换为numpy数组
        data = np.frombuffer(data_bytes, dtype=self.dtype)
        return data
    
    def get_batch(self, indices: List[int]) -> List[np.ndarray]:
        """
        批量获取多个文档的数据
        
        Args:
            indices: 文档索引列表
            
        Returns:
            文档数据列表
        """
        return [self[idx] for idx in indices]
    
    def get_document_info(self, idx: int) -> Dict[str, Any]:
        """
        获取文档的详细信息
        
        Args:
            idx: 文档索引
            
        Returns:
            文档信息字典
        """
        if idx >= self.doc_count:
            raise IndexError(f"Index {idx} out of range")
        
        return {
            'index': idx,
            'offset': self.doc_offsets[idx],
            'length': self.doc_lengths[idx],
            'size_bytes': self.doc_lengths[idx] * np.dtype(self.dtype).itemsize
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        lengths = np.array(self.doc_lengths)
        
        return {
            'total_documents': self.doc_count,
            'data_type': str(self.dtype),
            'total_tokens': np.sum(lengths),
            'avg_doc_length': np.mean(lengths),
            'median_doc_length': np.median(lengths),
            'min_doc_length': np.min(lengths),
            'max_doc_length': np.max(lengths),
            'std_doc_length': np.std(lengths),
            'percentiles': {
                '25%': np.percentile(lengths, 25),
                '75%': np.percentile(lengths, 75),
                '90%': np.percentile(lengths, 90),
                '95%': np.percentile(lengths, 95),
                '99%': np.percentile(lengths, 99)
            }
        }
    
    def sample_documents(self, n_samples: int = 5, random_seed: int = 42) -> List[Dict]:
        """
        随机采样一些文档进行检查
        
        Args:
            n_samples: 采样数量
            random_seed: 随机种子
            
        Returns:
            采样文档信息列表
        """
        np.random.seed(random_seed)
        indices = np.random.choice(self.doc_count, min(n_samples, self.doc_count), replace=False)
        
        samples = []
        for idx in indices:
            try:
                data = self[idx]
                info = self.get_document_info(idx)
                info.update({
                    'first_10_tokens': data[:10].tolist() if len(data) > 0 else [],
                    'last_10_tokens': data[-10:].tolist() if len(data) > 10 else data.tolist(),
                    'token_range': f"{data.min()} - {data.max()}" if len(data) > 0 else "empty",
                    'actual_length': len(data)
                })
                samples.append(info)
            except Exception as e:
                samples.append({
                    'index': idx,
                    'error': str(e)
                })
        
        return samples
    
    def close(self):
        """关闭文件句柄"""
        if hasattr(self, 'bin_file') and self.bin_file:
            self.bin_file.close()
    
    def __del__(self):
        """析构函数"""
        self.close()


def load_openseek_optimized(path_prefix: str) -> OpenSeekOptimizedDataset:
    """
    加载 OpenSeek 数据集（优化版本）
    
    Args:
        path_prefix: 数据文件路径前缀
        
    Returns:
        OpenSeekOptimizedDataset 实例
    """
    return OpenSeekOptimizedDataset(path_prefix)


def main():
    """
    示例用法
    """
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 openseek_optimized_loader.py <data_path_prefix>")
        print("示例: python3 openseek_optimized_loader.py /path/to/007_00000_text_document")
        return
    
    path_prefix = sys.argv[1]
    
    try:
        print("=== OpenSeek 优化数据加载器 ===\n")
        
        # 加载数据集
        dataset = load_openseek_optimized(path_prefix)
        
        # 显示统计信息
        print(f"\n=== 数据集统计信息 ===")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            if key == 'percentiles':
                print(f"{key}:")
                for p_key, p_value in value.items():
                    print(f"  {p_key}: {p_value:.1f}")
            else:
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
        
        # 随机采样一些文档
        print(f"\n=== 随机采样文档 ===")
        samples = dataset.sample_documents(5)
        for i, sample in enumerate(samples):
            print(f"\n样本 {i+1}:")
            for key, value in sample.items():
                print(f"  {key}: {value}")
        
        print(f"\n=== 使用示例 ===")
        print("# 获取单个文档")
        print("doc_0 = dataset[0]")
        print("print(f'第一个文档长度: {len(doc_0)}')")
        print()
        print("# 批量获取文档")
        print("batch = dataset.get_batch([0, 1, 2, 3, 4])")
        print("print(f'批次大小: {len(batch)}')")
        print()
        print("# 获取文档信息")
        print("info = dataset.get_document_info(0)")
        print("print(info)")
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'dataset' in locals():
            dataset.close()


if __name__ == "__main__":
    main()