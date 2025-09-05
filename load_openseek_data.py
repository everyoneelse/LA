#!/usr/bin/env python3
"""
OpenSeek-Pretrain-100B 数据加载器

该脚本用于加载 OpenSeek-Pretrain-100B 数据集，该数据集使用 .bin 和 .idx 文件格式
这种格式是 Megatron-LM 框架的标准索引数据格式
"""

import os
import numpy as np
import struct
from typing import List, Optional, Union


class IndexedDataset:
    """
    索引数据集类，用于加载 .bin 和 .idx 文件格式的数据
    """
    
    def __init__(self, path_prefix: str):
        """
        初始化索引数据集
        
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
        
        # 读取索引文件
        self._read_index()
        
        # 打开二进制文件
        self.bin_file = open(self.bin_path, 'rb')
        
    def _read_index(self):
        """读取索引文件获取元数据"""
        with open(self.idx_path, 'rb') as f:
            # 读取头部信息
            magic = struct.unpack('<Q', f.read(8))[0]
            version = struct.unpack('<Q', f.read(8))[0]
            
            # 读取数据类型和文档数量
            dtype_code = struct.unpack('<B', f.read(1))[0]
            self.dtype = self._get_dtype(dtype_code)
            
            # 读取文档数量
            self.doc_count = struct.unpack('<Q', f.read(8))[0]
            
            # 读取每个文档的长度
            self.doc_lengths = []
            for _ in range(self.doc_count):
                length = struct.unpack('<I', f.read(4))[0]
                self.doc_lengths.append(length)
            
            # 读取每个文档的偏移量
            self.doc_offsets = []
            for _ in range(self.doc_count):
                offset = struct.unpack('<Q', f.read(8))[0]
                self.doc_offsets.append(offset)
    
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
        
        # 转换为numpy数组
        data = np.frombuffer(data_bytes, dtype=self.dtype)
        return data
    
    def get_document_text(self, idx: int, tokenizer=None) -> Union[np.ndarray, str]:
        """
        获取文档的文本内容
        
        Args:
            idx: 文档索引
            tokenizer: 可选的分词器，用于将token转换为文本
            
        Returns:
            如果提供了tokenizer，返回解码后的文本字符串；否则返回token数组
        """
        tokens = self[idx]
        
        if tokenizer is not None:
            try:
                # 尝试解码tokens为文本
                text = tokenizer.decode(tokens.tolist())
                return text
            except Exception as e:
                print(f"Error decoding tokens: {e}")
                return tokens
        else:
            return tokens
    
    def get_batch(self, indices: List[int]) -> List[np.ndarray]:
        """
        批量获取多个文档的数据
        
        Args:
            indices: 文档索引列表
            
        Returns:
            文档数据列表
        """
        return [self[idx] for idx in indices]
    
    def close(self):
        """关闭文件句柄"""
        if hasattr(self, 'bin_file') and self.bin_file:
            self.bin_file.close()
    
    def __del__(self):
        """析构函数，确保文件被正确关闭"""
        self.close()


def load_openseek_dataset(data_dir: str, file_prefix: str = "018_00000_text_document") -> IndexedDataset:
    """
    加载 OpenSeek-Pretrain-100B 数据集
    
    Args:
        data_dir: 数据文件所在目录
        file_prefix: 文件前缀（默认为 "018_00000_text_document"）
        
    Returns:
        IndexedDataset 实例
    """
    path_prefix = os.path.join(data_dir, file_prefix)
    return IndexedDataset(path_prefix)


def main():
    """
    示例用法
    """
    # 设置数据路径
    data_dir = "/workspace"  # 根据您的实际路径调整
    file_prefix = "018_00000_text_document"
    
    try:
        # 加载数据集
        print("正在加载 OpenSeek-Pretrain-100B 数据集...")
        dataset = load_openseek_dataset(data_dir, file_prefix)
        
        print(f"数据集加载成功！")
        print(f"文档总数: {len(dataset)}")
        print(f"数据类型: {dataset.dtype}")
        
        # 获取第一个文档的数据
        if len(dataset) > 0:
            print("\n获取第一个文档的数据:")
            first_doc = dataset[0]
            print(f"第一个文档的长度: {len(first_doc)}")
            print(f"前10个token: {first_doc[:10]}")
            
            # 获取文档长度统计
            print(f"\n文档长度统计:")
            lengths = dataset.doc_lengths
            print(f"平均长度: {np.mean(lengths):.2f}")
            print(f"最小长度: {np.min(lengths)}")
            print(f"最大长度: {np.max(lengths)}")
        
        # 批量获取数据示例
        if len(dataset) >= 5:
            print("\n批量获取前5个文档:")
            batch_data = dataset.get_batch([0, 1, 2, 3, 4])
            for i, doc_data in enumerate(batch_data):
                print(f"文档 {i}: 长度={len(doc_data)}, 前5个token={doc_data[:5]}")
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        print("请检查数据文件路径和格式是否正确")
    
    finally:
        # 清理资源
        if 'dataset' in locals():
            dataset.close()


if __name__ == "__main__":
    main()