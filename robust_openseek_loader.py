#!/usr/bin/env python3
"""
健壮的 OpenSeek 数据加载器

支持多种索引文件格式，自动检测和适配
"""

import os
import struct
import numpy as np
from typing import List, Optional, Union, Dict, Any


class RobustIndexedDataset:
    """
    健壮的索引数据集类，支持多种格式
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
        
        # 检测并读取索引文件
        self.format_info = self._detect_and_parse_index()
        
        # 打开二进制文件
        self.bin_file = open(self.bin_path, 'rb')
        
        print(f"✓ 数据集加载成功")
        print(f"  检测到的格式: {self.format_info.get('format_name', 'unknown')}")
        print(f"  文档数量: {self.doc_count}")
        print(f"  数据类型: {self.dtype}")
    
    def _detect_and_parse_index(self):
        """检测并解析索引文件格式"""
        file_size = os.path.getsize(self.idx_path)
        print(f"索引文件大小: {file_size:,} 字节")
        
        # 尝试不同的格式
        formats = [
            self._try_megatron_format,
            self._try_simple_format,
            self._try_huggingface_format,
            self._try_custom_format1,
            self._try_custom_format2,
            self._try_raw_offsets_format,
        ]
        
        for format_func in formats:
            try:
                result = format_func()
                if result:
                    return result
            except Exception as e:
                print(f"格式检测失败: {format_func.__name__}: {e}")
                continue
        
        raise ValueError("无法识别索引文件格式")
    
    def _try_megatron_format(self):
        """尝试 Megatron-LM 标准格式"""
        with open(self.idx_path, 'rb') as f:
            # Magic number (8 bytes)
            magic = struct.unpack('<Q', f.read(8))[0]
            
            # Version (8 bytes)  
            version = struct.unpack('<Q', f.read(8))[0]
            
            # Data type (1 byte)
            dtype_code = struct.unpack('<B', f.read(1))[0]
            
            # Document count (8 bytes)
            doc_count = struct.unpack('<Q', f.read(8))[0]
            
            # 验证数据合理性
            if doc_count <= 0 or doc_count > 100000000:  # 合理的文档数量范围
                raise ValueError(f"不合理的文档数量: {doc_count}")
            
            # 读取文档长度
            doc_lengths = []
            for _ in range(doc_count):
                length = struct.unpack('<I', f.read(4))[0]
                doc_lengths.append(length)
            
            # 读取文档偏移量
            doc_offsets = []
            for _ in range(doc_count):
                offset = struct.unpack('<Q', f.read(8))[0]
                doc_offsets.append(offset)
            
            self.doc_count = doc_count
            self.doc_lengths = doc_lengths
            self.doc_offsets = doc_offsets
            self.dtype = self._get_dtype(dtype_code)
            
            return {
                'format_name': 'Megatron-LM Standard',
                'magic': magic,
                'version': version,
                'dtype_code': dtype_code
            }
    
    def _try_simple_format(self):
        """尝试简化格式：只有文档数量和偏移"""
        with open(self.idx_path, 'rb') as f:
            file_size = os.path.getsize(self.idx_path)
            
            # 尝试4字节文档数量
            doc_count = struct.unpack('<I', f.read(4))[0]
            
            if doc_count <= 0 or doc_count > 10000000:
                raise ValueError(f"不合理的文档数量: {doc_count}")
            
            # 检查剩余字节是否足够存储偏移量
            remaining_bytes = file_size - 4
            expected_bytes = doc_count * 8  # 假设每个偏移量8字节
            
            if remaining_bytes == expected_bytes:
                # 读取偏移量
                doc_offsets = []
                for _ in range(doc_count):
                    offset = struct.unpack('<Q', f.read(8))[0]
                    doc_offsets.append(offset)
                
                # 计算文档长度
                doc_lengths = []
                for i in range(doc_count):
                    if i < doc_count - 1:
                        length = (doc_offsets[i + 1] - doc_offsets[i]) // 4  # 假设int32
                    else:
                        # 最后一个文档的长度
                        bin_size = os.path.getsize(self.bin_path)
                        length = (bin_size - doc_offsets[i]) // 4
                    doc_lengths.append(length)
                
                self.doc_count = doc_count
                self.doc_lengths = doc_lengths
                self.doc_offsets = doc_offsets
                self.dtype = np.int32  # 默认假设
                
                return {'format_name': 'Simple Offset Format'}
            else:
                raise ValueError("文件大小不匹配简化格式")
    
    def _try_huggingface_format(self):
        """尝试 HuggingFace datasets 格式"""
        with open(self.idx_path, 'rb') as f:
            # HuggingFace 可能使用不同的头部
            header = f.read(16)
            
            # 尝试解析为文档数量 (8字节) + 其他信息
            f.seek(0)
            doc_count = struct.unpack('<Q', f.read(8))[0]
            
            if 1 <= doc_count <= 50000000:
                # 假设接下来是偏移量数组
                doc_offsets = []
                for _ in range(doc_count + 1):  # +1 for end offset
                    offset = struct.unpack('<Q', f.read(8))[0]
                    doc_offsets.append(offset)
                
                # 计算长度
                doc_lengths = []
                for i in range(doc_count):
                    length = (doc_offsets[i + 1] - doc_offsets[i]) // 4
                    doc_lengths.append(length)
                
                self.doc_count = doc_count
                self.doc_lengths = doc_lengths
                self.doc_offsets = doc_offsets[:-1]  # 移除最后的结束偏移
                self.dtype = np.int32
                
                return {'format_name': 'HuggingFace Style'}
            else:
                raise ValueError("不符合 HuggingFace 格式")
    
    def _try_custom_format1(self):
        """尝试自定义格式1：大端序"""
        with open(self.idx_path, 'rb') as f:
            # 尝试大端序
            doc_count = struct.unpack('>I', f.read(4))[0]
            
            if 1 <= doc_count <= 10000000:
                # 读取偏移量（大端序）
                doc_offsets = []
                for _ in range(doc_count):
                    offset = struct.unpack('>Q', f.read(8))[0]
                    doc_offsets.append(offset)
                
                # 计算长度
                doc_lengths = []
                for i in range(doc_count):
                    if i < doc_count - 1:
                        length = (doc_offsets[i + 1] - doc_offsets[i]) // 4
                    else:
                        bin_size = os.path.getsize(self.bin_path)
                        length = (bin_size - doc_offsets[i]) // 4
                    doc_lengths.append(length)
                
                self.doc_count = doc_count
                self.doc_lengths = doc_lengths
                self.doc_offsets = doc_offsets
                self.dtype = np.int32
                
                return {'format_name': 'Custom Big Endian'}
            else:
                raise ValueError("不符合自定义格式1")
    
    def _try_custom_format2(self):
        """尝试自定义格式2：无头部，直接偏移量数组"""
        file_size = os.path.getsize(self.idx_path)
        
        # 假设全部都是8字节偏移量
        if file_size % 8 == 0:
            offset_count = file_size // 8
            
            with open(self.idx_path, 'rb') as f:
                doc_offsets = []
                for _ in range(offset_count):
                    offset = struct.unpack('<Q', f.read(8))[0]
                    doc_offsets.append(offset)
                
                # 文档数量比偏移量少1
                doc_count = offset_count - 1
                
                if doc_count > 0:
                    # 计算长度
                    doc_lengths = []
                    for i in range(doc_count):
                        length = (doc_offsets[i + 1] - doc_offsets[i]) // 4
                        doc_lengths.append(length)
                    
                    self.doc_count = doc_count
                    self.doc_lengths = doc_lengths
                    self.doc_offsets = doc_offsets[:-1]
                    self.dtype = np.int32
                    
                    return {'format_name': 'Raw Offsets Array'}
        
        raise ValueError("不符合自定义格式2")
    
    def _try_raw_offsets_format(self):
        """尝试原始偏移量格式：4字节偏移量"""
        file_size = os.path.getsize(self.idx_path)
        
        # 假设全部都是4字节偏移量
        if file_size % 4 == 0:
            offset_count = file_size // 4
            
            with open(self.idx_path, 'rb') as f:
                doc_offsets = []
                for _ in range(offset_count):
                    offset = struct.unpack('<I', f.read(4))[0]
                    doc_offsets.append(offset)
                
                doc_count = offset_count - 1
                
                if doc_count > 0:
                    doc_lengths = []
                    for i in range(doc_count):
                        length = (doc_offsets[i + 1] - doc_offsets[i]) // 4
                        doc_lengths.append(length)
                    
                    self.doc_count = doc_count
                    self.doc_lengths = doc_lengths
                    self.doc_offsets = doc_offsets[:-1]
                    self.dtype = np.int32
                    
                    return {'format_name': 'Raw 4-byte Offsets'}
        
        raise ValueError("不符合原始偏移量格式")
    
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
            print(f"警告: 期望读取 {bytes_to_read} 字节，实际读取 {len(data_bytes)} 字节")
        
        # 转换为numpy数组
        data = np.frombuffer(data_bytes, dtype=self.dtype)
        return data
    
    def get_info(self):
        """获取数据集信息"""
        info = {
            'format': self.format_info.get('format_name', 'unknown'),
            'doc_count': self.doc_count,
            'dtype': str(self.dtype),
            'bin_file_size': os.path.getsize(self.bin_path),
            'idx_file_size': os.path.getsize(self.idx_path),
        }
        
        if self.doc_lengths:
            info.update({
                'avg_doc_length': np.mean(self.doc_lengths),
                'min_doc_length': np.min(self.doc_lengths),
                'max_doc_length': np.max(self.doc_lengths),
            })
        
        return info
    
    def close(self):
        """关闭文件句柄"""
        if hasattr(self, 'bin_file') and self.bin_file:
            self.bin_file.close()
    
    def __del__(self):
        """析构函数，确保文件被正确关闭"""
        self.close()


def load_openseek_dataset_robust(data_dir: str, file_prefix: str = "018_00000_text_document"):
    """
    使用健壮加载器加载 OpenSeek 数据集
    
    Args:
        data_dir: 数据文件所在目录
        file_prefix: 文件前缀
        
    Returns:
        RobustIndexedDataset 实例
    """
    path_prefix = os.path.join(data_dir, file_prefix)
    return RobustIndexedDataset(path_prefix)


def main():
    """
    示例用法
    """
    import sys
    
    # 从命令行参数获取数据目录
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "."
    
    if len(sys.argv) > 2:
        file_prefix = sys.argv[2]
    else:
        file_prefix = "018_00000_text_document"
    
    try:
        print("=== 健壮的 OpenSeek 数据加载器 ===\n")
        print(f"数据目录: {data_dir}")
        print(f"文件前缀: {file_prefix}")
        
        # 加载数据集
        dataset = load_openseek_dataset_robust(data_dir, file_prefix)
        
        # 显示信息
        info = dataset.get_info()
        print(f"\n数据集信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 获取样本数据
        if len(dataset) > 0:
            print(f"\n样本数据:")
            for i in range(min(3, len(dataset))):
                try:
                    doc_data = dataset[i]
                    print(f"文档 {i}:")
                    print(f"  长度: {len(doc_data)}")
                    print(f"  前10个token: {doc_data[:10].tolist()}")
                    if len(doc_data) > 0:
                        print(f"  值范围: {doc_data.min()} - {doc_data.max()}")
                    print()
                except Exception as e:
                    print(f"  读取文档 {i} 时出错: {e}")
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        print("\n建议:")
        print("1. 检查文件路径是否正确")
        print("2. 运行 analyze_data_format.py 分析文件格式")
        print("3. 确认文件是否完整下载")
    
    finally:
        if 'dataset' in locals():
            dataset.close()


if __name__ == "__main__":
    main()