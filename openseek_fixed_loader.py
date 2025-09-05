#!/usr/bin/env python3
"""
OpenSeek 修复版数据加载器

修复缓冲区大小不匹配问题，添加更健壮的解析逻辑
"""

import os
import struct
import numpy as np
from typing import List, Optional, Union, Dict, Any


class OpenSeekFixedDataset:
    """
    修复版的 OpenSeek 数据集类
    
    添加了更严格的缓冲区检查和错误处理
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
        self._parse_index_file_safe()
        
        # 打开二进制文件
        self.bin_file = open(self.bin_path, 'rb')
        
        print(f"✓ 数据集加载成功")
        print(f"  文档数量: {self.doc_count:,}")
        print(f"  数据类型: {self.dtype}")
        if hasattr(self, 'doc_lengths') and self.doc_lengths:
            print(f"  平均文档长度: {np.mean(self.doc_lengths):.1f} tokens")
    
    def _parse_index_file_safe(self):
        """安全解析索引文件，添加缓冲区检查"""
        idx_file_size = os.path.getsize(self.idx_path)
        
        with open(self.idx_path, 'rb') as f:
            # 读取头部信息
            if idx_file_size < 25:  # 至少需要25字节的头部
                raise ValueError(f"索引文件太小: {idx_file_size} 字节")
            
            magic = struct.unpack('<Q', f.read(8))[0]
            version = struct.unpack('<Q', f.read(8))[0]
            dtype_code = struct.unpack('<B', f.read(1))[0]
            doc_count = struct.unpack('<Q', f.read(8))[0]
            
            print(f"索引文件头部信息:")
            print(f"  Magic: {magic}")
            print(f"  Version: {version}")
            print(f"  Data type code: {dtype_code}")
            print(f"  Document count: {doc_count:,}")
            
            # 验证文档数量的合理性
            if doc_count <= 0 or doc_count > 100000000:
                raise ValueError(f"不合理的文档数量: {doc_count}")
            
            # 处理特殊的 dtype_code = 0
            if dtype_code == 0:
                self.dtype = np.int32
                print(f"  注意: dtype_code=0，使用默认类型 int32")
            else:
                self.dtype = self._get_dtype(dtype_code)
            
            self.doc_count = doc_count
            
            # 计算期望的文件大小
            expected_size = 25 + (doc_count * 4) + (doc_count * 8)  # 头部 + 长度数组 + 偏移数组
            print(f"  期望索引文件大小: {expected_size:,} 字节")
            print(f"  实际索引文件大小: {idx_file_size:,} 字节")
            
            if idx_file_size < expected_size:
                print(f"  警告: 索引文件大小不足，可能格式不同")
                # 尝试其他解析方法
                return self._parse_alternative_format(f, doc_count)
            
            # 当前文件位置
            current_pos = f.tell()
            print(f"  当前文件位置: {current_pos}")
            
            # 计算剩余字节数
            remaining_bytes = idx_file_size - current_pos
            print(f"  剩余字节数: {remaining_bytes:,}")
            
            # 读取文档长度数组
            print("正在读取文档长度信息...")
            self.doc_lengths = []
            
            # 安全的分批读取
            batch_size = min(100000, doc_count)  # 限制批次大小
            bytes_per_length = 4
            
            for i in range(0, doc_count, batch_size):
                current_batch_size = min(batch_size, doc_count - i)
                bytes_needed = current_batch_size * bytes_per_length
                
                # 检查是否有足够的字节
                remaining_in_file = idx_file_size - f.tell()
                if remaining_in_file < bytes_needed:
                    print(f"  警告: 在位置 {f.tell()} 需要 {bytes_needed} 字节，但只剩 {remaining_in_file} 字节")
                    # 调整批次大小
                    current_batch_size = remaining_in_file // bytes_per_length
                    bytes_needed = current_batch_size * bytes_per_length
                    
                    if current_batch_size <= 0:
                        print(f"  错误: 无法读取更多长度数据")
                        break
                
                try:
                    length_data = f.read(bytes_needed)
                    if len(length_data) < bytes_needed:
                        print(f"  警告: 期望读取 {bytes_needed} 字节，实际读取 {len(length_data)} 字节")
                        current_batch_size = len(length_data) // bytes_per_length
                    
                    if current_batch_size > 0:
                        lengths = struct.unpack(f'<{current_batch_size}I', length_data[:current_batch_size * bytes_per_length])
                        self.doc_lengths.extend(lengths)
                    
                    if (i + current_batch_size) % 500000 == 0:
                        print(f"  已读取 {i + current_batch_size:,} / {doc_count:,} 文档长度")
                        
                except struct.error as e:
                    print(f"  结构解析错误: {e}")
                    print(f"  尝试读取的字节数: {bytes_needed}")
                    print(f"  实际数据长度: {len(length_data)}")
                    break
            
            # 更新实际读取的文档数量
            actual_doc_count = len(self.doc_lengths)
            if actual_doc_count < doc_count:
                print(f"  警告: 只读取了 {actual_doc_count} / {doc_count} 个文档长度")
                self.doc_count = actual_doc_count
            
            # 读取文档偏移量数组
            print("正在读取文档偏移量信息...")
            self.doc_offsets = []
            
            bytes_per_offset = 8
            for i in range(0, self.doc_count, batch_size):
                current_batch_size = min(batch_size, self.doc_count - i)
                bytes_needed = current_batch_size * bytes_per_offset
                
                # 检查剩余字节
                remaining_in_file = idx_file_size - f.tell()
                if remaining_in_file < bytes_needed:
                    print(f"  警告: 偏移量数据不足，需要 {bytes_needed} 字节，剩余 {remaining_in_file} 字节")
                    current_batch_size = remaining_in_file // bytes_per_offset
                    bytes_needed = current_batch_size * bytes_per_offset
                    
                    if current_batch_size <= 0:
                        print(f"  错误: 无法读取更多偏移量数据")
                        break
                
                try:
                    offset_data = f.read(bytes_needed)
                    if len(offset_data) < bytes_needed:
                        print(f"  警告: 期望读取 {bytes_needed} 字节，实际读取 {len(offset_data)} 字节")
                        current_batch_size = len(offset_data) // bytes_per_offset
                    
                    if current_batch_size > 0:
                        offsets = struct.unpack(f'<{current_batch_size}Q', offset_data[:current_batch_size * bytes_per_offset])
                        self.doc_offsets.extend(offsets)
                    
                    if (i + current_batch_size) % 500000 == 0:
                        print(f"  已读取 {i + current_batch_size:,} / {self.doc_count:,} 文档偏移量")
                        
                except struct.error as e:
                    print(f"  偏移量解析错误: {e}")
                    break
            
            # 最终验证
            final_doc_count = min(len(self.doc_lengths), len(self.doc_offsets))
            if final_doc_count < self.doc_count:
                print(f"  最终调整文档数量: {self.doc_count} -> {final_doc_count}")
                self.doc_count = final_doc_count
                self.doc_lengths = self.doc_lengths[:final_doc_count]
                self.doc_offsets = self.doc_offsets[:final_doc_count]
            
            print("✓ 索引信息加载完成")
    
    def _parse_alternative_format(self, f, doc_count):
        """尝试解析其他可能的格式"""
        print("尝试解析其他格式...")
        
        # 重置文件指针到头部之后
        f.seek(25)
        
        # 尝试直接读取所有剩余数据
        remaining_data = f.read()
        print(f"剩余数据长度: {len(remaining_data)} 字节")
        
        # 尝试解析为交替的长度和偏移量
        if len(remaining_data) >= doc_count * 12:  # 4字节长度 + 8字节偏移量
            print("尝试解析为交替的长度/偏移量格式...")
            self.doc_lengths = []
            self.doc_offsets = []
            
            for i in range(doc_count):
                offset = i * 12
                if offset + 12 <= len(remaining_data):
                    length = struct.unpack('<I', remaining_data[offset:offset+4])[0]
                    file_offset = struct.unpack('<Q', remaining_data[offset+4:offset+12])[0]
                    self.doc_lengths.append(length)
                    self.doc_offsets.append(file_offset)
                else:
                    break
            
            self.doc_count = len(self.doc_lengths)
            print(f"成功解析 {self.doc_count} 个文档的信息")
            return
        
        raise ValueError("无法解析索引文件格式")
    
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
        """获取指定索引的文档数据"""
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
        """批量获取多个文档的数据"""
        return [self[idx] for idx in indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not hasattr(self, 'doc_lengths') or not self.doc_lengths:
            return {'error': 'No document length information available'}
        
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
        }
    
    def close(self):
        """关闭文件句柄"""
        if hasattr(self, 'bin_file') and self.bin_file:
            self.bin_file.close()
    
    def __del__(self):
        """析构函数"""
        self.close()


def load_openseek_fixed(path_prefix: str) -> OpenSeekFixedDataset:
    """
    加载 OpenSeek 数据集（修复版本）
    
    Args:
        path_prefix: 数据文件路径前缀
        
    Returns:
        OpenSeekFixedDataset 实例
    """
    return OpenSeekFixedDataset(path_prefix)


def main():
    """示例用法"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 openseek_fixed_loader.py <data_path_prefix>")
        print("示例: python3 openseek_fixed_loader.py /path/to/007_00000_text_document")
        return
    
    path_prefix = sys.argv[1]
    
    try:
        print("=== OpenSeek 修复版数据加载器 ===\n")
        
        # 加载数据集
        dataset = load_openseek_fixed(path_prefix)
        
        # 显示统计信息
        print(f"\n=== 数据集统计信息 ===")
        stats = dataset.get_statistics()
        if 'error' not in stats:
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
        else:
            print(f"统计信息不可用: {stats['error']}")
        
        # 测试读取几个文档
        print(f"\n=== 测试读取文档 ===")
        test_indices = [0, min(100, len(dataset)-1), len(dataset)-1]
        
        for idx in test_indices:
            if idx < len(dataset):
                try:
                    doc = dataset[idx]
                    print(f"文档 {idx}: 长度={len(doc)}, 前5个tokens={doc[:5].tolist()}")
                except Exception as e:
                    print(f"文档 {idx}: 读取失败 - {e}")
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'dataset' in locals():
            dataset.close()


if __name__ == "__main__":
    main()