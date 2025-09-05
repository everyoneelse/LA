#!/usr/bin/env python3
"""
OpenSeek 自定义格式解析器

专门处理实际的 OpenSeek 索引文件格式
基于观察：734KB 索引文件 vs 940万文档，说明使用了特殊的压缩或分块格式
"""

import os
import struct
import numpy as np
from typing import List, Optional, Union, Dict, Any


class OpenSeekCustomDataset:
    """
    自定义 OpenSeek 数据集解析器
    
    基于实际文件大小分析，推测可能的格式：
    1. 分块索引格式（每个块包含多个文档）
    2. 压缩索引格式
    3. 稀疏索引格式（只存储关键点）
    """
    
    def __init__(self, path_prefix: str):
        """
        初始化数据集
        
        Args:
            path_prefix: 数据文件路径前缀
        """
        self.path_prefix = path_prefix
        self.bin_path = path_prefix + '.bin'
        self.idx_path = path_prefix + '.idx'
        
        # 检查文件是否存在
        if not os.path.exists(self.bin_path):
            raise FileNotFoundError(f"Binary file not found: {self.bin_path}")
        if not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"Index file not found: {self.idx_path}")
        
        self.bin_size = os.path.getsize(self.bin_path)
        self.idx_size = os.path.getsize(self.idx_path)
        
        print(f"正在分析数据集: {path_prefix}")
        print(f"二进制文件大小: {self.bin_size / (1024**3):.2f} GB")
        print(f"索引文件大小: {self.idx_size / (1024**2):.2f} MB")
        
        # 分析和解析索引文件
        self._analyze_and_parse_index()
        
        # 打开二进制文件
        self.bin_file = open(self.bin_path, 'rb')
        
        print(f"✓ 数据集解析成功")
        print(f"  实际文档数量: {len(self.doc_info)}")
        print(f"  索引格式: {self.index_format}")
    
    def _analyze_and_parse_index(self):
        """分析并解析索引文件"""
        with open(self.idx_path, 'rb') as f:
            # 读取头部
            magic = struct.unpack('<Q', f.read(8))[0]
            version = struct.unpack('<Q', f.read(8))[0]
            dtype_code = struct.unpack('<B', f.read(1))[0]
            reported_doc_count = struct.unpack('<Q', f.read(8))[0]
            
            print(f"头部信息:")
            print(f"  Magic: {magic}")
            print(f"  Version: {version}")
            print(f"  Data type code: {dtype_code}")
            print(f"  报告的文档数量: {reported_doc_count:,}")
            
            # 计算剩余数据
            header_size = 25
            remaining_bytes = self.idx_size - header_size
            print(f"  剩余数据: {remaining_bytes:,} 字节")
            
            # 分析可能的格式
            bytes_per_doc = remaining_bytes / reported_doc_count if reported_doc_count > 0 else 0
            print(f"  平均每文档字节数: {bytes_per_doc:.3f}")
            
            if bytes_per_doc < 1:
                print("  -> 可能是分块/压缩索引格式")
                self._parse_compressed_format(f, reported_doc_count)
            elif 1 <= bytes_per_doc < 4:
                print("  -> 可能是稀疏索引格式")
                self._parse_sparse_format(f, reported_doc_count)
            else:
                print("  -> 尝试标准格式的变体")
                self._parse_standard_variant(f, reported_doc_count)
    
    def _parse_compressed_format(self, f, reported_doc_count):
        """解析压缩/分块格式"""
        print("尝试解析压缩/分块格式...")
        
        # 读取所有剩余数据
        remaining_data = f.read()
        print(f"剩余数据长度: {len(remaining_data)} 字节")
        
        # 假设这是一个分块索引，每个块代表一个数据段
        # 尝试解析为连续的偏移量数组
        
        # 尝试4字节偏移量
        if len(remaining_data) % 4 == 0:
            num_offsets = len(remaining_data) // 4
            print(f"尝试解析为 {num_offsets} 个4字节偏移量...")
            
            offsets = []
            for i in range(num_offsets):
                offset = struct.unpack('<I', remaining_data[i*4:(i+1)*4])[0]
                offsets.append(offset)
            
            # 验证偏移量是否合理
            valid_offsets = [off for off in offsets if 0 <= off <= self.bin_size]
            print(f"有效偏移量: {len(valid_offsets)} / {num_offsets}")
            
            if len(valid_offsets) > num_offsets * 0.8:  # 80%以上有效
                self._build_doc_info_from_offsets(valid_offsets, reported_doc_count)
                self.index_format = "4-byte offsets array"
                return
        
        # 尝试8字节偏移量
        if len(remaining_data) % 8 == 0:
            num_offsets = len(remaining_data) // 8
            print(f"尝试解析为 {num_offsets} 个8字节偏移量...")
            
            offsets = []
            for i in range(num_offsets):
                offset = struct.unpack('<Q', remaining_data[i*8:(i+1)*8])[0]
                offsets.append(offset)
            
            valid_offsets = [off for off in offsets if 0 <= off <= self.bin_size]
            print(f"有效偏移量: {len(valid_offsets)} / {num_offsets}")
            
            if len(valid_offsets) > num_offsets * 0.8:
                self._build_doc_info_from_offsets(valid_offsets, reported_doc_count)
                self.index_format = "8-byte offsets array"
                return
        
        # 尝试混合格式：偏移量 + 长度信息
        self._try_mixed_format(remaining_data, reported_doc_count)
    
    def _build_doc_info_from_offsets(self, offsets, reported_doc_count):
        """从偏移量数组构建文档信息"""
        # 排序偏移量
        offsets = sorted(set(offsets))  # 去重并排序
        
        print(f"构建文档信息，偏移量数: {len(offsets)}")
        
        self.doc_info = []
        
        # 如果偏移量数量接近报告的文档数量，每个偏移量对应一个文档
        if len(offsets) >= reported_doc_count * 0.8:
            for i, offset in enumerate(offsets):
                if i < len(offsets) - 1:
                    length = (offsets[i + 1] - offset) // 4  # 假设int32
                else:
                    length = (self.bin_size - offset) // 4
                
                if length > 0:
                    self.doc_info.append({
                        'offset': offset,
                        'length': length,
                        'doc_id': i
                    })
        else:
            # 偏移量数量远少于文档数量，可能是分块格式
            # 每个偏移量对应一个包含多个文档的块
            docs_per_chunk = reported_doc_count // len(offsets)
            print(f"推测每块包含约 {docs_per_chunk} 个文档")
            
            for i, offset in enumerate(offsets):
                if i < len(offsets) - 1:
                    chunk_size = offsets[i + 1] - offset
                else:
                    chunk_size = self.bin_size - offset
                
                # 假设块内文档长度相等
                doc_length = (chunk_size // 4) // docs_per_chunk if docs_per_chunk > 0 else chunk_size // 4
                
                # 为这个块中的每个文档创建条目
                for j in range(docs_per_chunk):
                    doc_offset = offset + j * doc_length * 4
                    if doc_offset < self.bin_size:
                        self.doc_info.append({
                            'offset': doc_offset,
                            'length': doc_length,
                            'doc_id': i * docs_per_chunk + j,
                            'chunk_id': i
                        })
        
        print(f"生成了 {len(self.doc_info)} 个文档条目")
    
    def _try_mixed_format(self, data, reported_doc_count):
        """尝试混合格式"""
        print("尝试混合格式解析...")
        
        # 尝试解析为块描述符数组
        # 每个描述符可能包含：起始偏移量 + 块大小 + 文档数量
        
        if len(data) % 12 == 0:  # 4+4+4 字节
            num_blocks = len(data) // 12
            print(f"尝试解析为 {num_blocks} 个块描述符（12字节每个）...")
            
            self.doc_info = []
            total_docs = 0
            
            for i in range(num_blocks):
                offset = i * 12
                block_offset = struct.unpack('<I', data[offset:offset+4])[0]
                block_size = struct.unpack('<I', data[offset+4:offset+8])[0] 
                block_doc_count = struct.unpack('<I', data[offset+8:offset+12])[0]
                
                if (0 <= block_offset <= self.bin_size and 
                    block_size > 0 and 
                    block_doc_count > 0 and
                    block_doc_count < 1000000):  # 合理的文档数量
                    
                    # 计算块内每个文档的平均长度
                    avg_doc_length = (block_size // 4) // block_doc_count
                    
                    # 为块内每个文档创建条目
                    for j in range(block_doc_count):
                        doc_offset = block_offset + j * avg_doc_length * 4
                        if doc_offset < self.bin_size:
                            self.doc_info.append({
                                'offset': doc_offset,
                                'length': avg_doc_length,
                                'doc_id': total_docs + j,
                                'block_id': i
                            })
                    
                    total_docs += block_doc_count
            
            if len(self.doc_info) > 0:
                print(f"混合格式解析成功，生成 {len(self.doc_info)} 个文档条目")
                self.index_format = "block descriptors (12-byte)"
                return
        
        # 如果所有方法都失败，创建一个简单的均匀分布
        print("使用均匀分布作为后备方案...")
        self._create_uniform_distribution(reported_doc_count)
    
    def _create_uniform_distribution(self, doc_count):
        """创建均匀分布的文档索引（作为后备方案）"""
        print(f"创建 {doc_count} 个文档的均匀分布索引...")
        
        # 假设文档均匀分布在二进制文件中
        avg_doc_size_bytes = self.bin_size // doc_count
        avg_doc_length = avg_doc_size_bytes // 4  # 假设int32
        
        print(f"平均文档大小: {avg_doc_size_bytes} 字节 ({avg_doc_length} tokens)")
        
        self.doc_info = []
        for i in range(doc_count):
            offset = i * avg_doc_size_bytes
            self.doc_info.append({
                'offset': offset,
                'length': avg_doc_length,
                'doc_id': i
            })
        
        self.index_format = "uniform distribution (fallback)"
    
    def _parse_sparse_format(self, f, reported_doc_count):
        """解析稀疏格式"""
        print("尝试解析稀疏格式...")
        # 实现稀疏格式解析逻辑
        remaining_data = f.read()
        self._create_uniform_distribution(reported_doc_count)
    
    def _parse_standard_variant(self, f, reported_doc_count):
        """解析标准格式的变体"""
        print("尝试解析标准格式变体...")
        remaining_data = f.read()
        self._create_uniform_distribution(reported_doc_count)
    
    def __len__(self):
        """返回文档数量"""
        return len(self.doc_info)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """获取指定索引的文档数据"""
        if idx >= len(self.doc_info):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.doc_info)} documents")
        
        doc = self.doc_info[idx]
        offset = doc['offset']
        length = doc['length']
        
        # 读取数据
        self.bin_file.seek(offset)
        bytes_to_read = length * 4  # 假设int32
        data_bytes = self.bin_file.read(bytes_to_read)
        
        if len(data_bytes) < bytes_to_read:
            # 调整长度以匹配实际读取的数据
            actual_length = len(data_bytes) // 4
            data_bytes = data_bytes[:actual_length * 4]
        
        # 转换为numpy数组
        data = np.frombuffer(data_bytes, dtype=np.int32)
        return data
    
    def get_batch(self, indices: List[int]) -> List[np.ndarray]:
        """批量获取文档"""
        return [self[idx] for idx in indices]
    
    def get_info(self):
        """获取数据集信息"""
        return {
            'total_documents': len(self.doc_info),
            'index_format': self.index_format,
            'bin_file_size': self.bin_size,
            'idx_file_size': self.idx_size,
            'avg_doc_length': np.mean([doc['length'] for doc in self.doc_info]) if self.doc_info else 0
        }
    
    def close(self):
        """关闭文件"""
        if hasattr(self, 'bin_file') and self.bin_file:
            self.bin_file.close()
    
    def __del__(self):
        """析构函数"""
        self.close()


def load_openseek_custom(path_prefix: str) -> OpenSeekCustomDataset:
    """加载 OpenSeek 数据集（自定义解析器）"""
    return OpenSeekCustomDataset(path_prefix)


def main():
    """示例用法"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 openseek_custom_parser.py <data_path_prefix>")
        return
    
    path_prefix = sys.argv[1]
    
    try:
        print("=== OpenSeek 自定义格式解析器 ===\n")
        
        dataset = load_openseek_custom(path_prefix)
        
        # 显示信息
        info = dataset.get_info()
        print(f"\n=== 数据集信息 ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # 测试读取
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
        print(f"解析失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'dataset' in locals():
            dataset.close()


if __name__ == "__main__":
    main()