#!/usr/bin/env python3
"""
OpenSeek 数据格式分析工具

用于分析 .bin 和 .idx 文件的实际格式结构
"""

import os
import struct
import sys
from pathlib import Path


def analyze_file_header(filepath, max_bytes=256):
    """
    分析文件头部的字节结构
    
    Args:
        filepath: 文件路径
        max_bytes: 要分析的最大字节数
    """
    print(f"\n=== 分析文件: {filepath} ===")
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    file_size = os.path.getsize(filepath)
    print(f"文件大小: {file_size:,} 字节 ({file_size / (1024**3):.3f} GB)")
    
    with open(filepath, 'rb') as f:
        # 读取文件头部
        header_bytes = f.read(min(max_bytes, file_size))
        
        print(f"\n前 {len(header_bytes)} 字节的十六进制表示:")
        for i in range(0, len(header_bytes), 16):
            chunk = header_bytes[i:i+16]
            hex_str = ' '.join(f'{b:02x}' for b in chunk)
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            print(f"{i:04x}: {hex_str:<48} |{ascii_str}|")
        
        # 尝试不同的数据类型解释
        print(f"\n尝试解释前几个字节:")
        
        # 8字节整数 (little endian)
        if len(header_bytes) >= 8:
            try:
                value = struct.unpack('<Q', header_bytes[:8])[0]
                print(f"前8字节作为uint64 (little): {value}")
            except:
                pass
        
        # 8字节整数 (big endian)
        if len(header_bytes) >= 8:
            try:
                value = struct.unpack('>Q', header_bytes[:8])[0]
                print(f"前8字节作为uint64 (big): {value}")
            except:
                pass
        
        # 4字节整数 (little endian)
        if len(header_bytes) >= 4:
            try:
                value = struct.unpack('<I', header_bytes[:4])[0]
                print(f"前4字节作为uint32 (little): {value}")
            except:
                pass
        
        # 4字节整数 (big endian)
        if len(header_bytes) >= 4:
            try:
                value = struct.unpack('>I', header_bytes[:4])[0]
                print(f"前4字节作为uint32 (big): {value}")
            except:
                pass
        
        # 检查是否包含文本
        try:
            text = header_bytes.decode('utf-8', errors='ignore')
            if text.strip():
                print(f"可能的文本内容: {repr(text[:100])}")
        except:
            pass


def detect_idx_format(idx_filepath):
    """
    检测 .idx 文件的格式
    
    Args:
        idx_filepath: .idx 文件路径
    """
    print(f"\n=== 检测 .idx 文件格式: {idx_filepath} ===")
    
    if not os.path.exists(idx_filepath):
        print(f"文件不存在: {idx_filepath}")
        return None
    
    file_size = os.path.getsize(idx_filepath)
    print(f"索引文件大小: {file_size:,} 字节")
    
    with open(idx_filepath, 'rb') as f:
        # 尝试不同的格式解析
        formats_to_try = [
            # 格式名称, 头部结构, 解析函数
            ("Megatron-LM 标准格式", parse_megatron_format),
            ("简化索引格式", parse_simple_format),
            ("自定义格式1", parse_custom_format1),
            ("自定义格式2", parse_custom_format2),
        ]
        
        for format_name, parse_func in formats_to_try:
            f.seek(0)  # 重置文件指针
            try:
                result = parse_func(f)
                if result:
                    print(f"✓ 可能的格式: {format_name}")
                    print(f"  解析结果: {result}")
                    return format_name, result
            except Exception as e:
                print(f"✗ {format_name} 格式解析失败: {e}")
        
        print("未能识别文件格式")
        return None


def parse_megatron_format(f):
    """解析 Megatron-LM 标准格式"""
    # Magic number (8 bytes)
    magic = struct.unpack('<Q', f.read(8))[0]
    
    # Version (8 bytes)
    version = struct.unpack('<Q', f.read(8))[0]
    
    # Data type (1 byte)
    dtype_code = struct.unpack('<B', f.read(1))[0]
    
    # Document count (8 bytes)
    doc_count = struct.unpack('<Q', f.read(8))[0]
    
    return {
        'magic': magic,
        'version': version,
        'dtype_code': dtype_code,
        'doc_count': doc_count
    }


def parse_simple_format(f):
    """解析简化索引格式"""
    # 只有文档数量 (4 bytes)
    doc_count = struct.unpack('<I', f.read(4))[0]
    
    # 检查是否合理
    if doc_count > 0 and doc_count < 1000000:  # 合理的文档数量范围
        return {'doc_count': doc_count, 'format': 'simple'}
    else:
        raise ValueError(f"不合理的文档数量: {doc_count}")


def parse_custom_format1(f):
    """解析自定义格式1"""
    # 尝试 8字节文档数量
    doc_count = struct.unpack('<Q', f.read(8))[0]
    
    if doc_count > 0 and doc_count < 10000000:
        return {'doc_count': doc_count, 'format': 'custom1'}
    else:
        raise ValueError(f"不合理的文档数量: {doc_count}")


def parse_custom_format2(f):
    """解析自定义格式2 - big endian"""
    # 尝试大端序
    doc_count = struct.unpack('>I', f.read(4))[0]
    
    if doc_count > 0 and doc_count < 1000000:
        return {'doc_count': doc_count, 'format': 'custom2_big_endian'}
    else:
        raise ValueError(f"不合理的文档数量: {doc_count}")


def find_data_files(search_dir="."):
    """搜索数据文件"""
    search_path = Path(search_dir)
    bin_files = list(search_path.rglob("*text_document.bin"))
    idx_files = list(search_path.rglob("*text_document.idx"))
    
    # 也搜索其他可能的模式
    if not bin_files:
        bin_files = list(search_path.rglob("*.bin"))
    if not idx_files:
        idx_files = list(search_path.rglob("*.idx"))
    
    return bin_files, idx_files


def main():
    """主函数"""
    print("=== OpenSeek 数据格式分析工具 ===\n")
    
    # 搜索数据文件
    if len(sys.argv) > 1:
        search_dir = sys.argv[1]
    else:
        search_dir = "."
    
    print(f"在目录 '{search_dir}' 中搜索数据文件...")
    bin_files, idx_files = find_data_files(search_dir)
    
    print(f"找到 {len(bin_files)} 个 .bin 文件")
    print(f"找到 {len(idx_files)} 个 .idx 文件")
    
    if not bin_files and not idx_files:
        print("未找到任何数据文件")
        return
    
    # 分析找到的文件
    for bin_file in bin_files[:3]:  # 只分析前3个
        analyze_file_header(str(bin_file))
    
    for idx_file in idx_files[:3]:  # 只分析前3个
        analyze_file_header(str(idx_file))
        detect_idx_format(str(idx_file))
    
    # 提供修复建议
    print(f"\n=== 修复建议 ===")
    print(f"1. 检查文件是否完整下载")
    print(f"2. 确认数据集的具体格式和版本")
    print(f"3. 尝试使用不同的解析方法")
    print(f"4. 联系数据提供方获取格式说明")


if __name__ == "__main__":
    main()