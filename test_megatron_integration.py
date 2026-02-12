#!/usr/bin/env python3
"""
测试 Megatron-LM 数据集成功能的脚本
"""

import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])

from pack_tokens_enhanced import detect_file_type, read_megatron_dataset, scan_for_datasets
from accessory.model.tokenizer import Tokenizer

def test_file_detection():
    """测试文件类型检测功能"""
    print("=== 测试文件类型检测 ===")
    
    # 测试不同的文件路径
    test_files = [
        "test.parquet",
        "output_file",  # Megatron 格式（假设存在 output_file.idx 和 output_file.bin）
        "unknown_file.txt"
    ]
    
    for file in test_files:
        file_type = detect_file_type(file)
        print(f"文件: {file} -> 类型: {file_type}")


def test_megatron_reading():
    """测试 Megatron 数据集读取功能"""
    print("\n=== 测试 Megatron 数据集读取 ===")
    
    # 这里需要一个实际的 Megatron 数据集文件
    # 假设你有一个名为 "output_file" 的 Megatron 数据集
    dataset_prefix = "output_file"
    
    try:
        dataset = read_megatron_dataset(dataset_prefix)
        if dataset is not None:
            print(f"成功加载 Megatron 数据集: {dataset_prefix}")
            print(f"数据集大小: {len(dataset)}")
            
            # 读取前几个文档
            for i in range(min(3, len(dataset))):
                doc = dataset[i]
                print(f"文档 {i}: 类型={type(doc)}, 长度={len(doc) if hasattr(doc, '__len__') else 'N/A'}")
                if hasattr(doc, 'shape'):
                    print(f"  形状: {doc.shape}")
                # 显示前10个token
                if hasattr(doc, 'tolist'):
                    tokens = doc.tolist()[:10]
                elif hasattr(doc, '__iter__'):
                    tokens = list(doc)[:10]
                else:
                    tokens = str(doc)[:50]
                print(f"  前几个tokens: {tokens}")
        else:
            print(f"无法加载 Megatron 数据集: {dataset_prefix}")
            
    except Exception as e:
        print(f"测试 Megatron 读取时出错: {e}")


def test_tokenizer():
    """测试 tokenizer 是否正常工作"""
    print("\n=== 测试 Tokenizer ===")
    
    try:
        tokenizer = Tokenizer('./internlm2-chat-126m/tokenizer.model')
        
        # 测试编码
        test_text = "Hello, this is a test sentence."
        tokens = tokenizer.encode(test_text, bos=True, eos=True)
        print(f"原文: {test_text}")
        print(f"编码结果: {tokens}")
        print(f"Token 数量: {len(tokens)}")
        
        # 测试解码（如果支持）
        if hasattr(tokenizer, 'decode'):
            decoded = tokenizer.decode(tokens)
            print(f"解码结果: {decoded}")
            
    except Exception as e:
        print(f"测试 Tokenizer 时出错: {e}")


def test_directory_scanning():
    """测试目录扫描功能"""
    print("\n=== 测试目录扫描 ===")
    
    # 测试扫描当前目录
    test_dirs = ['.', 'CCI-DATA']  # 可以根据实际情况调整
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\n扫描目录: {test_dir}")
            files = scan_for_datasets([test_dir])
            print(f"找到 {len(files)} 个数据集文件")
            
            # 显示每种类型的文件数量
            parquet_count = sum(1 for f in files if detect_file_type(f) == 'parquet')
            megatron_count = sum(1 for f in files if detect_file_type(f) == 'megatron')
            unknown_count = len(files) - parquet_count - megatron_count
            
            print(f"  - Parquet 文件: {parquet_count}")
            print(f"  - Megatron 文件: {megatron_count}")
            print(f"  - 未知类型: {unknown_count}")
            
            # 显示前几个文件
            for i, file in enumerate(files[:5]):
                file_type = detect_file_type(file)
                print(f"  {i+1}. {file} ({file_type})")
        else:
            print(f"目录不存在: {test_dir}")


if __name__ == "__main__":
    print("开始测试 Megatron-LM 集成功能...\n")
    
    # 运行所有测试
    test_file_detection()
    test_tokenizer()
    test_directory_scanning()
    test_megatron_reading()
    
    print("\n测试完成！")
    print("\n使用说明:")
    print("1. 确保安装了 Megatron-LM: pip install megatron-lm")
    print("2. 将你的 Megatron 数据集文件（.idx 和 .bin）放在数据目录中")
    print("3. 运行 pack_tokens_enhanced.py 来处理数据")
    print("4. 脚本会自动检测文件类型并使用相应的处理方法")