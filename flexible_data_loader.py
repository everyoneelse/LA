#!/usr/bin/env python3
"""
灵活的 OpenSeek-Pretrain-100B 数据加载器

支持指定数据文件路径，并提供多种加载选项
"""

import os
import sys
import argparse
from pathlib import Path


def find_data_files(search_dir: str = "."):
    """
    在指定目录中搜索 .bin 和 .idx 文件
    
    Args:
        search_dir: 搜索目录
        
    Returns:
        找到的数据文件前缀列表
    """
    search_path = Path(search_dir)
    bin_files = list(search_path.rglob("*.bin"))
    idx_files = list(search_path.rglob("*.idx"))
    
    # 找到匹配的 .bin 和 .idx 文件对
    data_prefixes = []
    
    for bin_file in bin_files:
        # 检查是否有对应的 .idx 文件
        idx_file = bin_file.with_suffix('.idx')
        if idx_file in idx_files:
            # 获取文件前缀（去除扩展名）
            prefix = str(bin_file.with_suffix(''))
            data_prefixes.append(prefix)
    
    return data_prefixes


def load_openseek_data(data_prefix: str):
    """
    加载 OpenSeek 数据
    
    Args:
        data_prefix: 数据文件前缀路径
        
    Returns:
        数据集对象或 None
    """
    try:
        from load_openseek_data import IndexedDataset
        
        print(f"正在加载数据: {data_prefix}")
        dataset = IndexedDataset(data_prefix)
        
        print(f"✓ 数据加载成功！")
        print(f"  文档总数: {len(dataset)}")
        print(f"  数据类型: {dataset.dtype}")
        
        return dataset
        
    except FileNotFoundError as e:
        print(f"✗ 文件未找到: {e}")
        return None
    except Exception as e:
        print(f"✗ 加载数据时出错: {e}")
        return None


def interactive_file_selection(data_prefixes):
    """
    交互式文件选择
    
    Args:
        data_prefixes: 数据文件前缀列表
        
    Returns:
        选择的文件前缀
    """
    if not data_prefixes:
        print("未找到任何 .bin/.idx 文件对")
        return None
    
    print(f"\n找到 {len(data_prefixes)} 个数据文件:")
    for i, prefix in enumerate(data_prefixes):
        bin_size = os.path.getsize(prefix + '.bin') if os.path.exists(prefix + '.bin') else 0
        idx_size = os.path.getsize(prefix + '.idx') if os.path.exists(prefix + '.idx') else 0
        
        print(f"  {i+1}. {prefix}")
        print(f"     .bin 文件大小: {bin_size / (1024**3):.2f} GB")
        print(f"     .idx 文件大小: {idx_size / (1024**2):.2f} MB")
    
    while True:
        try:
            choice = input(f"\n请选择要加载的数据文件 (1-{len(data_prefixes)}): ")
            idx = int(choice) - 1
            
            if 0 <= idx < len(data_prefixes):
                return data_prefixes[idx]
            else:
                print(f"请输入 1 到 {len(data_prefixes)} 之间的数字")
                
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消操作")
            return None


def demonstrate_usage(dataset):
    """
    演示数据集的使用方法
    
    Args:
        dataset: 数据集对象
    """
    print(f"\n=== 数据集使用演示 ===")
    
    # 基本信息
    print(f"数据集大小: {len(dataset)}")
    
    if hasattr(dataset, 'doc_lengths'):
        lengths = dataset.doc_lengths
        print(f"文档长度统计:")
        print(f"  平均长度: {sum(lengths) / len(lengths):.2f}")
        print(f"  最小长度: {min(lengths)}")
        print(f"  最大长度: {max(lengths)}")
    
    # 获取样本数据
    print(f"\n获取前3个文档的数据:")
    for i in range(min(3, len(dataset))):
        try:
            doc_data = dataset[i]
            print(f"文档 {i}:")
            print(f"  长度: {len(doc_data)}")
            print(f"  数据类型: {doc_data.dtype}")
            print(f"  前10个token: {doc_data[:10].tolist()}")
            print(f"  值范围: {doc_data.min()} - {doc_data.max()}")
            print()
        except Exception as e:
            print(f"  读取文档 {i} 时出错: {e}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="OpenSeek-Pretrain-100B 数据加载器")
    parser.add_argument("--data-dir", type=str, default=".", 
                       help="数据文件搜索目录 (默认: 当前目录)")
    parser.add_argument("--data-prefix", type=str, 
                       help="直接指定数据文件前缀路径")
    parser.add_argument("--list-only", action="store_true", 
                       help="仅列出找到的数据文件，不加载")
    
    args = parser.parse_args()
    
    print("=== OpenSeek-Pretrain-100B 灵活数据加载器 ===\n")
    
    if args.data_prefix:
        # 直接使用指定的文件前缀
        print(f"使用指定的数据文件前缀: {args.data_prefix}")
        dataset = load_openseek_data(args.data_prefix)
        
        if dataset:
            demonstrate_usage(dataset)
            dataset.close()
    else:
        # 搜索数据文件
        print(f"在目录 '{args.data_dir}' 中搜索数据文件...")
        data_prefixes = find_data_files(args.data_dir)
        
        if args.list_only:
            if data_prefixes:
                print(f"找到 {len(data_prefixes)} 个数据文件:")
                for prefix in data_prefixes:
                    print(f"  {prefix}")
            else:
                print("未找到任何数据文件")
            return
        
        # 交互式选择和加载
        selected_prefix = interactive_file_selection(data_prefixes)
        
        if selected_prefix:
            dataset = load_openseek_data(selected_prefix)
            
            if dataset:
                demonstrate_usage(dataset)
                dataset.close()
        else:
            print("未选择任何数据文件")
    
    print(f"\n=== 使用说明 ===")
    print(f"1. 自动搜索并选择数据文件:")
    print(f"   python3 flexible_data_loader.py --data-dir /path/to/your/data")
    print(f"")
    print(f"2. 直接指定数据文件:")
    print(f"   python3 flexible_data_loader.py --data-prefix /path/to/your/data_prefix")
    print(f"")
    print(f"3. 仅列出找到的数据文件:")
    print(f"   python3 flexible_data_loader.py --list-only --data-dir /path/to/your/data")
    print(f"")
    print(f"注意: data_prefix 应该是不包含 .bin 或 .idx 扩展名的文件路径前缀")


if __name__ == "__main__":
    main()