#!/usr/bin/env python3
"""
简化的 Megatron-LM 集成测试脚本（不依赖外部包）
"""

import os
import sys

def test_file_detection():
    """测试文件类型检测功能（不导入外部模块）"""
    print("=== 测试文件类型检测逻辑 ===")
    
    def detect_file_type_simple(filename):
        """简化的文件类型检测"""
        if filename.endswith('.parquet'):
            return 'parquet'
        elif os.path.exists(filename + '.idx') and os.path.exists(filename + '.bin'):
            return 'megatron'
        elif os.path.basename(filename).find('.') == -1:
            if os.path.exists(filename + '.idx') and os.path.exists(filename + '.bin'):
                return 'megatron'
        return 'unknown'
    
    # 测试不同的文件路径
    test_files = [
        "test.parquet",
        "output_file",  # Megatron 格式
        "unknown_file.txt"
    ]
    
    for file in test_files:
        file_type = detect_file_type_simple(file)
        print(f"文件: {file} -> 类型: {file_type}")


def test_directory_structure():
    """测试目录结构"""
    print("\n=== 测试目录结构 ===")
    
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 检查一些关键目录和文件
    key_paths = [
        './accessory',
        './internlm2-chat-126m',
        './CCI-DATA',
        './accessory/model/tokenizer.py'
    ]
    
    for path in key_paths:
        exists = os.path.exists(path)
        path_type = "目录" if os.path.isdir(path) else "文件"
        status = "存在" if exists else "不存在"
        print(f"  {path} ({path_type}): {status}")


def test_imports():
    """测试关键模块的导入"""
    print("\n=== 测试模块导入 ===")
    
    modules_to_test = [
        ('sys', 'sys'),
        ('os', 'os'),
        ('glob', 'glob'),
        ('pickle', 'pickle'),
        ('multiprocessing', 'multiprocessing'),
        ('functools', 'functools.partial'),
    ]
    
    for module_name, import_path in modules_to_test:
        try:
            if '.' in import_path:
                parts = import_path.split('.')
                module = __import__(parts[0])
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                module = __import__(module_name)
            print(f"  ✓ {import_path}: 可用")
        except ImportError as e:
            print(f"  ✗ {import_path}: 不可用 ({e})")


def test_megatron_availability():
    """测试 Megatron-LM 的可用性"""
    print("\n=== 测试 Megatron-LM 可用性 ===")
    
    try:
        from megatron.data import indexed_dataset
        print("  ✓ Megatron-LM indexed_dataset: 可用")
        return True
    except ImportError as e:
        print(f"  ✗ Megatron-LM: 不可用 ({e})")
        print("  提示: 需要安装 Megatron-LM")
        return False


def test_accessory_availability():
    """测试 accessory 模块的可用性"""
    print("\n=== 测试 Accessory 模块可用性 ===")
    
    try:
        sys.path.append(os.path.abspath('.'))
        from accessory.model.tokenizer import Tokenizer
        print("  ✓ Accessory Tokenizer: 可用")
        return True
    except ImportError as e:
        print(f"  ✗ Accessory Tokenizer: 不可用 ({e})")
        return False


def show_usage_example():
    """显示使用示例"""
    print("\n=== 使用示例 ===")
    
    example_code = '''
# 基本使用方法
from pack_tokens_enhanced import scan_for_datasets, process_with_progress
from accessory.model.tokenizer import Tokenizer

# 配置参数
max_len = 1024
tokenizer = Tokenizer('./internlm2-chat-126m/tokenizer.model')

# 扫描数据目录
data_dirs = ['CCI-DATA']  # 可以包含 parquet 和 megatron 文件
files = scan_for_datasets(data_dirs)

# 处理文件
save_dir = "CCI-DATA/packed_tokens"
process_with_progress(files, save_dir, tokenizer, num_workers=24)
'''
    
    print(example_code)


if __name__ == "__main__":
    print("开始简化测试...\n")
    
    # 运行基础测试
    test_file_detection()
    test_directory_structure()
    test_imports()
    
    # 测试关键组件
    megatron_available = test_megatron_availability()
    accessory_available = test_accessory_availability()
    
    # 显示使用指南
    show_usage_example()
    
    print("\n=== 测试总结 ===")
    if megatron_available and accessory_available:
        print("✓ 所有核心组件都可用，可以正常使用集成功能")
    else:
        print("⚠ 部分组件不可用，需要安装相应依赖:")
        if not megatron_available:
            print("  - 安装 Megatron-LM: pip install megatron-lm")
        if not accessory_available:
            print("  - 检查 accessory 模块路径")
    
    print("\n使用 pack_tokens_enhanced.py 来处理你的数据！")