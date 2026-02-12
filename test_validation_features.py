#!/usr/bin/env python3
"""
验证功能测试脚本
用于测试新增的验证数据集功能是否正常工作
"""

import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 1)[0])

import json
import tempfile
import pickle
import numpy as np
from pathlib import Path

# 测试导入
try:
    from accessory.data.falcon_packed import FalconVal
    from accessory.model.tokenizer import Tokenizer
    print("✓ 成功导入所需模块")
except ImportError as e:
    print(f"✗ 导入模块失败: {e}")
    sys.exit(1)


def create_test_data():
    """创建测试用的数据文件"""
    print("创建测试数据...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"测试数据目录: {temp_dir}")
    
    # 创建模拟的packed数据
    test_data = []
    for i in range(100):  # 100个样本
        # 创建随机token序列
        seq_len = np.random.randint(50, 200)
        tokens = np.random.randint(1, 1000, seq_len).tolist()
        test_data.append(tokens)
    
    # 保存为pkl文件
    test_file1 = os.path.join(temp_dir, "test_val_001.pkl")
    test_file2 = os.path.join(temp_dir, "test_val_002.pkl")
    
    with open(test_file1, 'wb') as f:
        pickle.dump(test_data[:50], f)
    
    with open(test_file2, 'wb') as f:
        pickle.dump(test_data[50:], f)
    
    # 创建meta文件
    meta_file = os.path.join(temp_dir, "test_val_meta.json")
    with open(meta_file, 'w') as f:
        json.dump(["test_val_001.pkl", "test_val_002.pkl"], f)
    
    print(f"✓ 创建了 {len(test_data)} 个测试样本")
    return temp_dir, meta_file


def test_falcon_val_original():
    """测试原有的FalconVal功能"""
    print("\n测试原有FalconVal功能...")
    
    try:
        # 创建测试数据
        temp_dir, _ = create_test_data()
        
        # 创建训练数据meta文件（模拟原有行为）
        train_meta_file = os.path.join(temp_dir, "train_meta.json")
        with open(train_meta_file, 'w') as f:
            json.dump(["test_val_001.pkl", "test_val_002.pkl"], f)
        
        # 测试原有行为（使用最后一个文件作为验证集）
        dataset = FalconVal(
            data_meta_path=train_meta_file,
            data_root=temp_dir,
            tokenizer_path=None,  # 测试时不需要真实tokenizer
            max_words=2048
        )
        
        print(f"✓ 原有功能正常，数据集大小: {len(dataset)}")
        
        # 测试数据获取
        sample = dataset[0]
        print(f"✓ 成功获取样本，形状: {sample[0].shape}")
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ 原有功能测试失败: {e}")
        return False


def test_falcon_val_separate():
    """测试新的独立验证数据集功能"""
    print("\n测试独立验证数据集功能...")
    
    try:
        # 创建测试数据
        temp_dir, val_meta_file = create_test_data()
        
        # 测试新功能（使用独立的验证数据集）
        dataset = FalconVal(
            data_meta_path=None,  # 不使用训练数据
            data_root=None,
            tokenizer_path=None,  # 测试时不需要真实tokenizer
            max_words=2048,
            val_data_meta_path=val_meta_file,
            val_data_root=temp_dir
        )
        
        print(f"✓ 独立验证数据集功能正常，数据集大小: {len(dataset)}")
        
        # 测试数据获取
        sample = dataset[0]
        print(f"✓ 成功获取样本，形状: {sample[0].shape}")
        
        # 验证数据是否正确合并了多个文件
        if len(dataset) == 100:  # 应该是两个文件的总和
            print("✓ 多文件合并功能正常")
        else:
            print(f"⚠ 数据合并可能有问题，期望100个样本，实际{len(dataset)}个")
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ 独立验证数据集功能测试失败: {e}")
        return False


def test_argument_parsing():
    """测试参数解析功能"""
    print("\n测试参数解析功能...")
    
    try:
        # 测试evaluate_checkpoint.py的参数解析
        from evaluate_checkpoint import get_args_parser
        
        parser = get_args_parser()
        
        # 测试基本参数
        test_args = [
            "--checkpoint_path", "/test/path",
            "--val_data_meta_path", "/test/val_meta.json",
            "--tokenizer_path", "/test/tokenizer.model",
            "--packed_data",
            "--batch_size", "8",
            "--max_words", "1024"
        ]
        
        args = parser.parse_args(test_args)
        
        assert args.checkpoint_path == "/test/path"
        assert args.val_data_meta_path == "/test/val_meta.json"
        assert args.packed_data == True
        assert args.batch_size == 8
        assert args.max_words == 1024
        
        print("✓ evaluate_checkpoint.py 参数解析正常")
        
        # 测试batch_evaluate_checkpoints.py的参数解析
        from batch_evaluate_checkpoints import get_args_parser as get_batch_parser
        
        batch_parser = get_batch_parser()
        
        batch_test_args = [
            "--output_dir", "/test/output",
            "--val_data_meta_path", "/test/val_meta.json",
            "--tokenizer_path", "/test/tokenizer.model",
            "--packed_data",
            "--plot_results",
            "--min_iter", "1000",
            "--max_iter", "10000"
        ]
        
        batch_args = batch_parser.parse_args(batch_test_args)
        
        assert batch_args.output_dir == "/test/output"
        assert batch_args.plot_results == True
        assert batch_args.min_iter == 1000
        assert batch_args.max_iter == 10000
        
        print("✓ batch_evaluate_checkpoints.py 参数解析正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 参数解析测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("验证功能测试")
    print("=" * 60)
    
    tests = [
        ("原有FalconVal功能", test_falcon_val_original),
        ("独立验证数据集功能", test_falcon_val_separate),
        ("参数解析功能", test_argument_parsing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ 测试 '{test_name}' 出现异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！功能实现正常。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查相关功能。")
        return 1


if __name__ == '__main__':
    sys.exit(main())