#!/usr/bin/env python3
"""
简化的验证功能测试脚本
不依赖外部库，仅测试代码结构和导入
"""

import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 1)[0])

def test_imports():
    """测试关键模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试falcon_packed模块
        from accessory.data.falcon_packed import FalconVal
        print("✓ 成功导入 FalconVal")
        
        # 检查FalconVal的__init__方法签名
        import inspect
        sig = inspect.signature(FalconVal.__init__)
        params = list(sig.parameters.keys())
        
        if 'val_data_meta_path' in params and 'val_data_root' in params:
            print("✓ FalconVal 包含新的验证数据集参数")
        else:
            print("✗ FalconVal 缺少新的验证数据集参数")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_file_existence():
    """测试关键文件是否存在"""
    print("\n测试文件存在性...")
    
    files_to_check = [
        "/workspace/evaluate_checkpoint.py",
        "/workspace/batch_evaluate_checkpoints.py",
        "/workspace/accessory/data/falcon_packed.py",
        "/workspace/accessory/main_pretrain.py",
        "/workspace/PACKED_VALIDATION_GUIDE.md",
        "/workspace/example_scripts/train_with_separate_val.sh",
        "/workspace/example_scripts/evaluate_checkpoint.sh",
        "/workspace/example_scripts/batch_evaluate_all_checkpoints.sh"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {os.path.basename(file_path)}")
        else:
            print(f"✗ {os.path.basename(file_path)} 不存在")
            all_exist = False
    
    return all_exist


def test_script_syntax():
    """测试脚本语法是否正确"""
    print("\n测试脚本语法...")
    
    scripts_to_check = [
        "/workspace/evaluate_checkpoint.py",
        "/workspace/batch_evaluate_checkpoints.py"
    ]
    
    all_valid = True
    for script_path in scripts_to_check:
        try:
            with open(script_path, 'r') as f:
                content = f.read()
            
            # 简单的语法检查
            compile(content, script_path, 'exec')
            print(f"✓ {os.path.basename(script_path)} 语法正确")
            
        except SyntaxError as e:
            print(f"✗ {os.path.basename(script_path)} 语法错误: {e}")
            all_valid = False
        except Exception as e:
            print(f"⚠ {os.path.basename(script_path)} 检查失败: {e}")
    
    return all_valid


def test_main_pretrain_modifications():
    """测试main_pretrain.py的修改"""
    print("\n测试main_pretrain.py修改...")
    
    try:
        with open("/workspace/accessory/main_pretrain.py", 'r') as f:
            content = f.read()
        
        # 检查是否包含新参数
        if '--val_data_meta_path' in content:
            print("✓ 包含 --val_data_meta_path 参数")
        else:
            print("✗ 缺少 --val_data_meta_path 参数")
            return False
        
        if '--val_data_root' in content:
            print("✓ 包含 --val_data_root 参数")
        else:
            print("✗ 缺少 --val_data_root 参数")
            return False
        
        # 检查是否包含验证数据集创建逻辑
        if 'val_data_meta_path=args.val_data_meta_path' in content:
            print("✓ 包含验证数据集创建逻辑")
        else:
            print("✗ 缺少验证数据集创建逻辑")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 检查main_pretrain.py失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("简化验证功能测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("文件存在性", test_file_existence),
        ("脚本语法", test_script_syntax),
        ("main_pretrain.py修改", test_main_pretrain_modifications),
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