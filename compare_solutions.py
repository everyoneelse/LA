#!/usr/bin/env python3
"""
对比您的代码和修复后的代码
展示关键区别
"""

import math

def your_original_format_flops(flops: float) -> str:
    """您原来的代码"""
    try:
        if flops >= 1e21:  # 超大数值使用科学计数法
            exponent = int(math.log10(flops))
            mantissa = flops / (10 ** exponent)
            return f"{mantissa:.2f}e{exponent} FLOPS"
        elif flops >= 1e18:  # Exaflops (EFLOPs) - 新增
            return f"{flops/1e18:.2f} EFLOPS"
        elif flops >= 1e15:  # Petaflops (PFLOPs) - 新增
            return f"{flops/1e15:.2f} PFLOPS"
        elif flops >= 1e12:  # Teraflops (TFLOPs)
            return f"{flops/1e12:.2f} TFLOPS"
        elif flops >= 1e9:   # Gigaflops (GFLOPs)
            return f"{flops/1e9:.2f} GFLOPS"
        elif flops >= 1e6:   # Megaflops (MFLOPs)
            return f"{flops/1e6:.2f} MFLOPS"
        elif flops >= 1e3:   # Kiloflops (KFLOPs)
            return f"{flops/1e3:.2f} KFLOPS"
        else:
            return f"{flops:.2f} FLOPS"  # ← 负数会走到这里！
    except (ValueError, OverflowError, ZeroDivisionError):
        return "OVERFLOW"

def fixed_format_flops(flops: float) -> str:
    """修复后的代码"""
    
    # 🔑 关键区别：处理负数！
    if flops < 0:
        flops = abs(flops)  # ← 这是关键！
    
    if flops == 0:
        return "0 FLOPS"
    
    try:
        if flops >= 1e21:
            exponent = int(math.log10(flops))
            mantissa = flops / (10 ** exponent)
            return f"{mantissa:.2f}e{exponent} FLOPS"
        elif flops >= 1e18:
            return f"{flops/1e18:.2f} EFLOPS"
        elif flops >= 1e15:
            return f"{flops/1e15:.2f} PFLOPS"
        elif flops >= 1e12:
            return f"{flops/1e12:.2f} TFLOPS"
        elif flops >= 1e9:
            return f"{flops/1e9:.2f} GFLOPS"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f} MFLOPS"
        elif flops >= 1e3:
            return f"{flops/1e3:.2f} KFLOPS"
        else:
            return f"{flops:.2f} FLOPS"
    except (ValueError, OverflowError, ZeroDivisionError):
        return "OVERFLOW"

def test_both_versions():
    """测试两个版本的差异"""
    
    print("=== 代码对比测试 ===")
    print()
    
    test_values = [
        ("正常值", 1.22e12),
        ("大数值", 6.38e18), 
        ("溢出负数", -5691793225967927296),
        ("您日志中的负数", -100007955.59e6),
        ("另一个负数", -105595189.47e6),
    ]
    
    print(f"{'测试值':<15} | {'您的原代码':<25} | {'修复后代码':<25}")
    print("-" * 70)
    
    for name, value in test_values:
        original = your_original_format_flops(value)
        fixed = fixed_format_flops(value)
        print(f"{name:<15} | {original:<25} | {fixed:<25}")
    
    print("\n=== 关键区别分析 ===")
    
    negative_value = -5691793225967927296
    print(f"测试负数: {negative_value}")
    print()
    
    print("您的代码执行路径:")
    print(f"1. flops = {negative_value}")
    print(f"2. flops >= 1e18? {negative_value >= 1e18} (False)")
    print(f"3. flops >= 1e15? {negative_value >= 1e15} (False)")
    print(f"4. ... 所有条件都是 False")
    print(f"5. 走到 else: return f'{negative_value:.2f} FLOPS'")
    print(f"6. 结果: {your_original_format_flops(negative_value)}")
    
    print("\n修复后代码执行路径:")
    abs_value = abs(negative_value)
    print(f"1. flops = {negative_value}")
    print(f"2. flops < 0? True, 转换为 flops = {abs_value}")
    print(f"3. flops >= 1e18? {abs_value >= 1e18} (True)")
    print(f"4. 返回: {abs_value/1e18:.2f} EFLOPS")
    print(f"5. 结果: {fixed_format_flops(negative_value)}")

if __name__ == "__main__":
    test_both_versions()
    
    print("\n" + "="*60)
    print("🔑 核心问题：负数处理")
    print("   您的代码：负数 → 不满足条件 → else 分支 → 显示负数")
    print("   修复代码：负数 → abs() → 满足条件 → 正常格式化")
    print()
    print("🚀 解决方案：在代码开头添加一行")
    print("   if flops < 0:")
    print("       flops = abs(flops)")
    print("="*60)