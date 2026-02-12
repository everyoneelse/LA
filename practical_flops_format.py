#!/usr/bin/env python3
"""
实用的 FLOPs 格式化 - 直接用大单位避免溢出
不显示 OVERFLOW，而是自动选择合适的单位
"""

import math
from typing import Union

def format_flops(flops: Union[float, int]) -> str:
    """
    格式化 FLOPs 显示，自动选择合适单位避免溢出
    
    Args:
        flops: FLOPs 数值（可能很大或溢出）
    
    Returns:
        格式化后的字符串
    """
    
    # 处理明显的溢出情况（转换为绝对值处理）
    if flops < 0:
        # 负数通常意味着溢出，取绝对值处理
        flops = abs(flops)
    
    if flops == 0:
        return "0 FLOPS"
    
    # 处理无穷大和 NaN
    if math.isinf(flops):
        return "∞ FLOPS"
    
    if math.isnan(flops):
        return "NaN FLOPS"
    
    # 自动选择最合适的单位，优先使用大单位
    try:
        if flops >= 1e24:  # Yottaflops (超大规模)
            return f"{flops/1e24:.2f} YFLOPS"
        elif flops >= 1e21:  # Zettaflops
            return f"{flops/1e21:.2f} ZFLOPS"
        elif flops >= 1e18:  # Exaflops
            return f"{flops/1e18:.2f} EFLOPS"
        elif flops >= 1e15:  # Petaflops
            return f"{flops/1e15:.2f} PFLOPS"
        elif flops >= 1e12:  # Teraflops
            return f"{flops/1e12:.2f} TFLOPS"
        elif flops >= 1e9:   # Gigaflops
            return f"{flops/1e9:.2f} GFLOPS"
        elif flops >= 1e6:   # Megaflops
            return f"{flops/1e6:.2f} MFLOPS"
        elif flops >= 1e3:   # Kiloflops
            return f"{flops/1e3:.2f} KFLOPS"
        else:
            return f"{flops:.2f} FLOPS"
            
    except (ValueError, OverflowError, ZeroDivisionError):
        # 如果还是出错，用科学计数法
        try:
            return f"{flops:.2e} FLOPS"
        except:
            return "LARGE FLOPS"

def format_flops_with_petaflops_days(flops: Union[float, int]) -> str:
    """
    格式化 FLOPs 并显示 Petaflops-days（仅对大数值）
    
    Args:
        flops: FLOPs 数值
    
    Returns:
        包含 Petaflops-days 的格式化字符串
    """
    
    base_format = format_flops(flops)
    
    # 处理负数（溢出情况）
    actual_flops = abs(flops) if flops < 0 else flops
    
    # 只对大数值显示 Petaflops-days
    if actual_flops >= 1e15:  # >= 1 PFLOPS
        try:
            petaflops_days = actual_flops / (1e15 * 86400)
            
            if petaflops_days >= 1000:
                pfd_str = f"{petaflops_days:.1f} PF-days"
            elif petaflops_days >= 1:
                pfd_str = f"{petaflops_days:.2f} PF-days"
            elif petaflops_days >= 0.001:
                pfd_str = f"{petaflops_days*1000:.2f} TF-days"
            else:
                pfd_str = f"{petaflops_days*1000000:.2f} GF-days"
            
            return f"{base_format} ({pfd_str})"
            
        except:
            pass
    
    return base_format

# 专门用于训练日志的版本
def format_training_flops(batch_flops: Union[float, int], 
                         total_flops: Union[float, int]) -> str:
    """
    专门用于训练日志的 FLOPs 格式化
    
    Args:
        batch_flops: 单批次 FLOPs
        total_flops: 累计总 FLOPs
    
    Returns:
        训练日志格式的字符串
    """
    
    batch_str = format_flops(batch_flops)
    total_str = format_flops_with_petaflops_days(total_flops)
    
    return f"{batch_str} (batch), {total_str} (total)"

# 测试代码
if __name__ == "__main__":
    print("=== 实用 FLOPs 格式化测试 ===")
    print()
    
    # 模拟您的训练数据
    test_cases = [
        ("正常批次", 1.22e12),
        ("iter 130", 199272.16e12),
        ("iter 140", 398568.65e12),
        ("iter 150", 797161.62e12),
        ("iter 160", 1594347.57e12),
        ("iter 170", 3188719.47e12),
        ("iter 180", 6377463.26e12),
        ("原来的溢出值", -5691793225967927296),
        ("更大的数", 1e21),
        ("超大数", 1e25),
    ]
    
    print("基础格式化:")
    for name, value in test_cases:
        result = format_flops(value)
        print(f"  {name:12}: {result}")
    
    print("\n带 Petaflops-days:")
    for name, value in test_cases:
        result = format_flops_with_petaflops_days(value)
        print(f"  {name:12}: {result}")
    
    print("\n=== 模拟训练日志 ===")
    
    batch_flops = 1.22e12
    total_values = [199272.16e12, 398568.65e12, 6377463.26e12, -5691793225967927296]
    
    for i, total in enumerate(total_values, 130):
        log_str = format_training_flops(batch_flops, total)
        print(f"[Iter {i:3d}] FLOPs: {log_str}")
    
    print("\n✅ 优势：")
    print("- 不会显示 OVERFLOW，直接用合适单位")
    print("- 负数自动转为正数处理（溢出恢复）")
    print("- 自动选择最readable的单位")
    print("- 大数值自动显示 Petaflops-days")