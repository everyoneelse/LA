#!/usr/bin/env python3
"""
修复溢出问题的 FLOPs 格式化函数
直接替换您原来的 format_flops 函数
"""

import math
from typing import Union

def format_flops(flops: Union[float, int]) -> str:
    """
    格式化 FLOPs 显示，修复溢出问题并添加 Petaflops-days 支持
    
    Args:
        flops: FLOPs 数值
    
    Returns:
        格式化后的字符串
    """
    
    # 处理特殊情况和溢出
    if _is_overflow_value(flops):
        return "OVERFLOW"
    
    if flops < 0:
        return "OVERFLOW (negative)"
    
    if flops == 0:
        return "0 FLOPS"
    
    if math.isinf(flops) or math.isnan(flops):
        return "INVALID FLOPS"
    
    # 正常格式化
    try:
        if flops >= 1e21:  # 超大数值使用科学计数法
            exponent = int(math.log10(flops))
            mantissa = flops / (10 ** exponent)
            return f"{mantissa:.2f}e{exponent} FLOPS"
        elif flops >= 1e18:  # Exaflops (EFLOPs)
            return f"{flops/1e18:.2f} EFLOPS"
        elif flops >= 1e15:  # Petaflops (PFLOPs)  
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
            return f"{flops:.2f} FLOPS"
            
    except (ValueError, OverflowError, ZeroDivisionError):
        return "OVERFLOW"

def _is_overflow_value(flops: Union[float, int]) -> bool:
    """检查是否为溢出值"""
    try:
        # 检查负数且绝对值很大（典型溢出标志）
        if flops < 0 and abs(flops) > 1e15:
            return True
        
        # 检查是否超出合理的计算范围
        if abs(flops) > 1e25:
            return True
            
        # 检查是否为典型的整数溢出值
        if isinstance(flops, int) and (flops < -2**63 or flops > 2**63):
            return True
            
        return False
        
    except (ValueError, TypeError):
        return True

def format_flops_with_petaflops_days(flops: Union[float, int], 
                                   training_time_seconds: float) -> str:
    """
    格式化 FLOPs 并包含 Petaflops-days 信息
    
    Args:
        flops: FLOPs 数值
        training_time_seconds: 训练时间（秒）
    
    Returns:
        包含 Petaflops-days 的格式化字符串
    """
    base_format = format_flops(flops)
    
    if "OVERFLOW" in base_format or "INVALID" in base_format:
        return base_format
    
    if training_time_seconds <= 0:
        return base_format
    
    try:
        # 计算 Petaflops-days
        # 1 Petaflops-day = 1e15 FLOPS/s * 86400 s = 8.64e19 FLOPS
        petaflops_days = flops / (1e15 * 86400)
        
        if petaflops_days >= 1000:
            pfd_str = f"{petaflops_days:.1f} PF-days"
        elif petaflops_days >= 1:
            pfd_str = f"{petaflops_days:.2f} PF-days"
        elif petaflops_days >= 0.001:
            pfd_str = f"{petaflops_days*1000:.2f} TF-days"
        elif petaflops_days >= 0.000001:
            pfd_str = f"{petaflops_days*1000000:.2f} GF-days"
        else:
            pfd_str = f"{petaflops_days*1000000000:.2f} MF-days"
        
        return f"{base_format} ({pfd_str})"
        
    except (ValueError, OverflowError, ZeroDivisionError):
        return base_format

# 测试代码
if __name__ == "__main__":
    print("=== 修复后的 FLOPs 格式化测试 ===")
    
    # 测试您日志中的具体数值
    test_values = [
        ("批次 FLOPs", 1.22e12),
        ("iter 130 总计", 199272.16e12),
        ("iter 140 总计", 398568.65e12),
        ("iter 150 总计", 797161.62e12),
        ("iter 160 总计", 1594347.57e12),
        ("iter 170 总计", 3188719.47e12),
        ("iter 180 总计", 6377463.26e12),
        ("溢出值", -5691793225967927296),
        ("另一个大数", 1e20),
        ("超大数", 1e25),
    ]
    
    print("基础格式化:")
    for name, value in test_values:
        result = format_flops(value)
        print(f"  {name:15}: {result}")
    
    print("\n带 Petaflops-days 格式化 (假设训练1小时):")
    training_time = 3600  # 1小时
    for name, value in test_values[:7]:  # 跳过溢出值
        result = format_flops_with_petaflops_days(value, training_time)
        print(f"  {name:15}: {result}")
    
    print(f"\n溢出值处理: {format_flops(-5691793225967927296)}")