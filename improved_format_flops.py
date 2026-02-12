#!/usr/bin/env python3
"""
改进的 FLOPs 格式化函数
解决溢出问题并添加 Petaflops-days 支持
"""

import math
from typing import Union

def format_flops(flops: Union[float, int], include_petaflops_days: bool = False, 
                 training_time_seconds: float = None) -> str:
    """
    格式化 FLOPs 显示，支持更大的数值范围并避免溢出
    
    Args:
        flops: FLOPs 数值
        include_petaflops_days: 是否包含 Petaflops-days 格式
        training_time_seconds: 训练时间（秒），用于计算 Petaflops-days
    
    Returns:
        格式化后的字符串
    """
    
    # 处理特殊情况
    if flops < 0:
        return "OVERFLOW (negative value detected)"
    
    if flops == 0:
        return "0 FLOPS"
    
    if math.isinf(flops):
        return "INFINITY FLOPS"
    
    if math.isnan(flops):
        return "NaN FLOPS"
    
    # 使用科学计数法处理超大数值
    if flops >= 1e21:  # 超过 1000 PFLOPS 时使用科学计数法
        exponent = int(math.log10(flops))
        mantissa = flops / (10 ** exponent)
        base_format = f"{mantissa:.2f}e{exponent} FLOPS"
    elif flops >= 1e18:  # Exaflops (EFLOPs)
        base_format = f"{flops/1e18:.2f} EFLOPS"
    elif flops >= 1e15:  # Petaflops (PFLOPs)  
        base_format = f"{flops/1e15:.2f} PFLOPS"
    elif flops >= 1e12:  # Teraflops (TFLOPs)
        base_format = f"{flops/1e12:.2f} TFLOPS"
    elif flops >= 1e9:   # Gigaflops (GFLOPs)
        base_format = f"{flops/1e9:.2f} GFLOPS"
    elif flops >= 1e6:   # Megaflops (MFLOPs)
        base_format = f"{flops/1e6:.2f} MFLOPS"
    elif flops >= 1e3:   # Kiloflops (KFLOPs)
        base_format = f"{flops/1e3:.2f} KFLOPS"
    else:
        base_format = f"{flops:.2f} FLOPS"
    
    # 如果需要 Petaflops-days 格式
    if include_petaflops_days and training_time_seconds is not None:
        petaflops_days = calculate_petaflops_days(flops, training_time_seconds)
        return f"{base_format} ({petaflops_days})"
    
    return base_format

def calculate_petaflops_days(total_flops: float, time_seconds: float) -> str:
    """
    计算 Petaflops-days
    
    Args:
        total_flops: 总 FLOPs
        time_seconds: 时间（秒）
    
    Returns:
        Petaflops-days 格式的字符串
    """
    if time_seconds <= 0:
        return "0 Petaflops-days"
    
    # 1 Petaflops-day = 1e15 FLOPS/s * 86400 seconds = 8.64e19 FLOPS
    petaflops_days = total_flops / (1e15 * 86400)
    
    if petaflops_days >= 1000:
        return f"{petaflops_days:.2f} Petaflops-days"
    elif petaflops_days >= 1:
        return f"{petaflops_days:.3f} Petaflops-days"
    elif petaflops_days >= 0.001:
        return f"{petaflops_days*1000:.2f} Teraflops-days"
    else:
        return f"{petaflops_days*1000000:.2f} Gigaflops-days"

def format_flops_safe(flops: Union[float, int]) -> str:
    """
    安全的 FLOPs 格式化函数，专门处理溢出情况
    """
    try:
        # 转换为字符串检查是否为负数（溢出标志）
        flops_str = str(flops)
        if flops_str.startswith('-') and abs(flops) > 1e15:
            return "OVERFLOW (value too large)"
        
        return format_flops(flops)
    
    except (ValueError, OverflowError):
        return "OVERFLOW (calculation error)"

# 使用示例和测试
if __name__ == "__main__":
    print("=== FLOPs 格式化测试 ===")
    
    # 正常值测试
    test_values = [
        1000,
        1e6,
        1e9, 
        1e12,
        1e15,
        1e18,
        1e21,
        6377463.26e12,  # 您日志中的值
        -5691793225967927296,  # 溢出的负值
    ]
    
    for val in test_values:
        print(f"FLOPs: {val:e} -> {format_flops_safe(val)}")
    
    print("\n=== Petaflops-days 测试 ===")
    
    # Petaflops-days 测试
    total_flops = 6377463.26e12  # 6.38 PFLOPS
    training_time = 3600  # 1小时
    
    result = format_flops(total_flops, include_petaflops_days=True, 
                         training_time_seconds=training_time)
    print(f"带 Petaflops-days: {result}")
    
    # 不同训练时间的测试
    times = [3600, 86400, 86400*7, 86400*30]  # 1小时, 1天, 1周, 1月
    time_names = ["1小时", "1天", "1周", "1月"]
    
    for time_sec, name in zip(times, time_names):
        pfd = calculate_petaflops_days(total_flops, time_sec)
        print(f"{name}训练: {pfd}")