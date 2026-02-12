#!/usr/bin/env python3
"""
简化的 Petaflops-days 计算，不需要时间信息
基于总 FLOPs 直接换算
"""

import math
from typing import Union

def format_flops_simple(flops: Union[float, int]) -> str:
    """
    格式化 FLOPs 显示，包含简化的 Petaflops-days
    不需要时间信息，直接基于总 FLOPs 换算
    """
    
    # 处理溢出和特殊情况
    if _is_overflow_value(flops):
        return "OVERFLOW"
    
    if flops < 0:
        return "OVERFLOW (negative)"
    
    if flops == 0:
        return "0 FLOPS"
    
    if math.isinf(flops) or math.isnan(flops):
        return "INVALID FLOPS"
    
    # 基础格式化
    base_format = _format_base_flops(flops)
    
    # 添加 Petaflops-days 等价值（基于标准换算）
    pfd_equivalent = _calculate_petaflops_days_equivalent(flops)
    
    return f"{base_format} (~{pfd_equivalent})"

def _format_base_flops(flops: float) -> str:
    """基础 FLOPs 格式化"""
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
    except:
        return "OVERFLOW"

def _calculate_petaflops_days_equivalent(total_flops: float) -> str:
    """
    计算 Petaflops-days 等价值
    基于标准定义：1 Petaflops-day = 8.64×10^19 FLOPS
    """
    try:
        # 标准换算：1 Petaflops-day = 1e15 FLOPS/s × 86400 s
        petaflops_days = total_flops / (1e15 * 86400)
        
        if petaflops_days >= 1000:
            return f"{petaflops_days:.1f} PF-days"
        elif petaflops_days >= 1:
            return f"{petaflops_days:.2f} PF-days"
        elif petaflops_days >= 0.001:
            return f"{petaflops_days*1000:.2f} TF-days"
        elif petaflops_days >= 0.000001:
            return f"{petaflops_days*1000000:.2f} GF-days"
        else:
            return f"{petaflops_days*1000000000:.2f} MF-days"
    except:
        return "0 PF-days"

def _is_overflow_value(flops: Union[float, int]) -> bool:
    """检查是否为溢出值"""
    try:
        if flops < 0 and abs(flops) > 1e15:
            return True
        if abs(flops) > 1e25:
            return True
        return False
    except:
        return True

# 更简洁的版本：只显示主要单位
def format_flops_concise(flops: Union[float, int]) -> str:
    """
    更简洁的格式化，只在大数值时显示 Petaflops-days
    """
    base_format = _format_base_flops(flops)
    
    if "OVERFLOW" in base_format or flops < 1e15:  # 小于 1 PFLOPS 时不显示
        return base_format
    
    pfd = flops / (1e15 * 86400)
    if pfd >= 0.01:  # 只有 >= 0.01 PF-days 时才显示
        return f"{base_format} (~{pfd:.2f} PF-days)"
    else:
        return base_format

if __name__ == "__main__":
    print("=== 简化 Petaflops-days 测试 ===")
    
    test_values = [
        ("小数值", 1.22e12),
        ("中等数值", 199.27e15),
        ("大数值", 6.38e18),
        ("超大数值", 1e21),
        ("溢出值", -5691793225967927296),
    ]
    
    print("完整格式（总是显示PF-days等价值）:")
    for name, value in test_values:
        result = format_flops_simple(value)
        print(f"  {name:10}: {result}")
    
    print("\n简洁格式（只在大数值时显示）:")
    for name, value in test_values:
        result = format_flops_concise(value)
        print(f"  {name:10}: {result}")
    
    print("\n=== 解释 ===")
    print("Petaflops-days 等价值基于标准定义：")
    print("1 Petaflops-day = 1×10¹⁵ FLOPS/s × 86400 s = 8.64×10¹⁹ FLOPS")
    print("这是一个固定的换算关系，不需要实际训练时间。")