#!/usr/bin/env python3
"""
正确的 Petaflops-days 实现
Petaflops-days = 总FLOPs / (1 Petaflops × 1天)
不需要任何时间信息！
"""

import math
from typing import Union

def format_flops(flops: Union[float, int]) -> str:
    """
    格式化 FLOPs 显示，修复溢出问题并添加 Petaflops-days
    
    Args:
        flops: FLOPs 数值
    
    Returns:
        格式化后的字符串，大数值时自动显示 Petaflops-days
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
    
    # 对于大数值，添加 Petaflops-days 显示
    if flops >= 1e15:  # >= 1 PFLOPS 时显示
        pfd = calculate_petaflops_days(flops)
        return f"{base_format} ({pfd})"
    
    return base_format

def calculate_petaflops_days(total_flops: float) -> str:
    """
    计算 Petaflops-days
    
    公式：Petaflops-days = 总FLOPs / (1×10^15 FLOPs/s × 86400 s/day)
    
    Args:
        total_flops: 总 FLOPs 数量
    
    Returns:
        Petaflops-days 格式的字符串
    """
    
    # 1 Petaflops-day = 1×10^15 FLOPS/s × 86400 s = 8.64×10^19 FLOPS
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

# 测试和验证
if __name__ == "__main__":
    print("=== 正确的 Petaflops-days 实现 ===")
    print()
    
    print("理解：Petaflops-days = 总FLOPs / (1 Petaflops × 1天)")
    print("公式：PF-days = 总FLOPs / (10^15 × 86400)")
    print("即：  PF-days = 总FLOPs / 8.64×10^19")
    print()
    
    # 验证计算
    one_petaflops_one_day = 1e15 * 86400  # 8.64×10^19 FLOPS
    print(f"1 Petaflops × 1天 = {one_petaflops_one_day:.2e} FLOPS")
    print()
    
    # 测试您日志中的数值
    test_values = [
        ("批次 FLOPs", 1.22e12),
        ("iter 130 总计", 199272.16e12),
        ("iter 140 总计", 398568.65e12),
        ("iter 150 总计", 797161.62e12),
        ("iter 160 总计", 1594347.57e12),
        ("iter 170 总计", 3188719.47e12),
        ("iter 180 总计", 6377463.26e12),
        ("溢出值", -5691793225967927296),
    ]
    
    print("测试结果:")
    for name, value in test_values:
        result = format_flops(value)
        print(f"  {name:15}: {result}")
    
    print()
    print("=== 手动验证 ===")
    
    # 手动验证一个计算
    test_flops = 6377463.26e12  # 6.38 EFLOPS
    manual_pfd = test_flops / (1e15 * 86400)
    auto_pfd = calculate_petaflops_days(test_flops)
    
    print(f"测试 FLOPs: {test_flops:.2e}")
    print(f"手动计算: {manual_pfd:.4f} PF-days")
    print(f"函数计算: {auto_pfd}")
    print()
    print("✅ 确认：不需要任何时间信息，只需要总 FLOPs！")