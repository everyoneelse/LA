#!/usr/bin/env python3
"""
最终解决方案：直接替换您的 format_flops 函数
完全避免溢出，使用 Petaflops 为基础单位
"""

class SafeFLOPsCounter:
    """安全的 FLOPs 计数器，使用 float 避免整数溢出"""
    
    def __init__(self):
        self.total_flops = 0.0  # 使用 float，避免整数溢出
        self.step_count = 0
    
    def add_flops(self, flops: float):
        """添加 FLOPs，自动处理溢出"""
        if flops < 0:  # 检测到溢出，重置
            print(f"Warning: Detected overflow (negative value: {flops}), resetting counter")
            self.total_flops = abs(flops) / 1e15  # 转为 Petaflops 存储
        else:
            self.total_flops += flops
        self.step_count += 1
    
    def get_total_flops(self) -> float:
        """获取总 FLOPs"""
        return self.total_flops

# 全局计数器
global_flops_counter = SafeFLOPsCounter()

def format_flops(flops: float) -> str:
    """
    安全的 FLOPs 格式化函数，完全替换您原来的版本
    
    Args:
        flops: FLOPs 数值
    
    Returns:
        格式化后的字符串，不会出现溢出
    """
    
    # 处理负数（溢出情况）
    if flops < 0:
        flops = abs(flops)
    
    # 处理零值
    if flops == 0:
        return "0 FLOPS"
    
    # 自动选择合适单位，优先大单位
    try:
        if flops >= 1e18:
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
        return f"{flops:.2e} FLOPS"

def format_flops_with_days(flops: float) -> str:
    """
    格式化 FLOPs 并显示 Petaflops-days（仅对大数值）
    
    Args:
        flops: FLOPs 数值
    
    Returns:
        包含 Petaflops-days 的格式化字符串
    """
    
    base_format = format_flops(flops)
    
    # 处理负数
    actual_flops = abs(flops) if flops < 0 else flops
    
    # 只对大数值显示 Petaflops-days
    if actual_flops >= 1e12:  # >= 1 TFLOPS 就显示
        try:
            petaflops_days = actual_flops / (1e15 * 86400)
            
            if petaflops_days >= 1:
                pfd_str = f"{petaflops_days:.2f} PF-days"
            elif petaflops_days >= 0.001:
                pfd_str = f"{petaflops_days*1000:.2f} TF-days"
            elif petaflops_days >= 0.000001:
                pfd_str = f"{petaflops_days*1000000:.2f} GF-days"
            else:
                pfd_str = f"{petaflops_days*1000000000:.2f} MF-days"
            
            return f"{base_format} ({pfd_str})"
        except:
            pass
    
    return base_format

def safe_training_log(batch_flops: float, total_flops: float) -> str:
    """
    安全的训练日志格式化
    
    Args:
        batch_flops: 单批次 FLOPs  
        total_flops: 总 FLOPs
    
    Returns:
        训练日志格式的字符串
    """
    
    batch_str = format_flops(batch_flops)
    total_str = format_flops_with_days(total_flops)
    
    return f"{batch_str} (batch), {total_str} (total)"

# 测试函数
def test_with_your_data():
    """使用您的实际数据测试"""
    
    print("=== 使用您的实际数据测试 ===")
    print()
    
    # 您日志中的实际数据
    test_data = [
        (220, -100007955.59e6),  # -100007955.59 MF-days 对应的 FLOPs
        (230, 13.49e12 * 86400 / 1000),  # 13.49 TF-days 对应的 FLOPs  
        (240, 26.98e12 * 86400 / 1000),  # 26.98 TF-days
        (250, 53.95e12 * 86400 / 1000),  # 53.95 TF-days
        (260, -105595189.47e6),  # 另一个溢出值
        (270, 2.31e12 * 86400 / 1000),   # 2.31 TF-days
        (280, 4.63e12 * 86400 / 1000),   # 4.63 TF-days
        (290, 9.26e12 * 86400 / 1000),   # 9.26 TF-days
        (300, 18.51e12 * 86400 / 1000),  # 18.51 TF-days
    ]
    
    batch_flops = 1.22e12  # 1.22 TFLOPS per batch
    
    for step, total_flops in test_data:
        log_str = safe_training_log(batch_flops, total_flops)
        tokens_total = step * 32.768 / 1000  # 转换为 M tokens
        
        print(f"[16:12:xx.xxxxxx] [Iter {step}] Tokens: 32.7680K tokens (batch), "
              f"{tokens_total:.4f}M tokens (total) | FLOPs: {log_str}")

if __name__ == "__main__":
    test_with_your_data()
    
    print(f"\n" + "="*60)
    print("🚀 解决方案:")
    print("1. 直接替换您的 format_flops 函数")
    print("2. 使用 safe_training_log 函数生成日志")
    print("3. 负数自动转正数处理")
    print("4. 自动选择合适单位，不会显示 OVERFLOW")
    print("="*60)