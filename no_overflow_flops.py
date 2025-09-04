#!/usr/bin/env python3
"""
完全避免溢出的 FLOPs 格式化
直接使用 Petaflops 作为基础单位存储和计算
"""

class FLOPsTracker:
    """使用 Petaflops 作为基础单位的 FLOPs 跟踪器"""
    
    def __init__(self):
        self.total_petaflops = 0.0  # 直接用 Petaflops 存储，避免溢出
        self.step_count = 0
    
    def add_step_flops(self, step_flops: float):
        """
        添加一步的 FLOPs
        
        Args:
            step_flops: 单步 FLOPs 数量
        """
        # 转换为 Petaflops 存储
        step_petaflops = step_flops / 1e15
        self.total_petaflops += step_petaflops
        self.step_count += 1
    
    def get_total_flops(self) -> float:
        """获取总 FLOPs（以原始单位）"""
        return self.total_petaflops * 1e15
    
    def get_total_petaflops(self) -> float:
        """获取总 Petaflops"""
        return self.total_petaflops
    
    def format_batch_flops(self, batch_flops: float) -> str:
        """格式化单批次 FLOPs"""
        return format_flops_simple(batch_flops)
    
    def format_total_flops(self) -> str:
        """格式化总 FLOPs"""
        return format_flops_simple(self.get_total_flops())
    
    def format_total_with_petaflops_days(self) -> str:
        """格式化总 FLOPs 并显示 Petaflops-days"""
        total_flops = self.get_total_flops()
        base_format = format_flops_simple(total_flops)
        
        # 计算 Petaflops-days（直接基于 Petaflops）
        petaflops_days = self.total_petaflops * 86400 / (86400)  # 简化为 total_petaflops
        petaflops_days = total_flops / (1e15 * 86400)
        
        if petaflops_days >= 1:
            pfd_str = f"{petaflops_days:.2f} PF-days"
        elif petaflops_days >= 0.001:
            pfd_str = f"{petaflops_days*1000:.2f} TF-days"
        elif petaflops_days >= 0.000001:
            pfd_str = f"{petaflops_days*1000000:.2f} GF-days"
        else:
            pfd_str = f"{petaflops_days*1000000000:.2f} MF-days"
        
        return f"{base_format} ({pfd_str})"

def format_flops_simple(flops: float) -> str:
    """
    简单的 FLOPs 格式化，不会溢出
    
    Args:
        flops: FLOPs 数值
    
    Returns:
        格式化后的字符串
    """
    
    # 处理异常情况
    if flops <= 0:
        return "0 FLOPS"
    
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

# 全局跟踪器实例
flops_tracker = FLOPsTracker()

def log_training_flops(batch_flops: float) -> str:
    """
    记录训练 FLOPs 并返回日志字符串
    
    Args:
        batch_flops: 单批次 FLOPs
    
    Returns:
        训练日志格式的字符串
    """
    
    # 添加到跟踪器
    flops_tracker.add_step_flops(batch_flops)
    
    # 格式化输出
    batch_str = flops_tracker.format_batch_flops(batch_flops)
    total_str = flops_tracker.format_total_with_petaflops_days()
    
    return f"{batch_str} (batch), {total_str} (total)"

def reset_flops_tracker():
    """重置 FLOPs 跟踪器"""
    global flops_tracker
    flops_tracker = FLOPsTracker()

# 测试代码
if __name__ == "__main__":
    print("=== 无溢出 FLOPs 跟踪测试 ===")
    print()
    
    # 重置跟踪器
    reset_flops_tracker()
    
    # 模拟您的训练过程
    batch_flops = 1.22e12  # 1.22 TFLOPS per batch
    
    print("模拟训练日志:")
    for step in range(220, 301, 10):
        log_str = log_training_flops(batch_flops)
        tokens_total = step * 32.768  # 模拟 tokens
        
        print(f"[Iter {step:3d}] Tokens: 32.7680K tokens (batch), "
              f"{tokens_total:.4f}M tokens (total) | FLOPs: {log_str}")
    
    print(f"\n当前跟踪器状态:")
    print(f"  总步数: {flops_tracker.step_count}")
    print(f"  总 Petaflops: {flops_tracker.get_total_petaflops():.6f}")
    print(f"  总 FLOPs: {flops_tracker.get_total_flops():.2e}")
    
    print("\n=== 直接替换方案 ===")
    print("只需要在您的代码中:")
    print("1. 初始化: flops_tracker = FLOPsTracker()")
    print("2. 每步调用: log_str = log_training_flops(batch_flops)")
    print("3. 打印: print(f'[Iter {step}] ... | FLOPs: {log_str}')")
    print("\n✅ 这样就完全不会溢出了！")