#!/usr/bin/env python3
"""
专门用于训练过程的 FLOPs 格式化工具
解决溢出问题，支持 Petaflops-days 计算
"""

import math
import time
from typing import Union, Optional

class TrainingFLOPsFormatter:
    """训练过程 FLOPs 格式化器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_flops = 0
        self.step_count = 0
    
    def format_flops(self, flops: Union[float, int], show_petaflops_days: bool = False) -> str:
        """
        格式化 FLOPs 显示，处理溢出并支持 Petaflops-days
        
        Args:
            flops: FLOPs 数值
            show_petaflops_days: 是否显示 Petaflops-days
        
        Returns:
            格式化后的字符串
        """
        
        # 检查溢出情况
        if self._is_overflow(flops):
            return "OVERFLOW"
        
        if flops < 0:
            return "OVERFLOW (negative)"
        
        if flops == 0:
            return "0 FLOPS"
        
        if math.isinf(flops) or math.isnan(flops):
            return "INVALID"
        
        # 格式化基本单位
        base_str = self._format_base_flops(flops)
        
        # 如果需要显示 Petaflops-days
        if show_petaflops_days:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                pfd_str = self._calculate_petaflops_days(flops, elapsed_time)
                return f"{base_str} ({pfd_str})"
        
        return base_str
    
    def _is_overflow(self, flops: Union[float, int]) -> bool:
        """检查是否溢出"""
        try:
            # 检查是否为负数且绝对值很大（溢出标志）
            if flops < 0 and abs(flops) > 1e15:
                return True
            
            # 检查是否超出合理范围
            if abs(flops) > 1e25:  # 超过 10^25 认为是溢出
                return True
                
            return False
        except:
            return True
    
    def _format_base_flops(self, flops: float) -> str:
        """基础 FLOPs 格式化"""
        if flops >= 1e21:  # 超大数值用科学计数法
            exponent = int(math.log10(flops))
            mantissa = flops / (10 ** exponent)
            return f"{mantissa:.2f}e{exponent} FLOPS"
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
    
    def _calculate_petaflops_days(self, total_flops: float, elapsed_seconds: float) -> str:
        """计算 Petaflops-days"""
        if elapsed_seconds <= 0:
            return "0 PF-days"
        
        # 1 Petaflops-day = 1e15 FLOPS/s * 86400 s = 8.64e19 FLOPS
        petaflops_days = total_flops / (1e15 * 86400)
        
        if petaflops_days >= 1000:
            return f"{petaflops_days:.1f} PF-days"
        elif petaflops_days >= 1:
            return f"{petaflops_days:.2f} PF-days"
        elif petaflops_days >= 0.001:
            # Teraflops-days
            return f"{petaflops_days*1000:.2f} TF-days"
        elif petaflops_days >= 0.000001:
            # Gigaflops-days  
            return f"{petaflops_days*1000000:.2f} GF-days"
        else:
            # Megaflops-days
            return f"{petaflops_days*1000000000:.2f} MF-days"
    
    def update_training_stats(self, step_flops: float):
        """更新训练统计信息"""
        if not self._is_overflow(step_flops):
            self.total_flops += step_flops
            self.step_count += 1
    
    def get_average_flops_per_step(self) -> float:
        """获取平均每步 FLOPs"""
        if self.step_count == 0:
            return 0
        return self.total_flops / self.step_count
    
    def reset(self):
        """重置统计信息"""
        self.start_time = time.time()
        self.total_flops = 0
        self.step_count = 0

# 便捷函数
def format_training_flops(flops: Union[float, int]) -> str:
    """
    便捷的训练 FLOPs 格式化函数
    专门处理训练过程中的溢出问题
    """
    formatter = TrainingFLOPsFormatter()
    return formatter.format_flops(flops)

# 使用示例
if __name__ == "__main__":
    print("=== 训练 FLOPs 格式化测试 ===")
    
    # 创建格式化器
    formatter = TrainingFLOPsFormatter()
    
    # 模拟您日志中的数值
    test_cases = [
        ("正常批次 FLOPs", 1.22e12),
        ("累计 FLOPs (iter 130)", 199272.16e12),
        ("累计 FLOPs (iter 140)", 398568.65e12), 
        ("累计 FLOPs (iter 150)", 797161.62e12),
        ("累计 FLOPs (iter 160)", 1594347.57e12),
        ("累计 FLOPs (iter 170)", 3188719.47e12),
        ("累计 FLOPs (iter 180)", 6377463.26e12),
        ("溢出值", -5691793225967927296),
    ]
    
    for name, value in test_cases:
        formatted = formatter.format_flops(value)
        formatted_with_pfd = formatter.format_flops(value, show_petaflops_days=True)
        print(f"{name:25}: {formatted:20} | {formatted_with_pfd}")
    
    print("\n=== 便捷函数测试 ===")
    print(f"便捷格式化: {format_training_flops(6377463.26e12)}")
    print(f"溢出处理:   {format_training_flops(-5691793225967927296)}")
    
    print("\n=== 训练统计模拟 ===")
    
    # 模拟训练过程
    formatter.reset()
    batch_flops = 1.22e12
    
    for step in range(1, 6):
        formatter.update_training_stats(batch_flops)
        total = formatter.total_flops
        avg = formatter.get_average_flops_per_step()
        
        print(f"Step {step:3d}: Batch={formatter.format_flops(batch_flops):12} | "
              f"Total={formatter.format_flops(total):12} | "
              f"Avg={formatter.format_flops(avg):12}")
        
        # 模拟时间流逝
        time.sleep(0.1)