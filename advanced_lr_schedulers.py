"""
高级学习率调度器实现
包含多种LLM预训练中使用的学习率调度策略
"""

import math
import matplotlib.pyplot as plt
import numpy as np


class LRScheduler:
    """学习率调度器基类"""
    
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.base_lr = kwargs.get('lr', 1e-4)
        self.min_lr = kwargs.get('min_lr', 1e-5)
        
    def step(self, iteration):
        """更新学习率"""
        lr = self.get_lr(iteration)
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
    
    def get_lr(self, iteration):
        """获取当前迭代的学习率，子类需要实现"""
        raise NotImplementedError


class WarmupCosineScheduler(LRScheduler):
    """标准的Warmup + Cosine Decay调度器（当前repo使用的）"""
    
    def __init__(self, optimizer, warmup_iters=5000, decay_iters=400000, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
    
    def get_lr(self, iteration):
        if iteration < self.warmup_iters:
            # Linear warmup
            return self.base_lr * iteration / self.warmup_iters
        elif iteration > self.decay_iters:
            # Keep minimum learning rate
            return self.min_lr
        else:
            # Cosine decay
            decay_ratio = (iteration - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.min_lr + (self.base_lr - self.min_lr) * coeff


class PureCosineScheduler(LRScheduler):
    """纯Cosine调度器（无warmup）"""
    
    def __init__(self, optimizer, total_iters=400000, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.total_iters = total_iters
    
    def get_lr(self, iteration):
        decay_ratio = min(iteration / self.total_iters, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + (self.base_lr - self.min_lr) * coeff


class CosineAnnealingWithRestartsScheduler(LRScheduler):
    """多周期Cosine调度器（Cosine Annealing with Restarts）"""
    
    def __init__(self, optimizer, cycle_length=50000, decay_factor=0.8, 
                 warmup_iters=1000, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.cycle_length = cycle_length
        self.decay_factor = decay_factor  # 每个周期后最大学习率的衰减因子
        self.warmup_iters = warmup_iters
    
    def get_lr(self, iteration):
        # 计算当前在第几个周期
        cycle = iteration // self.cycle_length
        t_cur = iteration % self.cycle_length
        
        # 计算当前周期的最大学习率（逐渐衰减）
        current_max_lr = self.base_lr * (self.decay_factor ** cycle)
        
        # 每个周期内的warmup
        if t_cur < self.warmup_iters:
            # Linear warmup within cycle
            warmup_lr = current_max_lr * t_cur / self.warmup_iters
            return max(warmup_lr, self.min_lr)
        else:
            # Cosine annealing within cycle
            effective_t = (t_cur - self.warmup_iters) / (self.cycle_length - self.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * effective_t))
            return self.min_lr + (current_max_lr - self.min_lr) * coeff


class LinearDecayScheduler(LRScheduler):
    """线性衰减调度器"""
    
    def __init__(self, optimizer, warmup_iters=5000, total_iters=400000, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
    
    def get_lr(self, iteration):
        if iteration < self.warmup_iters:
            # Linear warmup
            return self.base_lr * iteration / self.warmup_iters
        else:
            # Linear decay
            decay_ratio = (iteration - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            decay_ratio = min(decay_ratio, 1.0)
            return self.base_lr * (1.0 - decay_ratio) + self.min_lr * decay_ratio


class ExponentialDecayScheduler(LRScheduler):
    """指数衰减调度器"""
    
    def __init__(self, optimizer, warmup_iters=5000, decay_rate=0.96, 
                 decay_steps=10000, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.warmup_iters = warmup_iters
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def get_lr(self, iteration):
        if iteration < self.warmup_iters:
            # Linear warmup
            return self.base_lr * iteration / self.warmup_iters
        else:
            # Exponential decay
            decay_steps_passed = (iteration - self.warmup_iters) / self.decay_steps
            lr = self.base_lr * (self.decay_rate ** decay_steps_passed)
            return max(lr, self.min_lr)


class PolynomialDecayScheduler(LRScheduler):
    """多项式衰减调度器"""
    
    def __init__(self, optimizer, warmup_iters=5000, total_iters=400000, 
                 power=1.0, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.power = power
    
    def get_lr(self, iteration):
        if iteration < self.warmup_iters:
            # Linear warmup
            return self.base_lr * iteration / self.warmup_iters
        else:
            # Polynomial decay
            decay_ratio = (iteration - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            decay_ratio = min(decay_ratio, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * ((1.0 - decay_ratio) ** self.power)
            return lr


class MultiStageScheduler(LRScheduler):
    """多阶段调度器（模拟GPT-4等大模型的复杂调度）"""
    
    def __init__(self, optimizer, stages=None, **kwargs):
        super().__init__(optimizer, **kwargs)
        # 默认三阶段：warmup -> stable -> decay
        if stages is None:
            stages = [
                {'end_iter': 5000, 'schedule': 'warmup'},
                {'end_iter': 200000, 'schedule': 'constant'},
                {'end_iter': 400000, 'schedule': 'cosine'}
            ]
        self.stages = stages
    
    def get_lr(self, iteration):
        for i, stage in enumerate(self.stages):
            if iteration <= stage['end_iter']:
                start_iter = self.stages[i-1]['end_iter'] if i > 0 else 0
                return self._get_stage_lr(iteration, start_iter, stage)
        
        # 超出所有阶段，返回最小学习率
        return self.min_lr
    
    def _get_stage_lr(self, iteration, start_iter, stage):
        stage_progress = (iteration - start_iter) / (stage['end_iter'] - start_iter)
        
        if stage['schedule'] == 'warmup':
            return self.base_lr * stage_progress
        elif stage['schedule'] == 'constant':
            return self.base_lr
        elif stage['schedule'] == 'cosine':
            coeff = 0.5 * (1.0 + math.cos(math.pi * stage_progress))
            return self.min_lr + (self.base_lr - self.min_lr) * coeff
        elif stage['schedule'] == 'linear':
            return self.base_lr * (1.0 - stage_progress) + self.min_lr * stage_progress
        else:
            return self.base_lr


def visualize_schedulers():
    """可视化不同调度器的学习率曲线"""
    
    # 创建一个虚拟的优化器（仅用于演示）
    class DummyOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': 1e-4}]
    
    optimizer = DummyOptimizer()
    total_iters = 100000
    iterations = range(0, total_iters, 1000)
    
    # 初始化不同的调度器
    schedulers = {
        'Warmup+Cosine (Current)': WarmupCosineScheduler(
            optimizer, warmup_iters=5000, decay_iters=80000),
        'Pure Cosine': PureCosineScheduler(
            optimizer, total_iters=total_iters),
        'Multi-Cycle Cosine': CosineAnnealingWithRestartsScheduler(
            optimizer, cycle_length=20000, decay_factor=0.9),
        'Linear Decay': LinearDecayScheduler(
            optimizer, warmup_iters=5000, total_iters=total_iters),
        'Exponential Decay': ExponentialDecayScheduler(
            optimizer, warmup_iters=5000, decay_rate=0.95, decay_steps=5000),
        'Multi-Stage': MultiStageScheduler(optimizer)
    }
    
    # 绘制学习率曲线
    plt.figure(figsize=(15, 10))
    
    for name, scheduler in schedulers.items():
        lrs = [scheduler.get_lr(it) for it in iterations]
        plt.plot(iterations, lrs, label=name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Comparison of Different Learning Rate Schedulers for LLM Pretraining')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('lr_schedulers_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 生成学习率调度器对比图
    visualize_schedulers()
    
    print("学习率调度器对比图已保存为 'lr_schedulers_comparison.png'")
    print("\n各调度器特点总结：")
    print("1. Warmup+Cosine: 标准策略，稳定可靠")
    print("2. Pure Cosine: 简单直接，适合小模型")
    print("3. Multi-Cycle Cosine: 多次重启，避免局部最优")
    print("4. Linear Decay: 计算简单，收敛稳定")
    print("5. Exponential Decay: 快速衰减，适合快速收敛")
    print("6. Multi-Stage: 灵活配置，适合复杂训练需求")