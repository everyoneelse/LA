"""
增强版学习率调度器 - 可直接替换accessory/util/lr_sched.py
支持多种调度策略，包括多周期cosine调度
"""

import math


def adjust_learning_rate(optimizer, it, args):
    """
    根据args.lr_schedule选择不同的学习率调度策略
    保持与原有接口的兼容性
    """
    schedule_type = getattr(args, 'lr_schedule', 'warmup_cosine')
    
    if schedule_type == 'warmup_cosine':
        lr = _warmup_cosine_schedule(it, args)
    elif schedule_type == 'pure_cosine':
        lr = _pure_cosine_schedule(it, args)
    elif schedule_type == 'multi_cycle_cosine':
        lr = _multi_cycle_cosine_schedule(it, args)
    elif schedule_type == 'linear_decay':
        lr = _linear_decay_schedule(it, args)
    elif schedule_type == 'exponential_decay':
        lr = _exponential_decay_schedule(it, args)
    elif schedule_type == 'polynomial_decay':
        lr = _polynomial_decay_schedule(it, args)
    elif schedule_type == 'multi_stage':
        lr = _multi_stage_schedule(it, args)
    else:
        # 默认使用原有的warmup_cosine策略
        lr = _warmup_cosine_schedule(it, args)
    
    # 应用学习率到优化器
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    
    return lr


def _warmup_cosine_schedule(it, args):
    """原有的warmup + cosine decay策略"""
    if it < args.warmup_iters:
        lr = args.lr * it / args.warmup_iters
    elif it > args.lr_decay_iters:
        lr = args.min_lr
    else:
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = args.min_lr + (args.lr - args.min_lr) * coeff
    return lr


def _pure_cosine_schedule(it, args):
    """纯cosine调度（无warmup）"""
    total_iters = getattr(args, 'total_iters', args.lr_decay_iters)
    decay_ratio = min(it / total_iters, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    lr = args.min_lr + (args.lr - args.min_lr) * coeff
    return lr


def _multi_cycle_cosine_schedule(it, args):
    """多周期cosine调度"""
    cycle_length = getattr(args, 'cycle_length', 50000)
    decay_factor = getattr(args, 'cycle_decay_factor', 0.8)
    warmup_iters = getattr(args, 'cycle_warmup_iters', 1000)
    
    # 计算当前周期
    cycle = it // cycle_length
    t_cur = it % cycle_length
    
    # 当前周期的最大学习率（逐渐衰减）
    current_max_lr = args.lr * (decay_factor ** cycle)
    
    if t_cur < warmup_iters:
        # 周期内warmup
        warmup_lr = current_max_lr * t_cur / warmup_iters
        lr = max(warmup_lr, args.min_lr)
    else:
        # 周期内cosine衰减
        effective_t = (t_cur - warmup_iters) / (cycle_length - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * effective_t))
        lr = args.min_lr + (current_max_lr - args.min_lr) * coeff
    
    return lr


def _linear_decay_schedule(it, args):
    """线性衰减调度"""
    if it < args.warmup_iters:
        lr = args.lr * it / args.warmup_iters
    else:
        total_iters = getattr(args, 'total_iters', args.lr_decay_iters)
        decay_ratio = (it - args.warmup_iters) / (total_iters - args.warmup_iters)
        decay_ratio = min(decay_ratio, 1.0)
        lr = args.lr * (1.0 - decay_ratio) + args.min_lr * decay_ratio
    return lr


def _exponential_decay_schedule(it, args):
    """指数衰减调度"""
    if it < args.warmup_iters:
        lr = args.lr * it / args.warmup_iters
    else:
        decay_rate = getattr(args, 'exp_decay_rate', 0.96)
        decay_steps = getattr(args, 'exp_decay_steps', 10000)
        decay_steps_passed = (it - args.warmup_iters) / decay_steps
        lr = args.lr * (decay_rate ** decay_steps_passed)
        lr = max(lr, args.min_lr)
    return lr


def _polynomial_decay_schedule(it, args):
    """多项式衰减调度"""
    if it < args.warmup_iters:
        lr = args.lr * it / args.warmup_iters
    else:
        total_iters = getattr(args, 'total_iters', args.lr_decay_iters)
        power = getattr(args, 'poly_power', 1.0)
        decay_ratio = (it - args.warmup_iters) / (total_iters - args.warmup_iters)
        decay_ratio = min(decay_ratio, 1.0)
        lr = args.min_lr + (args.lr - args.min_lr) * ((1.0 - decay_ratio) ** power)
    return lr


def _multi_stage_schedule(it, args):
    """多阶段调度"""
    # 默认三阶段配置，可通过args.stages自定义
    stages = getattr(args, 'stages', [
        {'end_iter': args.warmup_iters, 'schedule': 'warmup'},
        {'end_iter': args.lr_decay_iters // 2, 'schedule': 'constant'},
        {'end_iter': args.lr_decay_iters, 'schedule': 'cosine'}
    ])
    
    for i, stage in enumerate(stages):
        if it <= stage['end_iter']:
            start_iter = stages[i-1]['end_iter'] if i > 0 else 0
            stage_progress = (it - start_iter) / (stage['end_iter'] - start_iter)
            
            if stage['schedule'] == 'warmup':
                lr = args.lr * stage_progress
            elif stage['schedule'] == 'constant':
                lr = args.lr
            elif stage['schedule'] == 'cosine':
                coeff = 0.5 * (1.0 + math.cos(math.pi * stage_progress))
                lr = args.min_lr + (args.lr - args.min_lr) * coeff
            elif stage['schedule'] == 'linear':
                lr = args.lr * (1.0 - stage_progress) + args.min_lr * stage_progress
            else:
                lr = args.lr
            return lr
    
    # 超出所有阶段
    return args.min_lr


def adjust_learning_rate_epoch(optimizer, epoch, args):
    """基于epoch的学习率调度（保持原有接口兼容性）"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_lr_schedule_info(args):
    """获取当前学习率调度策略的信息"""
    schedule_type = getattr(args, 'lr_schedule', 'warmup_cosine')
    
    info = {
        'schedule_type': schedule_type,
        'base_lr': args.lr,
        'min_lr': args.min_lr,
        'warmup_iters': args.warmup_iters,
    }
    
    if schedule_type in ['warmup_cosine', 'linear_decay', 'polynomial_decay']:
        info['decay_iters'] = args.lr_decay_iters
    elif schedule_type == 'multi_cycle_cosine':
        info['cycle_length'] = getattr(args, 'cycle_length', 50000)
        info['decay_factor'] = getattr(args, 'cycle_decay_factor', 0.8)
    elif schedule_type == 'exponential_decay':
        info['decay_rate'] = getattr(args, 'exp_decay_rate', 0.96)
        info['decay_steps'] = getattr(args, 'exp_decay_steps', 10000)
    
    return info


# 为了方便测试和可视化，提供一个简单的测试函数
def test_scheduler(args, max_iters=100000, step=1000):
    """测试学习率调度器"""
    iterations = list(range(0, max_iters, step))
    learning_rates = []
    
    class DummyOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': args.lr}]
    
    optimizer = DummyOptimizer()
    
    for it in iterations:
        lr = adjust_learning_rate(optimizer, it, args)
        learning_rates.append(lr)
    
    return iterations, learning_rates


if __name__ == "__main__":
    # 简单测试
    class Args:
        def __init__(self):
            self.lr = 1e-4
            self.min_lr = 1e-5
            self.warmup_iters = 5000
            self.lr_decay_iters = 80000
            self.lr_schedule = 'multi_cycle_cosine'
            self.cycle_length = 20000
            self.cycle_decay_factor = 0.9
    
    args = Args()
    iterations, lrs = test_scheduler(args)
    
    print(f"测试调度器: {args.lr_schedule}")
    print(f"学习率范围: {min(lrs):.2e} - {max(lrs):.2e}")
    print(f"调度器信息: {get_lr_schedule_info(args)}")