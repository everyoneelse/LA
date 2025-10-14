# 学习率调度策略快速使用指南

## 🚀 快速开始

### 1. 替换学习率调度器（推荐）

将 `enhanced_lr_sched.py` 复制并重命名为 `accessory/util/lr_sched.py`：

```bash
cp enhanced_lr_sched.py accessory/util/lr_sched.py
```

### 2. 修改训练脚本

在现有的训练脚本中添加学习率调度参数：

```bash
# 原有参数保持不变
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000

# 新增调度策略参数
--lr_schedule multi_cycle_cosine \
--cycle_length 50000 \
--cycle_decay_factor 0.8 \
--cycle_warmup_iters 1000
```

## 📊 支持的调度策略

| 策略名称 | 参数 | 适用场景 |
|---------|------|----------|
| `warmup_cosine` | 无额外参数 | 默认策略，稳定可靠 |
| `multi_cycle_cosine` | `--cycle_length`, `--cycle_decay_factor` | 长期训练，避免局部最优 |
| `pure_cosine` | `--total_iters` | 小模型，快速实验 |
| `linear_decay` | `--total_iters` | 简单稳定，计算高效 |
| `exponential_decay` | `--exp_decay_rate`, `--exp_decay_steps` | 快速收敛 |
| `multi_stage` | 无额外参数（使用默认配置） | 复杂训练需求 |

## 🔧 参数说明

### 多周期Cosine调度参数
- `--cycle_length`: 每个周期的步数（默认50000）
- `--cycle_decay_factor`: 每个周期后最大学习率的衰减因子（默认0.8）
- `--cycle_warmup_iters`: 每个周期内的warmup步数（默认1000）

### 指数衰减参数
- `--exp_decay_rate`: 衰减率（默认0.96）
- `--exp_decay_steps`: 衰减步长（默认10000）

## 📈 推荐配置

### 标准配置（当前使用）
```bash
--lr_schedule warmup_cosine
```

### 长期训练优化配置
```bash
--lr_schedule multi_cycle_cosine \
--cycle_length 40000 \
--cycle_decay_factor 0.85 \
--cycle_warmup_iters 2000
```

### 快速实验配置
```bash
--lr_schedule linear_decay \
--total_iters 100000
```

## 🧪 实验对比

运行不同策略的对比实验：

```bash
# 给脚本执行权限
chmod +x lr_schedule_examples.sh

# 运行对比实验
./lr_schedule_examples.sh config_path tokenizer_path data_meta_path data_root
```

## 📝 监控和调试

### 查看学习率曲线
训练过程中学习率会记录在tensorboard中，可以通过以下方式查看：

```bash
tensorboard --logdir output/your_experiment_name
```

### 调试学习率调度器
使用提供的测试函数：

```python
from enhanced_lr_sched import test_scheduler, get_lr_schedule_info

class Args:
    lr = 1e-4
    min_lr = 1e-5
    warmup_iters = 5000
    lr_decay_iters = 80000
    lr_schedule = 'multi_cycle_cosine'
    cycle_length = 20000
    cycle_decay_factor = 0.9

args = Args()
iterations, lrs = test_scheduler(args)
print(f"学习率范围: {min(lrs):.2e} - {max(lrs):.2e}")
```

## ⚠️ 注意事项

1. **兼容性**: 新的调度器完全兼容现有代码，默认使用原有的warmup_cosine策略
2. **参数验证**: 建议先在小规模数据上验证新参数的效果
3. **资源消耗**: 多周期调度可能需要更长的训练时间
4. **收敛监控**: 密切关注验证集loss，及时调整参数

## 🎯 最佳实践

1. **从默认开始**: 先使用默认的warmup_cosine策略建立基线
2. **小步迭代**: 逐步尝试新策略，每次只改变一个参数
3. **记录对比**: 详细记录不同策略的性能表现
4. **长期观察**: 多周期策略的优势可能在训练后期才显现

## 📚 更多资源

- `LLM_Learning_Rate_Schedules_Research.md`: 详细的调研报告
- `LLM_LR_Schedule_Literature_Review.md`: 文献综述
- `advanced_lr_schedulers.py`: 完整的调度器实现和可视化

---

**总结**: 多周期cosine调度是一个值得尝试的方向，特别适合长期训练的大模型。建议在现有基础上进行渐进式改进，而不是完全替换现有策略。