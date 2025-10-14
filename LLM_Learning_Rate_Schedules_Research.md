# LLM预训练学习率调度策略调研报告

## 当前Repo分析

### 现有实现
当前repo使用的是经典的 **Warmup + Cosine Decay** 学习率调度策略：

```python
def adjust_learning_rate(optimizer, it, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if it < args.warmup_iters: # 1) linear warmup for warmup_iters steps
        lr = args.lr * it / args.warmup_iters
    elif it > args.lr_decay_iters: # 2) if it > lr_decay_iters, return min learning rate
        lr = args.min_lr
    else: # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        lr = args.min_lr + (args.lr - args.min_lr) * coeff
```

### 参数配置
- 学习率: 0.0001 (1e-4)
- 最小学习率: 0.00001 (1e-5)  
- Warmup步数: 5000
- 衰减步数: 400000
- 特点: **单次上升，单次下降**

---

## 其他LLM预训练学习率调度策略调研

### 1. 直接Cosine调度（无Warmup）

#### 实现方式
```python
def cosine_schedule(optimizer, it, args):
    """Pure cosine schedule without warmup"""
    decay_ratio = it / args.total_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    lr = args.min_lr + (args.lr - args.min_lr) * coeff
```

#### 优缺点
- **优点**: 简单直接，避免了warmup阶段的不稳定性
- **缺点**: 初始学习率过高可能导致训练不稳定
- **适用场景**: 小模型或已有良好初始化的模型

### 2. 多周期Cosine调度（Cosine Annealing with Restarts）

#### 实现方式
```python
def cosine_annealing_with_restarts(optimizer, it, args):
    """Multiple cycles of cosine annealing"""
    cycle_length = args.cycle_length
    cycle = it // cycle_length
    t_cur = it % cycle_length
    
    # Cosine annealing within current cycle
    coeff = 0.5 * (1.0 + math.cos(math.pi * t_cur / cycle_length))
    lr = args.min_lr + (args.lr - args.min_lr) * coeff
    
    # Optional: decay max lr after each cycle
    if args.decay_factor < 1.0:
        max_lr = args.lr * (args.decay_factor ** cycle)
        lr = args.min_lr + (max_lr - args.min_lr) * coeff
```

#### 特点
- **多次上升下降**: 每个周期都有完整的cosine曲线
- **周期性重启**: 帮助跳出局部最优
- **逐渐衰减**: 可选择每个周期降低最大学习率

### 3. 其他常见调度策略

#### 3.1 线性衰减 (Linear Decay)
```python
def linear_decay(optimizer, it, args):
    if it < args.warmup_iters:
        lr = args.lr * it / args.warmup_iters
    else:
        decay_ratio = (it - args.warmup_iters) / (args.total_iters - args.warmup_iters)
        lr = args.lr * (1.0 - decay_ratio)
```

#### 3.2 指数衰减 (Exponential Decay)
```python
def exponential_decay(optimizer, it, args):
    if it < args.warmup_iters:
        lr = args.lr * it / args.warmup_iters
    else:
        lr = args.lr * (args.decay_rate ** ((it - args.warmup_iters) / args.decay_steps))
```

#### 3.3 多项式衰减 (Polynomial Decay)
```python
def polynomial_decay(optimizer, it, args):
    if it < args.warmup_iters:
        lr = args.lr * it / args.warmup_iters
    else:
        decay_ratio = (it - args.warmup_iters) / (args.total_iters - args.warmup_iters)
        lr = args.min_lr + (args.lr - args.min_lr) * ((1.0 - decay_ratio) ** args.power)
```

---

## 主流LLM使用的学习率调度策略

### GPT系列
- **GPT-3**: Warmup + Cosine Decay (与当前repo类似)
- **GPT-4**: 推测使用改进的cosine调度，可能包含多阶段

### LLaMA系列  
- **LLaMA 1/2**: Warmup + Cosine Decay
- **参数**: warmup 2000步，总训练1.4T tokens

### PaLM系列
- **PaLM**: 使用Adafactor优化器配合inverse square root调度
- **特点**: 更适合超大规模模型

### ChatGLM系列
- **ChatGLM**: Warmup + Cosine Decay
- **特点**: 中文优化的调度参数

---

## 实验对比与建议

### 调度策略选择建议

1. **标准预训练**: 
   - 推荐: Warmup + Cosine Decay (当前实现)
   - 稳定可靠，被广泛验证

2. **长期训练/大模型**:
   - 推荐: 多阶段cosine或多周期cosine
   - 帮助避免过早收敛

3. **资源受限/快速实验**:
   - 推荐: 线性衰减或指数衰减
   - 计算开销更小

### 参数调优建议

1. **Warmup步数**: 通常为总步数的1-5%
2. **最小学习率**: 通常为最大学习率的5-10%
3. **学习率范围**: 根据模型大小调整，大模型通常需要更小的学习率

---

## 调研结论与实践建议

### 主要发现

1. **当前策略评估**: 你的repo使用的Warmup + Cosine Decay是业界主流且经过充分验证的策略，GPT-3、LLaMA、ChatGLM等主要模型都采用此策略。

2. **直接Cosine调度**: 确实存在直接使用cosine学习率的实现，主要用于：
   - 小规模模型或实验
   - 已有良好初始化的模型
   - 快速原型验证场景

3. **多周期调度**: 多周期上升下降的学习率调度（Cosine Annealing with Restarts）在学术界有广泛研究：
   - **优势**: 帮助跳出局部最优，提升长期训练效果
   - **应用**: 适合长期训练的大模型
   - **挑战**: 需要仔细调节周期参数

### 具体实现建议

#### 1. 保持现有策略（推荐）
- 当前的Warmup + Cosine Decay策略已经很优秀
- 可以微调参数：调整warmup步数、最小学习率比例

#### 2. 尝试多周期Cosine（实验性）
```bash
# 在训练脚本中添加参数
--lr_schedule multi_cycle_cosine \
--cycle_length 50000 \
--cycle_decay_factor 0.8 \
--cycle_warmup_iters 1000
```

#### 3. 渐进式改进策略
1. **第一阶段**: 在小规模数据上对比不同策略
2. **第二阶段**: 选择最优策略在中等规模上验证
3. **第三阶段**: 全规模训练采用验证过的策略

### 参数调优建议

1. **多周期调度参数**:
   - 周期长度: 总训练步数的10-20%
   - 衰减因子: 0.8-0.9（每个周期后最大学习率的衰减）
   - 周期内warmup: 1000-5000步

2. **学习率范围**:
   - 根据模型大小调整：大模型需要更小的学习率
   - 最小学习率通常为最大学习率的5-10%

### 实施路径

1. **立即可行**: 使用提供的`enhanced_lr_sched.py`替换现有的学习率调度器
2. **实验验证**: 运行`lr_schedule_examples.sh`中的不同策略对比
3. **生产部署**: 根据实验结果选择最适合的策略

### 长期发展趋势

- **自适应调度**: 根据训练状态动态调整
- **层级调度**: 不同层使用不同学习率
- **梯度感知**: 基于梯度统计的智能调度

多周期调度确实是一个值得尝试的方向，特别是对于需要长期训练的大模型，它可以帮助模型在训练后期重新获得探索能力，避免过早陷入局部最优。