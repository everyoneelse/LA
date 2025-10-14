# LLM预训练学习率调度策略文献调研

## 主流LLM论文中的学习率调度策略

### 1. GPT系列

#### GPT-1 (2018)
- **策略**: Linear warmup + Linear decay
- **参数**: 学习率2.5e-4，warmup over first 2000 updates
- **特点**: 早期简单策略

#### GPT-2 (2019) 
- **策略**: Linear warmup + Cosine decay
- **参数**: 学习率6e-4，warmup 375M tokens
- **论文引用**: "We use a cosine learning rate schedule"

#### GPT-3 (2020)
- **策略**: Linear warmup + Cosine decay
- **参数**: 学习率6e-4，warmup over first 375M tokens
- **总训练**: 300B tokens
- **论文引用**: "We use a cosine decay schedule with learning rate dropping to 10% of its value"

#### GPT-4 (2023)
- **策略**: 未完全公开，推测使用多阶段cosine
- **特点**: 可能包含多个训练阶段，每个阶段不同的学习率策略

### 2. LLaMA系列

#### LLaMA (2023)
- **策略**: Linear warmup + Cosine decay
- **参数**: 
  - 学习率3e-4 (7B), 1.5e-4 (13B+)
  - Warmup 2000 steps
  - 最小学习率为最大学习率的10%
- **论文引用**: "We use the AdamW optimizer with β1 = 0.9, β2 = 0.95, and weight decay of 0.1. We use a cosine learning rate schedule"

#### LLaMA 2 (2023)
- **策略**: Linear warmup + Cosine decay（与LLaMA相同）
- **改进**: 更长的训练（2T tokens vs 1.4T）
- **参数**: 学习率3e-4，warmup 2000 steps

### 3. PaLM系列

#### PaLM (2022)
- **策略**: Inverse square root schedule with Adafactor
- **公式**: lr = base_lr / sqrt(max(step, warmup_steps))
- **特点**: 适合超大规模模型（540B参数）
- **论文引用**: "We use Adafactor optimizer with inverse square root learning rate schedule"

#### PaLM 2 (2023)
- **策略**: 改进的inverse square root + multi-stage
- **特点**: 针对不同训练阶段采用不同策略

### 4. Claude系列 (Anthropic)

#### Claude (2022)
- **策略**: 推测使用warmup + cosine with restarts
- **特点**: 长期训练，可能使用多周期策略

### 5. ChatGLM系列

#### ChatGLM-6B (2023)
- **策略**: Linear warmup + Cosine decay
- **参数**: 学习率5e-5，warmup 5000 steps
- **特点**: 针对中文优化

#### ChatGLM2-6B (2023)
- **策略**: 改进的multi-stage cosine
- **特点**: 预训练和指令调优使用不同策略

### 6. Falcon系列

#### Falcon-40B/180B (2023)
- **策略**: Linear warmup + Cosine decay
- **参数**: 学习率1.85e-4，warmup 1500 steps
- **特点**: 高质量数据集训练

---

## 学术研究中的新兴调度策略

### 1. Cosine Annealing with Warm Restarts (SGDR)

#### 原始论文
- **论文**: "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017)
- **作者**: Loshchilov & Hutter
- **核心思想**: 周期性重启学习率，帮助跳出局部最优

#### 在LLM中的应用
- **优势**: 长期训练中避免过早收敛
- **挑战**: 需要仔细调节周期长度
- **实践**: 一些研究组在大模型训练中尝试

### 2. Polynomial Decay

#### 理论基础
- **论文**: "Polynomial Learning Rate Schedules" (various)
- **公式**: lr = initial_lr * (1 - step/total_steps)^power
- **特点**: power=1时退化为线性衰减

#### 实际应用
- **BERT**: 使用polynomial decay (power=1)
- **T5**: 使用inverse square root类似效果

### 3. Exponential Decay

#### 传统应用
- **公式**: lr = initial_lr * decay_rate^(step/decay_steps)
- **特点**: 快速衰减，适合快速收敛场景
- **局限**: 在长期训练中可能过早衰减

### 4. Multi-Stage Scheduling

#### 实践案例
- **BERT**: 不同阶段使用不同学习率
- **RoBERTa**: 多阶段预训练策略
- **特点**: 灵活性高，但需要经验调节

---

## 最新研究趋势

### 1. Adaptive Learning Rate Schedules

#### 论文研究
- **"On the Variance of the Adaptive Learning Rate and Beyond"** (ICLR 2019)
- **核心**: 根据梯度统计自适应调整学习率

### 2. Layer-wise Learning Rate Scaling

#### 实践发现
- **不同层使用不同学习率**: 底层较小，顶层较大
- **论文**: "Layer-wise Adaptive Rate Scaling for Large Batch Training" (2017)

### 3. Gradient-based Schedule Adaptation

#### 新兴方向
- **根据梯度范数动态调整**: 避免梯度爆炸/消失
- **实时监控训练状态**: 自动调节学习率策略

---

## 实验对比研究

### 1. 大规模对比实验

#### "Scaling Laws for Neural Language Models" (2020)
- **发现**: 学习率调度对最终性能影响显著
- **建议**: Cosine decay优于线性衰减
- **数据**: 在多个模型规模上验证

#### "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- **发现**: 最优学习率与模型大小和数据量相关
- **建议**: 大模型需要更小的学习率

### 2. 调度策略消融实验

#### 常见发现
1. **Warmup的必要性**: 几乎所有大模型都需要warmup
2. **Cosine vs Linear**: Cosine decay通常优于线性衰减
3. **最小学习率**: 通常设为最大学习率的5-10%
4. **Warmup长度**: 通常为总训练步数的1-5%

---

## 实践建议总结

### 1. 默认选择
- **推荐**: Linear Warmup + Cosine Decay
- **原因**: 经过广泛验证，稳定可靠
- **适用**: 大部分LLM预训练场景

### 2. 长期训练优化
- **推荐**: Multi-cycle Cosine Annealing
- **原因**: 避免过早收敛，提升最终性能
- **适用**: 训练资源充足，追求最优性能

### 3. 快速实验
- **推荐**: Linear Decay或Exponential Decay
- **原因**: 计算简单，收敛较快
- **适用**: 资源受限，快速验证想法

### 4. 超大模型
- **推荐**: Multi-stage或Inverse Square Root
- **原因**: 更好的数值稳定性
- **适用**: 参数量>100B的模型

---

## 参数调优指南

### 1. 学习率范围
- **小模型** (<1B): 1e-3 ~ 5e-4
- **中等模型** (1B-10B): 5e-4 ~ 1e-4  
- **大模型** (>10B): 1e-4 ~ 5e-5

### 2. Warmup步数
- **经验公式**: warmup_steps = 0.01 * total_steps
- **最小值**: 1000 steps
- **最大值**: 10000 steps

### 3. 最小学习率
- **推荐比例**: min_lr = 0.1 * max_lr
- **下限**: 不低于1e-6
- **上限**: 不超过0.2 * max_lr

### 4. 周期长度（多周期调度）
- **推荐**: 总步数的10-20%
- **最小**: 10000 steps
- **调节**: 根据验证集性能动态调整

---

## 结论

1. **主流策略**: Warmup + Cosine Decay仍是最可靠的选择
2. **新兴趋势**: 多周期和自适应调度显示出潜力
3. **实践原则**: 先用成熟策略，再根据具体需求优化
4. **调参重点**: 学习率范围比调度形状更重要
5. **长期发展**: 向更智能、自适应的方向发展