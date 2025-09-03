# 神经网络预训练 FLOPs 计算指南

基于 `calculate-flops.pytorch` 库，计算神经网络预训练时每个 epoch 的前向传播和反向传播 FLOPs，并表达为与 `batch_size` 和 `seq_size` 相关的量。

## 🎯 核心功能

1. **前向传播 FLOPs 计算**: 使用 `calflops` 库精确计算
2. **反向传播 FLOPs 估算**: 通常为前向传播的 2 倍
3. **Epoch FLOPs 计算**: 考虑批次数和样本数
4. **公式推导**: 表达为 `batch_size` 和 `seq_size` 的函数

## 📦 安装依赖

```bash
# 克隆 calculate-flops.pytorch 库
git clone https://github.com/MrYxJ/calculate-flops.pytorch.git

# 安装 PyTorch（如果未安装）
pip install torch torchvision

# 安装 calflops（可选，推荐）
pip install calflops
```

## 🚀 快速开始

### 基本使用

```python
import torch
import torch.nn as nn
from lightweight_flops_calculator import calculate_pretrain_epoch_flops

# 创建你的模型
model = YourModel()

# 计算预训练 FLOPs
results = calculate_pretrain_epoch_flops(
    model=model,
    batch_size=16,           # 批量大小
    seq_size=1024,          # 序列长度
    num_samples_per_epoch=100000,  # 每个 epoch 样本数
    model_name="Your Model"
)

print(f"每个 epoch 前向传播 FLOPs: {results['epoch_forward_flops']}")
print(f"每个 epoch 反向传播 FLOPs: {results['epoch_backward_flops']}")
print(f"每个 epoch 总 FLOPs: {results['epoch_total_flops']}")
```

### 使用完整计算器

```python
from pretrain_flops_calculator import PretrainFLOPsCalculator

# 创建计算器
calculator = PretrainFLOPsCalculator(model, "My Model")

# 计算 epoch FLOPs
results = calculator.calculate_epoch_flops(
    batch_size=16,
    seq_size=1024,
    num_samples_per_epoch=100000
)

# 生成公式
formulas = calculator.generate_flops_formulas(results)
print(formulas['epoch_total'])  # 输出: Epoch_Total_FLOPs(L, N) ≈ 2.48e+07 × L × N
```

## 📊 计算结果示例

基于一个简单的 Transformer 模型（6.71M 参数）：

### 配置
- 批量大小: 8
- 序列长度: 512
- 每个 epoch 样本数: 10,000

### 结果
- **单批次前向传播**: 33.88 GFLOPs
- **单批次反向传播**: 67.75 GFLOPs (≈ 2x 前向)
- **单批次总计**: 101.63 GFLOPs

- **每个 epoch 前向传播**: 42.35 TFLOPs
- **每个 epoch 反向传播**: 84.69 TFLOPs
- **每个 epoch 总计**: 127.04 TFLOPs

## 📐 核心公式

设：
- `B` = batch_size（批量大小）
- `L` = seq_size（序列长度）
- `N` = num_samples_per_epoch（每个 epoch 样本数）
- `E` = num_epochs（总 epoch 数）

### 单批次 FLOPs
```
Forward_FLOPs(B, L) = C_forward × B × L
Backward_FLOPs(B, L) = C_backward × B × L ≈ 2 × Forward_FLOPs(B, L)
Total_FLOPs(B, L) = C_total × B × L
```

### 每个 Epoch FLOPs
```
Epoch_Forward_FLOPs(L, N) ≈ C_forward × L × N
Epoch_Backward_FLOPs(L, N) ≈ C_backward × L × N  
Epoch_Total_FLOPs(L, N) ≈ C_total × L × N
```

### 整个训练 FLOPs
```
Training_Total_FLOPs(L, N, E) ≈ C_total × L × N × E
```

其中 `C_forward`, `C_backward`, `C_total` 是模型特定的系数，通过实际测量得到。

## 🔍 关键观察

1. **线性缩放**: FLOPs 与 `batch_size` 和 `seq_size` 呈线性关系
2. **反向传播倍数**: 反向传播通常是前向传播的 2 倍计算量
3. **Transformer 特殊性**: 对于 Transformer 模型，注意力机制实际上使 FLOPs ∝ seq_size²
4. **系数稳定性**: 对于给定模型，FLOPs/Token 比率相对稳定

## 📈 缩放关系验证

测试结果显示：

| Batch Size | Seq Size | Forward FLOPs | Backward FLOPs | Total FLOPs |
|------------|----------|---------------|----------------|-------------|
| 4          | 128      | 4.23 GFLOPs   | 8.47 GFLOPs    | 12.7 GFLOPs |
| 8          | 256      | 16.94 GFLOPs  | 33.88 GFLOPs   | 50.82 GFLOPs|
| 16         | 512      | 67.75 GFLOPs  | 135.51 GFLOPs  | 203.26 GFLOPs|
| 32         | 1024     | 271.02 GFLOPs | 542.04 GFLOPs  | 813.06 GFLOPs|

可以看到：
- 当 batch_size 和 seq_size 都翻倍时，FLOPs 增加 4 倍
- 符合 FLOPs ∝ B × L 的线性关系

## 🛠️ 实际应用

### 1. 训练预算估算
```python
# 估算训练所需的总计算量
L = 2048        # 序列长度
N = 1000000     # 每个 epoch 样本数
E = 10          # 总 epochs
C = 2.48e+07    # 从测量得到的系数

total_flops = C * L * N * E
print(f"总训练 FLOPs: {format_flops(total_flops)}")
```

### 2. 硬件需求评估
```python
# 基于 FLOPs 评估训练时间
target_flops_per_second = 100e12  # 100 TFLOPs/s（假设硬件性能）
training_time_seconds = total_flops / target_flops_per_second
training_time_hours = training_time_seconds / 3600

print(f"预估训练时间: {training_time_hours:.1f} 小时")
```

### 3. 参数调优
```python
# 比较不同配置的计算开销
configs = [
    (16, 1024),   # 大批次，长序列
    (32, 512),    # 大批次，短序列  
    (8, 2048),    # 小批次，很长序列
]

for batch_size, seq_size in configs:
    epoch_flops = C * seq_size * num_samples_per_epoch
    print(f"B={batch_size}, L={seq_size}: {format_flops(epoch_flops)}")
```

## 📝 注意事项

1. **反向传播倍数**: 2.0 是一个常用的估算值，实际可能在 1.5-3.0 之间
2. **模型特异性**: 不同架构的模型系数 C 会不同
3. **序列长度影响**: 对于 Transformer，实际关系可能是 FLOPs ∝ L²（注意力机制）
4. **内存限制**: 大模型可能因内存不足而无法直接测量

## 🔧 自定义使用

可以根据具体需求修改：
- `backward_factor`: 调整反向传播倍数
- 添加更精确的 Transformer FLOPs 计算公式
- 支持更多模型架构
- 添加内存使用量估算

## 📚 参考资料

- [calculate-flops.pytorch](https://github.com/MrYxJ/calculate-flops.pytorch)
- [Backward/Forward FLOP Ratio](https://epochai.org/blog/backward-forward-FLOP-ratio)
- [Transformer FLOPs 计算](https://arxiv.org/abs/2001.08361)