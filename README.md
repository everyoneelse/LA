# 神经网络预训练 FLOPs 计算工具

基于 [calculate-flops.pytorch](https://github.com/MrYxJ/calculate-flops.pytorch) 库，计算神经网络预训练时每个 epoch 的前向传播和反向传播计算量，支持与 `batch_size` 和 `seq_size` 相关的表达式。

## 安装依赖

```bash
pip install calflops torch transformers
```

## 核心功能

### 1. 实际测量 FLOPs

使用 calflops 库实际测量模型的计算量：

```python
from calflops import calculate_flops
import torch
import torch.nn as nn

# 创建模型
model = YourModel()

# 计算前向传播 FLOPs
forward_flops, macs, params = calculate_flops(
    model=model,
    input_shape=(batch_size, seq_size),  # 或提供实际输入
    include_backPropagation=False,
    print_results=False,
    output_as_string=False
)

# 计算包含反向传播的 FLOPs
total_flops, _, _ = calculate_flops(
    model=model,
    input_shape=(batch_size, seq_size),
    include_backPropagation=True,
    compute_bp_factor=2.0,  # 反向传播通常是前向的2倍
    print_results=False,
    output_as_string=False
)

backward_flops = total_flops - forward_flops
```

### 2. 理论计算 FLOPs

对于 Transformer 模型，可以使用理论公式：

```python
def calculate_transformer_flops(batch_size, seq_size, hidden_size, num_layers, vocab_size, ffn_size):
    """
    Transformer 模型理论 FLOPs 计算
    
    公式：
    前向传播 = B * S * (2*H*V + L*(4*H² + 2*S*H + 2*H*F))
    反向传播 = 2 * 前向传播
    """
    B, S, H, L, V, F = batch_size, seq_size, hidden_size, num_layers, vocab_size, ffn_size
    
    # 嵌入层和输出层
    embedding_output_flops = 2 * B * S * H * V
    
    # 每层编码器
    attention_flops = B * S * (4 * H * H + 2 * S * H)  # 注意力机制
    ffn_flops = 2 * B * S * H * F                      # 前馈网络
    single_layer_flops = attention_flops + ffn_flops
    all_layers_flops = single_layer_flops * L
    
    # 总前向传播 FLOPs
    forward_flops = embedding_output_flops + all_layers_flops
    
    return forward_flops
```

### 3. 与 batch_size 和 seq_size 的关系

**关键发现：**
- FLOPs 与 `batch_size` 成**线性关系**
- FLOPs 与 `seq_size` 成**线性关系**（对于线性层）或**平方关系**（对于注意力机制）
- 实际使用中，通常可以近似为线性关系

**计算公式：**
```
单次前向传播 FLOPs = unit_flops_per_token * batch_size * seq_size
单次反向传播 FLOPs = 2 * unit_flops_per_token * batch_size * seq_size
单次训练总 FLOPs = 3 * unit_flops_per_token * batch_size * seq_size

每个 epoch FLOPs = 3 * unit_flops_per_token * total_samples * seq_size
```

其中 `unit_flops_per_token` 可以通过基准测量获得：
```python
unit_flops_per_token = base_forward_flops / (base_batch_size * base_seq_size)
```

## 使用示例

### 示例 1：简单线性模型

```python
# 运行 flops_demo.py 查看完整示例
python3 flops_demo.py
```

输出示例：
```
batch_size= 1, seq_size= 128: 前向= 1.21 GFLOPS, 反向= 2.42 GFLOPS, 总计= 3.63 GFLOPS
batch_size=32, seq_size= 512: 前向=154.67 GFLOPS, 反向=309.34 GFLOPS, 总计=464.01 GFLOPS

FLOPs 计算公式:
  前向传播 = 9.44e+06 * batch_size * seq_size
  反向传播 = 1.89e+07 * batch_size * seq_size
  总计 = 2.83e+07 * batch_size * seq_size

每个 epoch 的 FLOPs 公式:
  epoch_total_flops = 2.83e+07 * total_samples * seq_size
```

### 示例 2：Transformer 模型

```python
# 运行 transformer_flops_calculator.py 查看完整示例
python3 transformer_flops_calculator.py
```

输出示例：
```
BERT 类型模型 (hidden_size=768, num_layers=12):
测量结果 - 前向: 27.76 GFLOPS
测量结果 - 反向: 55.51 GFLOPS
测量结果 - 总计: 83.27 GFLOPS

FLOPs 计算表达式:
  forward_flops: 2.17e+08 * batch_size * seq_size
  epoch_total_flops: 6.51e+08 * total_samples * seq_size

LLaMA-7B 理论计算:
batch_size=1, seq_size=128: 前向传播: 665.32 GFLOPS, 训练总计: 2.00 TFLOPS
batch_size=32, seq_size=512: 前向传播: 86.81 TFLOPS, 训练总计: 260.43 TFLOPS
```

## 实际应用场景

### 场景 1：预训练资源估算

```python
# 假设训练 GPT 类模型
total_samples = 100_000_000  # 1亿个训练样本
batch_size = 32
seq_size = 1024
num_epochs = 3

# 使用测量得到的单位 FLOPs
unit_flops_per_token = 2.17e+08  # 从基准测量获得

# 计算总训练 FLOPs
total_training_flops = 3 * unit_flops_per_token * total_samples * seq_size * num_epochs
print(f"总训练 FLOPs: {total_training_flops:.2e}")
```

### 场景 2：不同配置对比

```python
configs = [
    {'batch_size': 16, 'seq_size': 512},
    {'batch_size': 32, 'seq_size': 512},
    {'batch_size': 32, 'seq_size': 1024}
]

for config in configs:
    epoch_flops = 3 * unit_flops_per_token * total_samples * config['seq_size']
    print(f"配置 {config}: {epoch_flops:.2e} FLOPs/epoch")
```

## 核心公式总结

### Transformer 模型理论公式

**前向传播 FLOPs：**
```
FLOPs_forward = B * S * (2*H*V + L*(4*H² + 2*S*H + 2*H*F))
```

**训练总 FLOPs（含反向传播）：**
```
FLOPs_training = 3 * FLOPs_forward
```

**每个 epoch FLOPs：**
```
FLOPs_epoch = FLOPs_training * (total_samples / batch_size)
            = 3 * B * S * (2*H*V + L*(4*H² + 2*S*H + 2*H*F)) * (total_samples / B)
            = 3 * total_samples * S * (2*H*V + L*(4*H² + 2*S*H + 2*H*F))
```

### 符号说明
- `B` = batch_size (批次大小)
- `S` = seq_size (序列长度)
- `H` = hidden_size (隐藏层维度)
- `L` = num_layers (层数)
- `V` = vocab_size (词汇表大小)
- `F` = ffn_size (前馈网络中间层维度，通常是 4*H)

## 注意事项

1. **反向传播倍数**：通常设置为 2.0，但可以根据具体模型调整
2. **注意力复杂度**：注意力机制的计算复杂度与 `seq_size²` 相关，但在实际计算中通常简化为线性关系
3. **激活重计算**：如果使用激活重计算（activation recomputation），需要将反向传播倍数调整为 3.0
4. **模型差异**：不同的模型架构（如 RoPE、Flash Attention 等）可能影响实际的 FLOPs

## 文件说明

- `neural_network_flops_calculator.py`: 通用神经网络 FLOPs 计算器
- `transformer_flops_calculator.py`: 专门针对 Transformer 模型的计算器
- `flops_demo.py`: 简单演示程序
- `README.md`: 本说明文档

## 参考资料

- [calculate-flops.pytorch](https://github.com/MrYxJ/calculate-flops.pytorch)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Backward Forward FLOP Ratio](https://epochai.org/blog/backward-forward-FLOP-ratio)