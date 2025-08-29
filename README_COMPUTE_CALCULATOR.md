# InternLM2 训练 Compute 计算器

## 📋 概述

本项目提供了基于 OpenAI Scaling Laws 的 InternLM2 模型训练计算量（FLOPs）计算工具，帮助你准确估算不同配置下的训练资源需求。

## 🔑 核心发现

### OpenAI Scaling Law 公式
```
C = 6 × N × D
```
- `C`: 总训练计算量 (FLOPs)  
- `N`: 模型参数数量
- `D`: 训练 token 总数

### 计算方法对比
1. **OpenAI Scaling Law**: `6 × params × tokens` (理论上界)
2. **详细 FLOPs 计算**: 基于具体架构的精确计算 (~2x 更低)
3. **Chinchilla 最优**: 每参数使用 20 个 tokens

## 🚀 快速开始

### 运行演示
```bash
python3 quick_compute_demo.py
```

### 基本使用
```python
from internlm2_compute_calculator import InternLM2ComputeCalculator

# 加载模型配置
calculator = InternLM2ComputeCalculator("config.json")

# 计算训练 compute
results = calculator.calculate_training_compute(
    batch_size=4,
    seq_len=2048,
    total_tokens=50_000_000_000  # 50B tokens
)

print(f"训练计算量: {calculator.format_number(results['compute_estimates']['openai_scaling_law'], 'FLOPs')}")
```

## 📊 InternLM2 模型对比

| 模型 | 参数量 | Chinchilla 最优 Tokens | 训练 Compute | 预估训练时间* |
|------|--------|----------------------|-------------|-------------|
| 126M | 120M | 2.4B | 1.7P FLOPs | 0.2 天 |
| 355M | 312M | 6.2B | 11.7P FLOPs | 1.4 天 |
| 567M | 511M | 10.2B | 31.3P FLOPs | 3.6 天 |
| 992M | 851M | 17.0B | 86.8P FLOPs | 10.0 天 |
| 1.4B | 1.18B | 23.7B | 168.3P FLOPs | 19.5 天 |

\* 基于 100 TFLOPs/s 有效吞吐量估算

## 🛠️ 文件说明

- `internlm2_compute_calculator.py` - 主要计算器实现
- `quick_compute_demo.py` - 快速演示脚本  
- `COMPUTE_CALCULATION_RESEARCH.md` - 详细调研报告
- `test_flop_counter.py` - 测试脚本
- `internlm2_scaling/configs/` - 模型配置文件

## 💡 实用建议

### 资源规划
1. 使用 OpenAI Scaling Law 进行初步估算
2. 考虑 1.5-2x 安全边际
3. 根据硬件效率调整实际需求

### 训练策略  
1. 遵循 Chinchilla 最优比例 (20 tokens/param)
2. 使用梯度累积模拟大 batch size
3. 监控 GPU 利用率和训练效率

### 成本控制
1. 先用小模型验证超参数
2. 采用混合精度训练 (FP16/BF16)
3. 考虑检查点和恢复策略

## 🔍 关键洞察

- **OpenAI 公式 vs 详细计算**: OpenAI 公式通常高估 ~2x，详细计算更准确
- **Batch Size 影响**: 不影响总计算量，只影响训练时间和内存使用
- **架构优化**: InternLM2 的 GQA 和 SwiGLU 优化减少了实际计算量
- **硬件效率**: 实际 GPU 利用率通常在 30-70% 之间

## 📈 使用场景

- **模型设计**: 评估不同架构的计算成本
- **资源规划**: 估算训练所需的计算资源  
- **成本预算**: 计算训练的时间和费用
- **超参优化**: 在计算预算约束下优化配置

## 🔧 扩展功能

计算器支持：
- 多种计算方法对比
- 批量模型分析
- 自定义训练配置
- 人性化的结果格式化
- Chinchilla 最优 token 计算

## ⚠️ 注意事项

- 公式给出理论计算量，实际会受硬件效率影响
- 大模型训练往往受内存限制而非计算限制  
- 需要考虑激活重计算等内存优化策略
- 混合精度训练可减少内存但 FLOPs 基本不变

## 📚 参考文献

1. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models"
2. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models"  
3. InternLM2 Technical Report

---

**快速上手**: 运行 `python3 quick_compute_demo.py` 开始体验！