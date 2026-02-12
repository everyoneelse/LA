# Packed模式验证数据集使用指南

本指南介绍如何在预训练中使用独立的验证数据集，以及如何评估已保存的checkpoint在新验证数据集上的性能。

## 功能概述

我们为您实现了两个主要功能：

1. **训练时指定独立验证数据集**：在packed模式下，支持使用与训练数据完全独立的验证数据集
2. **Checkpoint评估工具**：可以加载已保存的checkpoint并在新的验证数据集上评估loss变化

## 1. 训练时使用独立验证数据集

### 1.1 数据准备

首先准备您的验证数据集meta文件，格式与训练数据相同：

```json
[
  "val-00001-of-00010-a1b2c3d4e5f6g7h8.pkl",
  "val-00002-of-00010-b2c3d4e5f6g7h8i9.pkl",
  "val-00003-of-00010-c3d4e5f6g7h8i9j0.pkl",
  ...
]
```

### 1.2 训练命令

使用以下参数启动训练：

```bash
torchrun --nproc_per_node=8 \
    accessory/main_pretrain.py \
    --llama_type llama \
    --tokenizer_path ../tokenizer.model \
    --data_meta_path /path/to/train/PretrainMetaPacked.json \
    --data_root /path/to/train/data \
    --val_data_meta_path /path/to/val/PretrainMetaVal.json \
    --val_data_root /path/to/val/data \
    --packed_data \
    --max_words 2048 \
    --batch_size 4 \
    --accum_iter 4 \
    --lr 0.001 \
    --output_dir ./output_packed_separate_val \
    --save_freq 5000 \
    --val_freq 10000 \
    --data_parallel fsdp \
    --precision bf16
```

### 1.3 新增参数说明

- `--val_data_meta_path`: 验证数据集的meta文件路径（可选）
- `--val_data_root`: 验证数据的根目录（可选，默认使用`--data_root`）

如果不提供这两个参数，系统将使用原有行为（训练数据集的最后一个文件作为验证集）。

## 2. Checkpoint评估工具

### 2.1 单个Checkpoint评估

使用`evaluate_checkpoint.py`评估单个checkpoint：

```bash
python evaluate_checkpoint.py \
    --checkpoint_path /path/to/checkpoint/epoch1-iter50000 \
    --tokenizer_path ../tokenizer.model \
    --val_data_meta_path /path/to/new_val/PretrainMetaVal.json \
    --val_data_root /path/to/new_val/data \
    --packed_data \
    --batch_size 4 \
    --max_words 2048 \
    --precision bf16 \
    --output_file ./checkpoint_eval_results.json \
    --verbose
```

#### 参数说明：

**必需参数：**
- `--checkpoint_path`: checkpoint目录或文件路径
- `--val_data_meta_path`: 验证数据meta文件路径
- `--tokenizer_path`: tokenizer文件路径

**可选参数：**
- `--val_data_root`: 验证数据根目录
- `--packed_data`: 使用packed数据格式
- `--batch_size`: 批次大小（默认4）
- `--max_words`: 最大token长度（默认2048）
- `--precision`: 精度设置（默认bf16）
- `--device`: 设备选择（默认cuda）
- `--max_batches`: 限制评估批次数（用于快速测试）
- `--output_file`: 结果保存文件
- `--verbose`: 详细输出

#### 输出结果：

评估完成后会显示：
```
EVALUATION RESULTS
==================
Average Loss: 2.345678
Loss Std Dev: 0.123456
Min Loss: 1.987654
Max Loss: 2.876543
Total Batches: 1000
Total Samples: 4000
```

### 2.2 批量Checkpoint评估

使用`batch_evaluate_checkpoints.py`批量评估所有checkpoint：

```bash
python batch_evaluate_checkpoints.py \
    --output_dir /path/to/training/output \
    --tokenizer_path ../tokenizer.model \
    --val_data_meta_path /path/to/new_val/PretrainMetaVal.json \
    --val_data_root /path/to/new_val/data \
    --packed_data \
    --batch_size 4 \
    --results_dir ./evaluation_results \
    --plot_results \
    --min_iter 10000 \
    --max_iter 100000 \
    --iter_step 10000
```

#### 批量评估特有参数：

- `--output_dir`: 训练输出目录（包含所有checkpoint）
- `--results_dir`: 评估结果保存目录
- `--plot_results`: 生成loss曲线图
- `--min_iter`: 最小迭代次数过滤
- `--max_iter`: 最大迭代次数过滤
- `--iter_step`: 迭代步长过滤

#### 输出文件：

批量评估会生成以下文件：
- `evaluation_summary.csv`: CSV格式汇总结果
- `evaluation_summary.json`: JSON格式汇总结果
- `loss_curves.png`: loss曲线图（如果启用`--plot_results`）
- `{checkpoint_name}_results.json`: 每个checkpoint的详细结果

## 3. 使用示例

### 3.1 完整训练流程

```bash
# 1. 启动带独立验证集的训练
bash example_scripts/train_with_separate_val.sh

# 2. 训练完成后，评估所有checkpoint
bash example_scripts/batch_evaluate_all_checkpoints.sh

# 3. 或者只评估特定checkpoint
bash example_scripts/evaluate_checkpoint.sh
```

### 3.2 评估新验证数据集

如果训练完成后有新的验证数据集需要评估：

```bash
# 准备新的验证数据集meta文件
echo '["new_val_001.pkl", "new_val_002.pkl"]' > new_val_meta.json

# 评估最佳checkpoint
python evaluate_checkpoint.py \
    --checkpoint_path ./output_packed_separate_val/epoch2-iter100000 \
    --val_data_meta_path ./new_val_meta.json \
    --val_data_root /path/to/new_val_data \
    --packed_data \
    --output_file new_val_results.json
```

## 4. 技术细节

### 4.1 Checkpoint加载机制

评估脚本支持多种checkpoint格式：
1. 自动检测并使用`MetaModel.from_pretrained`方法
2. 备用手动加载机制处理各种checkpoint格式
3. 支持tensor parallel模型的合并加载

### 4.2 验证数据集加载

- 支持加载多个验证文件并自动合并
- 与训练数据集完全独立，不会影响训练过程
- 支持不同的数据根目录配置

### 4.3 内存优化

- 评估过程中使用`@torch.no_grad()`装饰器
- 支持混合精度评估
- 批量评估时每个checkpoint评估完成后自动清理GPU内存

## 5. 故障排除

### 5.1 常见问题

**问题1：checkpoint加载失败**
- 检查checkpoint路径是否正确
- 确认checkpoint文件完整性
- 尝试使用不同的加载方法

**问题2：验证数据集加载失败**
- 检查meta文件格式是否正确
- 确认所有pkl文件路径存在
- 检查数据根目录配置

**问题3：GPU内存不足**
- 减小batch_size
- 使用fp16精度
- 限制max_batches进行快速测试

### 5.2 调试技巧

```bash
# 使用verbose模式查看详细信息
python evaluate_checkpoint.py --verbose ...

# 限制批次数量进行快速测试
python evaluate_checkpoint.py --max_batches 10 ...

# 检查checkpoint内容
python -c "import torch; print(torch.load('checkpoint.pth').keys())"
```

## 6. 性能建议

1. **批量评估时**：使用`--iter_step`参数减少需要评估的checkpoint数量
2. **大数据集评估**：考虑使用`--max_batches`限制评估样本数
3. **多GPU环境**：单个评估脚本目前为单GPU设计，可并行运行多个评估任务
4. **存储优化**：评估结果包含所有loss值，大数据集时可能占用较多存储空间

## 7. 扩展功能

代码设计具有良好的扩展性，您可以：

1. 添加更多评估指标（perplexity、BLEU等）
2. 支持其他数据格式（非packed模式的增强）
3. 添加分布式评估支持
4. 集成到训练监控系统

---

通过以上功能，您可以灵活地管理验证数据集并高效地评估模型性能。如有任何问题，请参考示例脚本或查看源码中的详细注释。