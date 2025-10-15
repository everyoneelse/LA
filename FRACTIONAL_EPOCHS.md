# 支持小数 Epoch (Fractional Epochs Support)

## 概述 (Overview)

现在训练脚本支持小数 epoch，允许您更精确地控制训练时长。

The training script now supports fractional epochs, allowing you to have finer control over training duration.

## 使用方法 (Usage)

### 命令行参数 (Command Line Arguments)

在运行 `accessory/main_finetune.py` 时，可以将 `--epochs` 参数设置为小数：

When running `accessory/main_finetune.py`, you can set the `--epochs` argument to a decimal value:

```bash
# 训练 2.5 个 epoch (Train for 2.5 epochs)
python accessory/main_finetune.py --epochs 2.5 ...

# 训练 0.5 个 epoch (Train for 0.5 epochs - half an epoch)
python accessory/main_finetune.py --epochs 0.5 ...

# 训练 10.75 个 epoch (Train for 10.75 epochs)
python accessory/main_finetune.py --epochs 10.75 ...
```

## 工作原理 (How It Works)

1. **完整的 epoch**: 代码正常运行完整的 epoch
2. **小数部分**: 在最后一个 epoch，代码会计算需要训练的步数比例
   - 例如：0.5 epoch = 训练数据集的 50% 步数
   - 例如：0.25 epoch = 训练数据集的 25% 步数

1. **Full epochs**: The code runs complete epochs normally
2. **Fractional part**: For the last epoch, it calculates the proportion of steps to train
   - E.g., 0.5 epoch = 50% of dataset steps
   - E.g., 0.25 epoch = 25% of dataset steps

## 示例 (Examples)

### 示例 1：快速测试模型 (Quick Model Testing)

```bash
# 只训练 0.1 个 epoch 来快速测试代码
python accessory/main_finetune.py \
    --epochs 0.1 \
    --batch_size 4 \
    --llama_type llama \
    --data_config data_example/ShareGPT.json
```

### 示例 2：精确控制训练时长 (Precise Training Duration)

```bash
# 训练 3.5 个 epoch
python accessory/main_finetune.py \
    --epochs 3.5 \
    --batch_size 16 \
    --accum_iter 4
```

## 注意事项 (Notes)

1. **学习率调度**: 学习率调度器已经支持小数 epoch（通过 `warmup_epochs` 参数）
2. **保存检查点**: 在小数 epoch 结束时会正常保存检查点
3. **日志记录**: 训练日志会显示实际训练的步数

1. **Learning Rate Scheduling**: The LR scheduler already supports fractional epochs (via `warmup_epochs`)
2. **Checkpoint Saving**: Checkpoints are saved normally at the end of fractional epochs
3. **Logging**: Training logs show the actual number of steps trained

## 兼容性 (Compatibility)

- ✅ 支持 `warmup_epochs` 为小数 (已有功能)
- ✅ 支持 `epochs` 为小数 (新功能)
- ✅ 学习率调度器自动处理小数 epoch
- ✅ 分布式训练兼容

- ✅ Supports fractional `warmup_epochs` (existing feature)
- ✅ Supports fractional `epochs` (new feature)
- ✅ Learning rate scheduler automatically handles fractional epochs
- ✅ Compatible with distributed training
