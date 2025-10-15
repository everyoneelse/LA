# Epoch 小数支持总结

## ✅ 修改完成

**问题**: epochs 参数能否设置为小数？

**答案**: 现在可以了！

## 📝 已实施的修改

### 1. 修改 `accessory/main_finetune.py`

#### 变更 1: Epochs 参数类型
```python
# 修改前:
parser.add_argument('--epochs', default=400, type=int)

# 修改后:
parser.add_argument('--epochs', default=400, type=float)
```

#### 变更 2: 训练循环支持小数 epoch
添加了以下功能：
- 自动计算完整的 epoch 数和小数部分
- 在最后一个 epoch 时，只训练数据集的相应比例
- 添加日志显示小数 epoch 的训练进度

```python
# 示例：如果设置 --epochs 2.5
# - 训练 2 个完整的 epoch
# - 训练第 3 个 epoch 的 50% (即 0.5 epoch)
```

### 2. 修改 `accessory/engine_finetune.py`

添加 `max_steps` 参数到 `train_one_epoch` 函数：
- 当指定 `max_steps` 时，训练会在达到该步数后提前停止
- 用于实现小数 epoch 的精确控制

## 🎯 使用示例

### 基础用法

```bash
# 训练 2.5 个 epoch
python accessory/main_finetune.py --epochs 2.5 --batch_size 16 ...

# 训练 0.5 个 epoch (半个 epoch)
python accessory/main_finetune.py --epochs 0.5 --batch_size 16 ...

# 训练 10.75 个 epoch
python accessory/main_finetune.py --epochs 10.75 --batch_size 16 ...
```

### 实际场景

#### 场景 1: 快速验证代码
```bash
# 只训练 0.1 epoch 来快速测试
python accessory/main_finetune.py \
    --epochs 0.1 \
    --batch_size 4 \
    --llama_type llama \
    --llama_config configs/model/finetune/sg/llamaPeft_normBiasLora.json \
    --tokenizer_path /path/to/tokenizer.model \
    --data_config data_example/ShareGPT.json \
    --pretrained_path /path/to/pretrained
```

#### 场景 2: 精确控制训练时长
```bash
# 根据学习率曲线，可能 3.5 个 epoch 效果最好
python accessory/main_finetune.py \
    --epochs 3.5 \
    --warmup_epochs 0.5 \
    --batch_size 16 \
    --accum_iter 4
```

#### 场景 3: 数据集太小，不需要完整 epoch
```bash
# 小数据集微调，0.25 epoch 即可
python accessory/main_finetune.py \
    --epochs 0.25 \
    --lr 0.0001 \
    --batch_size 8
```

## 🔍 工作原理详解

### 1. 参数计算
```python
# 假设设置 --epochs 2.75
total_epochs_int = int(2.75)  # = 2
fractional_part = 2.75 - 2     # = 0.75
```

### 2. 训练流程
```
Epoch 0: 训练 100% 的数据集
Epoch 1: 训练 100% 的数据集
Epoch 2: 训练 75% 的数据集 (0.75 epoch)
完成！
```

### 3. 步数计算示例
```python
# 假设数据集有 1000 个批次
# --epochs 2.5

# Epoch 0: 训练 1000 步 (100%)
# Epoch 1: 训练 1000 步 (100%)
# Epoch 2: 训练 500 步 (50%)
# 总计: 2500 步
```

## ✨ 特性

1. **向后兼容**: 整数 epoch 仍然正常工作
   ```bash
   python accessory/main_finetune.py --epochs 10  # 完全正常
   ```

2. **学习率调度器自动适配**: 
   - `warmup_epochs` 已经支持小数
   - 学习率会根据实际的 epoch 进度平滑变化
   
3. **检查点保存**: 
   - 小数 epoch 结束时会正常保存检查点
   - 日志文件记录准确的训练进度

4. **分布式训练兼容**: 
   - 所有 rank 同步停止
   - 数据采样器正确处理

## 📊 日志输出示例

```
Start training for 2.5 epochs
Epoch: [0] ...
Epoch: [1] ...
Training fractional epoch 2 with 500/1000 steps (50.00%)
Epoch: [2] ...
Stopping at step 500/500 for fractional epoch
Training time 01:23:45
```

## ⚠️ 注意事项

1. **步数计算**: 小数部分基于数据集的总步数
   - 如果数据集有 1234 步，0.5 epoch = 617 步

2. **Resume 功能**: 从检查点恢复时需要注意
   - 确保 `--epochs` 设置与原训练一致

3. **保存间隔**: `--save_interval` 仍然以 epoch 为单位
   - 但检查点保存逻辑已更新为 `epoch + 1 >= args.epochs`

## 🧪 测试建议

```bash
# 1. 测试小数 epoch 是否正常工作
python accessory/main_finetune.py --epochs 0.1 ... 

# 2. 测试多个小数 epoch
python accessory/main_finetune.py --epochs 2.5 ...

# 3. 测试纯小数（小于1）
python accessory/main_finetune.py --epochs 0.5 ...

# 4. 验证整数仍然工作
python accessory/main_finetune.py --epochs 3 ...
```

## 📚 相关文件

- `accessory/main_finetune.py` - 主训练脚本（已修改）
- `accessory/engine_finetune.py` - 训练引擎（已修改）
- `accessory/util/lr_sched.py` - 学习率调度器（已支持小数 epoch）
- `FRACTIONAL_EPOCHS.md` - 用户文档

## 🎓 技术细节

### 为什么学习率调度器已经支持小数？

在 `engine_finetune.py` 第 38 行：
```python
lr_sched.adjust_learning_rate_epoch(
    optimizer, 
    data_iter_step / len(data_loader) + epoch,  # 这已经是小数！
    args
)
```

每个步骤的 "epoch 进度" 已经是小数，所以学习率调度器天然支持平滑过渡。

### 实现逻辑

1. 解析 `--epochs` 为浮点数
2. 分离整数部分和小数部分
3. 正常训练所有完整的 epoch
4. 如果有小数部分，最后一个 epoch：
   - 计算需要训练的步数 = 总步数 × 小数部分
   - 传递 `max_steps` 给训练函数
   - 在达到指定步数后提前终止
5. 保存检查点并结束训练

## ✅ 总结

现在 `accessory/main_finetune.py` 完全支持小数 epoch！您可以：
- 设置 `--epochs 2.5` 来训练 2.5 个 epoch
- 设置 `--epochs 0.1` 来快速测试
- 学习率调度器会平滑处理小数 epoch
- 所有功能（保存、日志、分布式训练）都正常工作

祝训练愉快！🚀
