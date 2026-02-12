# 跨GPU数量恢复训练指南

## 问题回答

**可以从4张卡训练的checkpoint在2张卡上继续训练吗？**

**答案：可以！** 您的代码库完全支持这种操作。

## 技术原理

### 1. Tensor Parallel 兼容性

您的框架使用了智能的tensor parallel转换机制：

```python
# 支持的转换类型
if ckpt_mp_world_size % mp_world_size == 0:
    # 4→2, 8→4, 8→2 等 (合并ranks)
    local_state_dict = _load_checkpoint_and_merge_ranks(...)
elif mp_world_size % ckpt_mp_world_size == 0:  
    # 2→4, 2→8, 4→8 等 (分割ranks)
    local_state_dict = _load_checkpoint_and_split_rank(...)
```

### 2. 自动权重重分布

- **模型权重**: 4个分片自动合并为2个分片
- **优化器状态**: 自动重新分片以适应新的并行配置
- **训练状态**: epoch、iteration等状态完整保留

## 实操步骤

### 步骤1: 验证Checkpoint

```bash
# 使用提供的脚本验证兼容性
python resume_training_different_gpus.py \
    --original_checkpoint /path/to/4gpu/checkpoint \
    --new_model_parallel_size 2 \
    --output_dir /path/to/new/output \
    --dry_run
```

### 步骤2: 调整训练参数

关键参数调整：

| 参数 | 4卡设置 | 2卡设置 | 说明 |
|------|---------|---------|------|
| `--model_parallel_size` | 4 | 2 | MP大小 |
| `--batch_size` | 16 | 32 | 保持effective batch size |
| `nproc_per_node` | 4 | 2 | 进程数 |

### 步骤3: 执行恢复训练

```bash
# 示例命令
torchrun --nproc_per_node=2 accessory/main_finetune.py \
    --model_parallel_size 2 \
    --batch_size 32 \
    --resume /path/to/4gpu/checkpoint \
    --output_dir /path/to/new/output \
    --data_config /path/to/data_config.yaml \
    --llama_config /path/to/llama_config.json \
    --tokenizer_path /path/to/tokenizer.model \
    --epochs 10 \
    --lr 1e-4 \
    --precision bf16 \
    --data_parallel fsdp
```

## 支持的转换场景

### ✅ 完全支持的转换

| 原始GPU | 目标GPU | 转换类型 | 状态 |
|---------|---------|----------|------|
| 8 | 4 | 合并 | ✅ 支持 |
| 8 | 2 | 合并 | ✅ 支持 |
| 8 | 1 | 合并 | ✅ 支持 |
| 4 | 2 | 合并 | ✅ 支持 |
| 4 | 1 | 合并 | ✅ 支持 |
| 2 | 1 | 合并 | ✅ 支持 |
| 1 | 2 | 分割 | ✅ 支持 |
| 1 | 4 | 分割 | ✅ 支持 |
| 2 | 4 | 分割 | ✅ 支持 |
| 2 | 8 | 分割 | ✅ 支持 |

### ❌ 需要间接转换

| 原始GPU | 目标GPU | 建议路径 |
|---------|---------|----------|
| 3 | 2 | 3→1→2 |
| 5 | 4 | 5→1→4 |
| 6 | 4 | 6→2→4 |

## 重要注意事项

### 1. 内存需求

GPU数量减少时，每卡内存需求会增加：
- 4卡→2卡: 每卡内存需求约2倍
- 4卡→1卡: 每卡内存需求约4倍

### 2. 性能影响

- **训练速度**: 与GPU数量成正比下降
- **通信开销**: GPU数量少时相对通信开销可能增加

### 3. 数值一致性

- 梯度同步模式可能略有变化
- 建议在切换后监控loss曲线
- 必要时可以调整学习率

## 故障排除

### 常见问题1: 内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案:**
- 减少 `batch_size`
- 启用 `--checkpointing`
- 使用 `--precision fp16`

### 常见问题2: 找不到checkpoint文件
```
FileNotFoundError: consolidated.xx-of-xx.model.pth
```
**解决方案:**
- 检查checkpoint目录结构
- 确保所有必需文件都存在
- 验证文件权限

### 常见问题3: 模型配置不匹配
```
RuntimeError: Error(s) in loading state_dict
```
**解决方案:**
- 确保模型配置文件正确
- 检查 `llama_config` 路径
- 验证模型架构一致性

## 最佳实践

### 1. 渐进式验证
```bash
# 1. 先验证兼容性
python resume_training_different_gpus.py --dry_run ...

# 2. 小批量测试
# 使用少量数据验证恢复是否正常

# 3. 完整恢复训练
# 确认无误后进行完整训练
```

### 2. 监控指标
- GPU内存使用率
- 训练速度变化
- Loss曲线连续性
- 梯度范数稳定性

### 3. 备份策略
- 保留原始checkpoint
- 定期保存新的checkpoint
- 记录转换过程日志

## 总结

您的问题的答案是：**完全可以！** 从4卡checkpoint恢复到2卡训练是被完全支持的操作。关键是：

1. ✅ **技术可行**: 代码原生支持tensor parallel size转换
2. ✅ **自动处理**: 权重和优化器状态自动重分布  
3. ✅ **状态保持**: 训练进度完整保留
4. ⚠️ **注意内存**: 确保2卡有足够内存
5. ⚠️ **调整参数**: 适当调整batch size等参数

使用提供的脚本和指南，您可以安全地进行这种转换！