# 微调期间的Prompt测试功能

这个功能允许您在模型微调过程中定期运行一些测试prompts，并打印生成的结果，以便监控模型在训练过程中的表现变化。

## 新增的命令行参数

在 `main_finetune.py` 中新增了以下参数：

- `--test_prompts`: 要测试的prompt列表（可以指定多个）
- `--test_prompt_interval`: 每隔多少步运行一次prompt测试（默认：500）
- `--test_prompt_max_gen_len`: 测试prompt的最大生成长度（默认：64）
- `--test_prompt_temperature`: 生成时的temperature参数（默认：0.1）
- `--test_prompt_top_p`: 生成时的top_p参数（默认：0.9）

## 使用方法

### 基本用法

```bash
python accessory/main_finetune.py \
    --llama_type llama2_7B \
    --llama_config /path/to/config.json \
    --tokenizer_path /path/to/tokenizer.model \
    --pretrained_path /path/to/pretrained/model \
    --data_config /path/to/data_config.yaml \
    --test_prompts "What is AI?" "Explain machine learning" "Write a poem" \
    --test_prompt_interval 100 \
    --test_prompt_max_gen_len 64 \
    --test_prompt_temperature 0.1 \
    --test_prompt_top_p 0.9 \
    --output_dir ./output_dir
```

### 详细示例

```bash
python accessory/main_finetune.py \
    # 模型配置
    --llama_type llama2_7B \
    --llama_config /path/to/your/config.json \
    --tokenizer_path /path/to/your/tokenizer.model \
    --pretrained_path /path/to/your/pretrained/model \
    
    # 训练参数
    --batch_size 4 \
    --accum_iter 4 \
    --epochs 3 \
    --lr 2e-5 \
    --weight_decay 0.02 \
    --warmup_epochs 0.1 \
    
    # 数据配置
    --data_config /path/to/your/data_config.yaml \
    --max_words 1024 \
    
    # 输出配置
    --output_dir ./output_finetune_with_prompts \
    --save_interval 1 \
    --save_iteration_interval 1000 \
    
    # Prompt测试配置（新功能）
    --test_prompts "What is the capital of France?" "Explain machine learning" "Write a haiku" \
    --test_prompt_interval 100 \
    --test_prompt_max_gen_len 64 \
    --test_prompt_temperature 0.1 \
    --test_prompt_top_p 0.9 \
    
    # 分布式训练
    --precision bf16 \
    --data_parallel fsdp
```

## 输出格式

当达到指定的测试间隔时，控制台会输出类似以下的格式：

```
============================================================
PROMPT TEST - Epoch: 0, Step: 100
============================================================

Prompt 1: What is the capital of France?
Response: The capital of France is Paris, which is located in the north-central part of the country.
----------------------------------------

Prompt 2: Explain machine learning
Response: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.
----------------------------------------

Prompt 3: Write a haiku
Response: Cherry blossoms fall
Gentle breeze carries petals
Spring's fleeting beauty
----------------------------------------
============================================================
```

## 实现原理

1. **参数解析**: 在 `main_finetune.py` 中添加了新的命令行参数来配置prompt测试
2. **测试函数**: 在 `engine_finetune.py` 中添加了 `test_prompts_during_training()` 函数
3. **训练集成**: 在训练循环中，每当达到指定步数间隔时，会调用测试函数
4. **生成逻辑**: 使用与您提供的示例相同的生成代码：
   ```python
   with torch.cuda.amp.autocast(dtype=torch.bfloat16):
       results = model.generate(
           prompts, 
           None,  # image
           max_gen_len=64, 
           temperature=0.1, 
           top_p=0.9
       )
   ```

## 注意事项

1. **性能影响**: prompt测试会暂时暂停训练，建议设置合理的测试间隔
2. **内存使用**: 生成过程会使用额外的GPU内存
3. **分布式训练**: 只在主进程上运行测试，避免重复输出
4. **错误处理**: 如果生成过程出错，会打印错误信息但不会中断训练
5. **模型状态**: 测试前会将模型设置为eval模式，测试后恢复为train模式

## 自定义扩展

您可以根据需要修改 `test_prompts_during_training()` 函数来：

- 添加更复杂的prompt格式
- 保存生成结果到文件
- 添加评估指标
- 支持图像输入的多模态测试

## 示例脚本

参考 `example_finetune_with_prompt_testing.py` 获取完整的使用示例。