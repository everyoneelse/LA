# 999M模型评估快速参考

## 🎯 一分钟快速开始

```bash
# 1. 设置环境变量
export MODEL_PATH=/path/to/your/999m/model
export LLAMA_CONFIG=/path/to/params.json
export TOKENIZER_PATH=/path/to/tokenizer.model

# 2. 运行快速评估
bash quick_eval_999m.sh
```

## 📊 评估类型速查

| 评估类型 | 时间 | 命令 | 作用 |
|---------|------|------|------|
| 基础评估 | 10-30分钟 | `--eval_types basic` | 测试各种基础能力 |
| 困惑度 | 5分钟 | `--eval_types perplexity` | 衡量语言建模能力 |
| 质量评估 | 15分钟 | `--eval_types quality` | 评估响应质量 |
| 速度评估 | 10分钟 | `--eval_types speed` | 测试推理速度 |
| GSM8K | 2-4小时 | `bash run_benchmark_suite.sh` 选1 | 数学推理 |
| MMLU | 3-5小时 | `bash run_benchmark_suite.sh` 选2 | 知识理解 |

## 🔍 结果解读速查

### 困惑度 (Perplexity)
```
< 15    ⭐⭐⭐ 优秀
15-25   ⭐⭐  良好
25-40   ⭐    可接受
> 40    ❌    需改进
```

### GSM8K准确率
```
> 50%   ⭐⭐⭐ 优秀
30-50%  ⭐⭐  良好
< 30%   ❌    需改进
```

### 推理速度
```
> 50 tok/s   ⚡⚡⚡ 快
20-50 tok/s  ⚡⚡  中等
< 20 tok/s   ❌   慢
```

## 🎯 常见优化方向

### 问题：困惑度高
```bash
# 解决方案1: 继续训练
- 增加训练步数
- 使用更多数据

# 解决方案2: 调整学习率
- 降低学习率
- 使用余弦衰减
```

### 问题：响应质量差
```bash
# 解决方案: 指令微调
cd accessory
bash exps/finetune/sg/alpaca.sh  # 使用Alpaca数据
```

### 问题：数学能力弱
```bash
# 解决方案: 专项训练
- 收集更多数学数据
- 使用思维链提示
```

### 问题：推理速度慢
```bash
# 解决方案: 量化
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --quant  # 启用4bit量化
```

## 📁 输出文件说明

```
evaluation_results/
├── eval_YYYYMMDD_HHMMSS/
│   ├── evaluation_report.md      # 📝 主报告(推荐先看这个)
│   ├── evaluation_report.json    # 🔧 完整数据
│   ├── basic_capabilities.json   # 📊 基础能力详情
│   ├── perplexity.json           # 📈 困惑度详情
│   ├── response_quality.json     # ⭐ 质量评估详情
│   ├── inference_speed.json      # ⚡ 速度评估详情
│   └── optimization_suggestions.json  # 💡 优化建议
```

## 🚀 完整命令示例

### 示例1: 完整评估
```bash
python comprehensive_model_evaluation.py \
  --pretrained_path /path/to/model \
  --llama_config /path/to/params.json \
  --tokenizer_path /path/to/tokenizer.model \
  --eval_types basic perplexity quality speed \
  --output_dir ./my_eval_results \
  --dtype bf16
```

### 示例2: 仅测试困惑度（快速）
```bash
python comprehensive_model_evaluation.py \
  --pretrained_path /path/to/model \
  --llama_config /path/to/params.json \
  --tokenizer_path /path/to/tokenizer.model \
  --eval_types perplexity
```

### 示例3: 量化推理评估
```bash
python comprehensive_model_evaluation.py \
  --pretrained_path /path/to/model \
  --llama_config /path/to/params.json \
  --tokenizer_path /path/to/tokenizer.model \
  --quant \
  --eval_types speed
```

## 🔧 故障排查

| 问题 | 解决方案 |
|------|---------|
| CUDA内存不足 | 添加 `--quant` 启用量化 |
| 找不到模型文件 | 检查路径是否正确 |
| 评估速度太慢 | 先运行单项测试，如 `--eval_types perplexity` |
| 结果不理想 | 查看 `optimization_suggestions.json` |

## 📞 获取详细帮助

```bash
# 查看完整指南
cat MODEL_EVALUATION_GUIDE.md

# 查看配置说明
cat eval_config_999m.json

# 查看脚本帮助
python comprehensive_model_evaluation.py --help
```

## ⏱️ 时间规划建议

### 第一次评估 (推荐)
1. ✅ 快速评估 (30分钟)
   ```bash
   bash quick_eval_999m.sh
   ```

2. ✅ GSM8K测试 (3小时)
   ```bash
   bash run_benchmark_suite.sh  # 选择1
   ```

3. ✅ 分析结果，制定计划 (1小时)

### 后续评估
- 每次改进后运行快速评估
- 重大变更后运行完整benchmark
- 定期（每周）跟踪进展

## 🎯 下一步行动建议

1. **立即执行**: 运行快速评估了解基线
2. **今天完成**: 运行GSM8K获取数学能力指标
3. **本周完成**: 根据结果制定优化计划
4. **持续进行**: 记录每次改进的效果

---

**记住**: 评估不是目的，优化才是！🚀
