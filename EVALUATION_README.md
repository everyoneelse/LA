# 999M模型评估工具包

> 完整的模型评估、测试和优化方向分析工具

## 📦 工具包内容

本工具包为你的999M模型提供了全面的评估方案，包括：

### 🔧 核心工具

| 文件 | 用途 | 使用难度 |
|------|------|---------|
| `quick_eval_999m.sh` | ⚡ 快速评估脚本 | ⭐ 简单 |
| `comprehensive_model_evaluation.py` | 📊 综合评估工具 | ⭐⭐ 中等 |
| `run_benchmark_suite.sh` | 📚 标准测试套件 | ⭐⭐ 中等 |
| `example_evaluation_workflow.sh` | 🎯 完整评估流程 | ⭐ 简单 |
| `text_completion_test.py` | 💬 交互式测试 | ⭐ 简单 |

### 📚 文档资源

| 文件 | 内容 |
|------|------|
| `MODEL_EVALUATION_GUIDE.md` | 📖 详细评估指南（必读！） |
| `QUICK_REFERENCE.md` | 🎯 快速参考卡片 |
| `eval_config_999m.json` | ⚙️ 评估配置说明 |
| `EVALUATION_README.md` | 📋 本文件 |

## 🚀 快速开始（5分钟上手）

### 第一步：设置环境变量

```bash
# 根据你的实际路径修改
export MODEL_PATH=/path/to/your/999m/model
export LLAMA_CONFIG=/path/to/params.json
export TOKENIZER_PATH=/path/to/tokenizer.model
```

### 第二步：运行快速评估

```bash
# 最简单的方式
bash quick_eval_999m.sh
```

就这么简单！脚本会自动运行评估并生成详细报告。

## 📊 评估内容

### 1️⃣ 基础能力评估

测试模型在以下方面的表现：
- ✍️ 文本理解和生成
- 🧮 数学计算能力
- 💻 代码生成能力
- 🤔 逻辑推理能力
- 📚 知识问答能力
- 🎨 创意写作能力

**运行方式**：
```bash
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --llama_config $LLAMA_CONFIG \
  --tokenizer_path $TOKENIZER_PATH \
  --eval_types basic
```

### 2️⃣ 困惑度评估

衡量模型的语言建模能力（核心指标）

**性能标准**：
- 🟢 优秀: < 15
- 🟡 良好: 15-25
- 🟠 可接受: 25-40
- 🔴 需改进: > 40

**运行方式**：
```bash
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --llama_config $LLAMA_CONFIG \
  --tokenizer_path $TOKENIZER_PATH \
  --eval_types perplexity
```

### 3️⃣ 标准Benchmark测试

在业界标准数据集上测试：

| Benchmark | 测试内容 | 所需时间 |
|-----------|---------|---------|
| **GSM8K** | 数学推理 | 2-4小时 |
| **MMLU** | 知识理解 | 3-5小时 |
| **C-Eval** | 中文能力 | 2-4小时 |
| **HumanEval** | 代码生成 | 1-2小时 |

**运行方式**：
```bash
bash run_benchmark_suite.sh
# 然后选择要运行的测试
```

### 4️⃣ 推理性能评估

测试模型的推理速度和效率

**性能标准**：
- ⚡⚡⚡ 快速: > 50 tokens/秒
- ⚡⚡ 中等: 20-50 tokens/秒
- ⚡ 较慢: < 20 tokens/秒

**运行方式**：
```bash
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --llama_config $LLAMA_CONFIG \
  --tokenizer_path $TOKENIZER_PATH \
  --eval_types speed
```

## 🎯 完整评估流程

如果你想运行一个完整的、循序渐进的评估流程：

```bash
bash example_evaluation_workflow.sh
```

这个脚本会引导你完成：
1. ✅ 环境检查
2. ✅ 快速诊断
3. ✅ 困惑度评估
4. ✅ 基础能力测试
5. ✅ 性能优化测试（可选）
6. ✅ Benchmark测试（可选）
7. ✅ 生成综合报告

## 📈 评估结果

所有评估结果会保存在 `evaluation_results/` 目录下：

```
evaluation_results/
└── eval_YYYYMMDD_HHMMSS/
    ├── evaluation_report.md           # 📝 主报告（推荐先看这个）
    ├── evaluation_report.json         # 🔧 完整数据
    ├── basic_capabilities.json        # 📊 基础能力详情
    ├── perplexity.json               # 📈 困惑度详情
    ├── response_quality.json         # ⭐ 质量评估详情
    ├── inference_speed.json          # ⚡ 速度评估详情
    └── optimization_suggestions.json # 💡 优化建议
```

### 查看报告

```bash
# 查看最新的markdown报告
ls -t evaluation_results/eval_*/evaluation_report.md | head -1 | xargs cat

# 查看优化建议
ls -t evaluation_results/eval_*/optimization_suggestions.json | head -1 | xargs cat
```

## 💡 优化方向

评估完成后，工具会自动生成优化建议。常见的优化方向包括：

### 🔴 高优先级

1. **降低困惑度**
   - 增加训练数据
   - 优化学习率调度
   - 延长训练时间

2. **提升响应质量**
   - 进行指令微调
   - 使用高质量数据集
   - 考虑RLHF/DPO对齐

3. **增强数学能力**
   - 增加数学推理数据
   - 使用思维链训练

### 🟡 中优先级

1. **优化推理速度**
   - 模型量化（INT8/INT4）
   - Flash Attention
   - KV-cache优化

2. **提升代码能力**
   - 增加代码训练数据
   - 使用WizardCoder等数据集

### 详细优化指南

查看完整的优化建议和实施方案：
```bash
cat MODEL_EVALUATION_GUIDE.md
```

## 🔍 交互式测试

如果你想快速测试模型的文本生成能力：

```bash
python text_completion_test.py \
  --pretrained_path $MODEL_PATH \
  --llama_config $LLAMA_CONFIG \
  --tokenizer_path $TOKENIZER_PATH
```

然后选择"交互式测试"，你可以：
- 输入任何提示词测试模型
- 实时调整生成参数
- 快速验证模型能力

## 🎓 学习路径

### 新手上路

1. 阅读 `QUICK_REFERENCE.md` （5分钟）
2. 运行 `bash quick_eval_999m.sh` （30分钟）
3. 查看生成的报告
4. 根据建议制定优化计划

### 深入评估

1. 阅读 `MODEL_EVALUATION_GUIDE.md` （30分钟）
2. 运行 `bash example_evaluation_workflow.sh` （1-2小时）
3. 运行标准benchmark测试 （2-8小时）
4. 分析详细结果
5. 实施优化措施

### 持续优化

1. 每次改进后运行快速评估
2. 定期（每周）运行完整评估
3. 记录每次改进的效果
4. 迭代优化

## 📋 评估清单

使用这个清单确保完整评估：

- [ ] 运行快速评估获取基线
- [ ] 测试困惑度
- [ ] 评估基础能力
- [ ] 测试推理速度
- [ ] 运行GSM8K（数学）
- [ ] 运行MMLU（知识）
- [ ] 查看优化建议
- [ ] 制定改进计划
- [ ] 记录评估结果
- [ ] 设定下一步目标

## 🛠️ 高级选项

### 使用量化加速评估

```bash
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --llama_config $LLAMA_CONFIG \
  --tokenizer_path $TOKENIZER_PATH \
  --quant  # 启用4bit量化
```

### 自定义评估类型

```bash
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --llama_config $LLAMA_CONFIG \
  --tokenizer_path $TOKENIZER_PATH \
  --eval_types perplexity quality  # 只运行特定测试
```

### 调整生成参数

```bash
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --llama_config $LLAMA_CONFIG \
  --tokenizer_path $TOKENIZER_PATH \
  --temperature 0.7 \
  --top_p 0.9 \
  --max_gen_len 512
```

## ❓ 常见问题

### Q: 评估需要多长时间？

**A**: 
- 快速评估: 30分钟 - 1小时
- 完整评估: 2-3小时
- 包含benchmark: 6-12小时

### Q: 需要什么硬件？

**A**:
- 最低: 1个GPU（16GB显存）
- 推荐: 1个GPU（24GB或更大）
- 可以使用 `--quant` 减少显存需求

### Q: 第一次评估应该做什么？

**A**:
1. 运行快速评估
2. 查看困惑度和基础能力
3. 运行GSM8K测试
4. 根据结果制定计划

### Q: 结果不理想怎么办？

**A**:
1. 查看 `optimization_suggestions.json`
2. 阅读 `MODEL_EVALUATION_GUIDE.md` 中的优化部分
3. 一次改进一个方面
4. 重新评估验证效果

### Q: 如何对比不同版本？

**A**:
每次评估的结果都有时间戳，可以保存并对比：
```bash
# 保存结果
cp -r evaluation_results/eval_20250101_120000 results_v1.0
cp -r evaluation_results/eval_20250108_150000 results_v1.1

# 对比关键指标
diff results_v1.0/perplexity.json results_v1.1/perplexity.json
```

## 🔗 相关资源

### 项目内文档
- 📖 详细评估指南: `MODEL_EVALUATION_GUIDE.md`
- 🎯 快速参考: `QUICK_REFERENCE.md`
- ⚙️ 配置说明: `eval_config_999m.json`

### 外部资源
- [OpenCompass](https://github.com/open-compass/opencompass) - 评估平台
- [LLM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Papers with Code - LLM Benchmarks](https://paperswithcode.com/)

## 📞 获取帮助

1. 查看文档中的FAQ部分
2. 检查错误日志
3. 参考示例脚本
4. 提交Issue（附上详细信息）

## 🎉 开始评估

现在你已经了解了所有评估工具，开始评估你的999M模型吧！

```bash
# 最简单的开始方式
export MODEL_PATH=/your/model/path
export LLAMA_CONFIG=/your/config/path
export TOKENIZER_PATH=/your/tokenizer/path

bash quick_eval_99m.sh
```

**祝评估顺利！** 🚀

---

## 📝 更新日志

- **2025-10-22**: 创建初始版本
  - ✅ 综合评估脚本
  - ✅ 快速评估工具
  - ✅ Benchmark测试套件
  - ✅ 详细文档和指南

---

**记住**: 评估是手段，优化才是目的！持续迭代，你的模型会越来越好！💪
