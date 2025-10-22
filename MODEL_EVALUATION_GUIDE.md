# 999M模型评估与优化指南

## 📋 目录

1. [快速开始](#快速开始)
2. [评估方法](#评估方法)
3. [性能基准](#性能基准)
4. [优化方向](#优化方向)
5. [常见问题](#常见问题)

---

## 🚀 快速开始

### 1. 准备工作

确保你已经训练好了999M模型，并且有以下文件：
- 模型检查点文件
- 模型配置文件 (`params.json` 或类似文件)
- Tokenizer文件 (`tokenizer.model`)

### 2. 快速评估

使用快速评估脚本：

```bash
# 设置环境变量
export MODEL_PATH=/path/to/your/999m/model
export LLAMA_CONFIG=/path/to/params.json
export TOKENIZER_PATH=/path/to/tokenizer.model

# 运行快速评估
bash quick_eval_999m.sh
```

### 3. 详细评估

运行完整的评估套件：

```bash
python comprehensive_model_evaluation.py \
  --pretrained_path /path/to/model \
  --llama_config /path/to/params.json \
  --tokenizer_path /path/to/tokenizer.model \
  --eval_types basic perplexity quality speed \
  --output_dir ./evaluation_results \
  --dtype bf16
```

---

## 📊 评估方法

### 1. 基础能力评估

**目的**: 测试模型在各种任务上的基本表现

**测试类别**:
- 📝 **基础理解**: 概念解释、知识问答
- 🧮 **数学推理**: 算术计算、方程求解
- 💻 **代码生成**: 编程任务、算法实现
- 🤔 **逻辑推理**: 因果推理、规律发现
- 📚 **知识问答**: 领域知识、常识问答
- ✍️ **创意写作**: 文本创作、风格模仿

**如何运行**:
```bash
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --eval_types basic
```

**评估指标**:
- 响应相关性
- 答案完整性
- 生成质量

### 2. 困惑度评估 (Perplexity)

**目的**: 衡量模型对文本的预测能力

**什么是困惑度**:
- 困惑度越低，模型对语言的建模越好
- 困惑度 = exp(平均负对数似然)

**性能基准**:
- 优秀: < 15
- 良好: 15-25
- 可接受: 25-40
- 需改进: > 40

**如何运行**:
```bash
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --eval_types perplexity
```

### 3. 标准Benchmark测试

#### GSM8K (数学推理)

**描述**: 小学数学应用题，测试模型的数学推理能力

**运行方式**:
```bash
cd light-eval

# 修改 scripts/run_gsm8k.sh 中的配置
# pretrained_path=/path/to/your/model
# llama_config=/path/to/your/config
# tokenizer_path=/path/to/your/tokenizer

bash scripts/run_gsm8k.sh
```

**结果解读**:
- 准确率 > 50%: 优秀
- 准确率 30-50%: 良好
- 准确率 < 30%: 需要改进

#### MMLU (多任务语言理解)

**描述**: 覆盖57个学科的多选题，测试模型的知识广度

**运行方式**:
```bash
cd light-eval
bash scripts/run_mmlu.sh
```

**结果解读**:
- 准确率 > 60%: 优秀
- 准确率 40-60%: 良好
- 准确率 < 40%: 需要改进

#### HumanEval (代码生成)

**描述**: 编程任务，测试代码生成能力

**运行方式**:
```bash
cd light-eval
bash scripts/run_humaneval.sh
```

### 4. 推理速度评估

**目的**: 评估模型的推理效率

**关键指标**:
- **tokens/秒**: 每秒生成的token数量
- **首token延迟**: 生成第一个token的时间
- **总生成时间**: 完成整个响应的时间

**性能基准**:
- 快速: > 50 tokens/s
- 中等: 20-50 tokens/s  
- 较慢: < 20 tokens/s

---

## 🎯 性能基准参考

### 999M模型预期性能

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| 困惑度 | < 15 | 15-25 | > 25 |
| GSM8K准确率 | > 50% | 30-50% | < 30% |
| MMLU准确率 | > 60% | 40-60% | < 40% |
| 推理速度 | > 50 tok/s | 20-50 tok/s | < 20 tok/s |
| 质量分数 | > 0.8 | 0.6-0.8 | < 0.6 |

---

## 🔧 优化方向

### 根据评估结果确定优化重点

#### 1. 如果困惑度较高 (> 30)

**可能原因**:
- 训练数据不足或质量不高
- 训练未充分收敛
- 学习率设置不当

**优化建议**:
```python
# 选项1: 增加训练数据
- 收集更多高质量文本数据
- 确保数据多样性和代表性

# 选项2: 延长训练时间
- 增加训练步数
- 使用更多的训练epoch

# 选项3: 调整学习率
- 尝试余弦衰减学习率调度
- 适当降低学习率峰值
- 增加warmup步数

# 选项4: 优化器调整
- 尝试不同的beta参数
- 调整权重衰减
```

**具体代码示例**:
```python
# 在训练脚本中调整
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,  # 降低学习率
    betas=(0.9, 0.95),  # 调整beta
    weight_decay=0.1  # 增加权重衰减
)

# 使用余弦学习率调度
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)
```

#### 2. 如果响应质量不高 (质量分数 < 0.6)

**可能原因**:
- 预训练数据质量问题
- 缺少指令微调
- 输出格式不规范

**优化建议**:
```python
# 选项1: 数据清洗和过滤
- 移除低质量文本
- 去重复数据
- 平衡不同领域的数据

# 选项2: 指令微调 (Instruction Tuning)
- 使用高质量的指令数据集
- 如: Alpaca, ShareGPT, WizardLM等

# 选项3: 对齐训练
- RLHF (人类反馈强化学习)
- DPO (直接偏好优化)
```

**运行指令微调**:
```bash
cd accessory

# 使用Alpaca数据集微调
bash exps/finetune/sg/alpaca.sh

# 或使用ShareGPT数据集
bash exps/finetune/sg/dialog_sharegpt.sh
```

#### 3. 如果数学推理能力弱 (GSM8K < 30%)

**优化建议**:
```python
# 选项1: 增加数学相关数据
- 收集更多数学推理数据
- 包含详细的解题步骤

# 选项2: 使用思维链 (Chain-of-Thought)
- 训练数据中包含推理过程
- 使用 "让我们一步步思考" 的提示

# 选项3: 专项微调
- 在数学数据集上进行微调
- 如: GSM8K, MATH等
```

#### 4. 如果推理速度慢 (< 20 tokens/s)

**优化建议**:
```python
# 选项1: 模型量化
# INT8量化
model = model.to(torch.int8)

# INT4量化 (使用bitsandbytes)
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 选项2: Flash Attention
# 在模型配置中启用

# 选项3: KV-cache优化
# 减少内存占用，提高推理速度

# 选项4: 批处理推理
# 同时处理多个请求
```

**使用量化推理**:
```bash
python comprehensive_model_evaluation.py \
  --pretrained_path $MODEL_PATH \
  --quant \
  --dtype bf16
```

#### 5. 如果代码生成能力弱 (HumanEval低)

**优化建议**:
```python
# 选项1: 增加代码训练数据
- The Stack
- CodeParrot
- GitHub代码库

# 选项2: 使用代码专用数据集微调
- WizardCoder
- Code Alpaca

# 选项3: 多语言代码训练
- 包含Python, JavaScript, Java等
```

---

## 📈 优化流程建议

### 迭代优化流程

```
1. 全面评估
   ↓
2. 识别弱项
   ↓
3. 制定优化计划
   ↓
4. 实施改进
   ↓
5. 重新评估
   ↓
6. 对比改进效果
   ↓
7. 继续迭代
```

### 优先级建议

**高优先级**:
1. 降低困惑度 (基础能力提升)
2. 指令对齐 (提升可用性)
3. 数学推理能力 (重要benchmark)

**中优先级**:
1. 推理速度优化
2. 代码生成能力
3. 多语言支持

**低优先级**:
1. 创意写作
2. 特定领域知识

---

## 🛠️ 工具和资源

### 评估工具

1. **本项目提供的工具**:
   - `comprehensive_model_evaluation.py` - 综合评估
   - `text_completion_test.py` - 文本补全测试
   - `light-eval/` - 标准benchmark评估

2. **外部工具**:
   - [OpenCompass](https://github.com/open-compass/opencompass) - 全面的评估平台
   - [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Eleuther AI的评估工具

### 训练数据集

1. **预训练数据**:
   - [The Pile](https://pile.eleuther.ai/)
   - [RedPajama](https://github.com/togethercomputer/RedPajama-Data)
   - [C4](https://www.tensorflow.org/datasets/catalog/c4)

2. **指令微调数据**:
   - [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
   - [ShareGPT](https://sharegpt.com/)
   - [WizardLM](https://github.com/nlpxucan/WizardLM)

3. **专项数据**:
   - 数学: GSM8K, MATH
   - 代码: The Stack, WizardCoder
   - 中文: CLUE, C-Eval

---

## ❓ 常见问题

### Q1: 评估需要多长时间？

**A**: 取决于评估类型：
- 基础评估: 10-30分钟
- 完整评估: 1-2小时
- Benchmark测试: 每个2-4小时

### Q2: 评估需要什么硬件？

**A**: 
- 最低: 1个GPU (16GB显存)
- 推荐: 1个GPU (24GB或更大)
- 可以使用量化减少显存需求

### Q3: 如何解读评估结果？

**A**: 
1. 先看困惑度 - 反映基础建模能力
2. 再看具体任务表现 - 识别强项弱项
3. 对比同类模型 - 了解相对水平
4. 关注趋势变化 - 优化是否有效

### Q4: 第一次评估应该关注什么？

**A**: 
1. 困惑度（最基础的指标）
2. GSM8K（数学推理）
3. 基础文本生成质量
4. 推理速度

### Q5: 多久应该重新评估一次？

**A**: 
- 每次重要改进后都应评估
- 定期评估（如每周）跟踪进展
- 重大版本发布前必须全面评估

### Q6: 评估结果不理想怎么办？

**A**: 
1. 不要灰心，这是正常的
2. 仔细分析具体哪里不足
3. 一次只改进一个方面
4. 做对比实验验证改进
5. 保持记录，追踪进展

---

## 📝 评估结果记录模板

建议为每次评估创建记录：

```markdown
# 评估记录 - YYYY-MM-DD

## 模型信息
- 模型版本: v1.0
- 训练步数: 100k
- 训练数据: XXX

## 评估结果
- 困惑度: XX.X
- GSM8K: XX%
- MMLU: XX%
- 推理速度: XX tok/s

## 观察和发现
- [观察1]
- [观察2]

## 下一步计划
- [计划1]
- [计划2]
```

---

## 🎓 学习资源

### 推荐阅读

1. **论文**:
   - [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
   - [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla)
   - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

2. **博客**:
   - [How to evaluate LLMs](https://www.promptingguide.ai/)
   - [LLM Optimization Guide](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization)

---

## 🤝 获取帮助

如果遇到问题：

1. 检查错误日志
2. 查看本指南的常见问题部分
3. 查看项目文档
4. 提交Issue并附上详细信息

---

## 📅 评估清单

使用这个清单确保完整评估：

- [ ] 运行基础能力评估
- [ ] 计算困惑度
- [ ] 测试GSM8K
- [ ] 测试MMLU  
- [ ] 评估推理速度
- [ ] 生成评估报告
- [ ] 识别主要弱项
- [ ] 制定优化计划
- [ ] 记录评估结果
- [ ] 设定改进目标

---

**祝你的模型评估和优化顺利！** 🚀
