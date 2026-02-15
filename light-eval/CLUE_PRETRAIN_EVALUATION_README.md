# CLUE预训练模型评估指南

## 概述

本实现提供了专门针对**预训练模型**的CLUE评估工具，经过精心筛选只包含适合zero-shot和few-shot评估的任务。相比完整的CLUE benchmark，本实现更适合评估预训练模型的基础语言理解能力。

## 为什么需要筛选任务？

并非所有CLUE任务都适合直接评估预训练模型：

1. **不适合的任务**：
   - **IFLYTEK**（119类应用分类）：类别过多，需要fine-tuning
   - **TNEWS**（15类新闻分类）：需要特定领域知识
   - **CHID**（成语填空）：更像知识检索而非语言理解

2. **适合的任务**：
   - 基础语言理解任务（语义相似、逻辑推理）
   - 类别数量合理（2-3类）
   - 可通过上下文理解完成

## 精选任务列表

| 任务 | 描述 | 类型 | 难度 | Zero-shot | 推荐Shots |
|------|------|------|------|-----------|-----------|
| **CMNLI** | 中文自然语言推理 | 3分类 | 中等 | ✅ | 3 |
| **AFQMC** | 语义相似度判断 | 2分类 | 简单 | ✅ | 2 |
| **CSL** | 关键词识别 | 2分类 | 中等 | ✅ | 3 |
| **WSC** | 指代消解 | 2分类 | 困难 | ❌ | 5 |
| **OCNLI** | 原生中文推理 | 3分类 | 中等 | ✅ | 3 |

### 任务详细说明

#### 1. CMNLI - 中文自然语言推理（推荐）
- **评估能力**：逻辑推理
- **任务形式**：判断两个句子的逻辑关系（蕴含/中立/矛盾）
- **适合原因**：语言理解的核心能力，预训练模型应具备

#### 2. AFQMC - 语义相似度（推荐）
- **评估能力**：语义理解
- **任务形式**：判断两个句子是否语义相似
- **适合原因**：直接评估语义理解能力

#### 3. CSL - 关键词识别
- **评估能力**：文本理解和总结
- **任务形式**：判断关键词是否真实
- **适合原因**：考察对文本主题的理解

#### 4. WSC - 指代消解
- **评估能力**：上下文理解
- **任务形式**：判断代词的指代对象
- **适合原因**：基础语言能力，但较难

#### 5. OCNLI - 原生中文推理
- **评估能力**：自然语言推理
- **任务形式**：类似CMNLI但更自然的中文表达
- **适合原因**：更贴近真实中文使用场景

## 快速开始

### 1. 安装依赖

```bash
cd /workspace/light-eval
pip install requests tqdm pandas numpy
```

### 2. 下载数据

```bash
# 只下载推荐的任务
python src/clue/download_clue.py --data_dir data/clue --task cmnli
python src/clue/download_clue.py --data_dir data/clue --task afqmc
python src/clue/download_clue.py --data_dir data/clue --task csl
python src/clue/download_clue.py --data_dir data/clue --task wsc
python src/clue/download_clue.py --data_dir data/clue --task ocnli
```

### 3. 运行评估

#### 完整评估（推荐）

```bash
cd scripts
chmod +x run_clue_pretrain.sh

# 评估所有推荐任务，同时进行zero-shot和few-shot
./run_clue_pretrain.sh /path/to/model data/clue both
```

#### 快速测试

```bash
chmod +x run_clue_pretrain_quick.sh

# 快速测试两个任务，验证代码
./run_clue_pretrain_quick.sh /path/to/model data/clue
```

#### Python命令行

```bash
cd src

# Zero-shot评估
python eval_clue_pretrain.py \
    --pretrained_path /path/to/model \
    --llama_config /path/to/config.json \
    --tokenizer_path /path/to/tokenizer.model \
    --data_dir data/clue \
    --tasks recommended \
    --evaluation_mode zero-shot

# Few-shot评估（使用推荐的shot数）
python eval_clue_pretrain.py \
    --pretrained_path /path/to/model \
    --llama_config /path/to/config.json \
    --tokenizer_path /path/to/tokenizer.model \
    --data_dir data/clue \
    --tasks recommended \
    --evaluation_mode few-shot

# 同时进行zero-shot和few-shot
python eval_clue_pretrain.py \
    --pretrained_path /path/to/model \
    --llama_config /path/to/config.json \
    --tokenizer_path /path/to/tokenizer.model \
    --data_dir data/clue \
    --tasks recommended \
    --evaluation_mode both

# 评估特定难度的任务
python eval_clue_pretrain.py \
    --pretrained_path /path/to/model \
    --tasks easy \  # 或 medium, hard
    --evaluation_mode both

# 自定义shot数量
python eval_clue_pretrain.py \
    --pretrained_path /path/to/model \
    --tasks cmnli afqmc \
    --evaluation_mode few-shot \
    --num_shots 5
```

## 参数说明

### 任务选择参数
- `--tasks`：
  - `recommended`：推荐的任务集（默认）
  - `all`：所有适合预训练的任务
  - `easy`：简单任务（AFQMC）
  - `medium`：中等难度（CMNLI, CSL, OCNLI）
  - `hard`：困难任务（WSC）
  - 具体任务名：如 `cmnli afqmc`

### 评估模式
- `--evaluation_mode`：
  - `zero-shot`：零样本评估
  - `few-shot`：少样本评估
  - `both`：同时进行（推荐）

### Few-shot设置
- `--num_shots`：示例数量（默认使用每个任务的推荐值）
- `--seed`：随机种子，控制示例选择

## 输出格式

结果保存在 `results/{model_name}/clue_pretrain/evaluation_results_{mode}.json`：

```json
{
  "results": {
    "cmnli": {
      "zero_shot": {
        "accuracy": 0.4523,
        "num_samples": 1000,
        "mode": "zero-shot",
        "num_shots": 0
      },
      "few_shot": {
        "accuracy": 0.5234,
        "num_samples": 1000,
        "mode": "few-shot",
        "num_shots": 3
      }
    },
    "afqmc": {
      "zero_shot": {
        "accuracy": 0.6789,
        "num_samples": 1000
      },
      "few_shot": {
        "accuracy": 0.7123,
        "num_samples": 1000
      }
    }
  },
  "summary": {
    "task_scores": {
      "cmnli": {
        "zero_shot": 0.4523,
        "few_shot": 0.5234,
        "best": 0.5234
      }
    },
    "average_scores": {
      "zero_shot": 0.5656,
      "few_shot": 0.6178,
      "best": 0.6178
    }
  }
}
```

## 评估策略说明

### Zero-shot评估
- 直接使用任务描述和问题
- 不提供任何示例
- 评估模型的基础理解能力

### Few-shot评估
- 提供少量示例（2-5个）
- 示例从训练集中平衡选择
- 评估模型的学习和泛化能力

### 示例选择策略
1. 平衡各类别的示例
2. 随机打乱顺序
3. 可通过seed参数固定选择

## 性能优化建议

### 1. 内存不足
```bash
# 使用4bit量化
python eval_clue_pretrain.py \
    --pretrained_path /path/to/model \
    --quant \
    --max_seq_len 1024
```

### 2. 加速评估
```bash
# 减少评估样本
python eval_clue_pretrain.py \
    --pretrained_path /path/to/model \
    --max_eval_samples 100 \
    --tasks afqmc cmnli  # 只评估快速任务
```

### 3. 调试模式
```bash
# 极少样本快速验证
python eval_clue_pretrain.py \
    --pretrained_path /path/to/model \
    --max_eval_samples 5 \
    --tasks afqmc \
    --evaluation_mode zero-shot
```

## 结果解读

### 期望性能范围

基于模型规模的大致期望：

| 模型规模 | Zero-shot | Few-shot |
|---------|-----------|----------|
| 1-3B | 35-45% | 45-55% |
| 7B | 45-55% | 55-65% |
| 13B | 50-60% | 60-70% |
| 30B+ | 55-65% | 65-75% |

### 性能分析要点

1. **Zero-shot vs Few-shot差距**：
   - 差距大（>10%）：模型具有良好的上下文学习能力
   - 差距小（<5%）：可能需要更多训练或模型容量不足

2. **任务间差异**：
   - AFQMC通常最高（简单的二分类）
   - WSC通常最低（需要复杂推理）
   - CMNLI是很好的综合指标

3. **异常情况**：
   - Zero-shot接近随机（33%/50%）：提示词可能需要调整
   - Few-shot反而下降：可能存在示例选择问题

## 常见问题

### Q1: 为什么不包含所有CLUE任务？
A: 许多任务需要fine-tuning才能有效评估，不适合预训练模型的zero/few-shot评估。

### Q2: 如何选择评估模式？
A: 建议使用`both`模式，可以全面了解模型能力。Zero-shot反映基础能力，few-shot反映学习能力。

### Q3: Few-shot应该用几个示例？
A: 使用默认推荐值即可。一般2-5个，太多会占用过多上下文长度。

### Q4: 结果波动很大怎么办？
A: 
- 固定seed保证可重复性
- 增加评估样本数量
- 多次运行取平均

## 引用

使用本代码请引用：

```bibtex
@article{xu2020clue,
  title={CLUE: A Chinese Language Understanding Evaluation Benchmark},
  author={Xu, Liang and others},
  journal={arXiv preprint arXiv:2004.05986},
  year={2020}
}
```

## 更新日志

- v1.1.0: 针对预训练模型优化，筛选适合的任务
- v1.0.0: 初始版本，包含所有CLUE任务