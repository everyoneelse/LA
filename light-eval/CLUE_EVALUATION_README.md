# CLUE Benchmark 评估指南

## 概述

本实现提供了使用CLUE（Chinese Language Understanding Evaluation）基准评估预训练模型的完整工具链。CLUE是中文语言理解的权威评估基准，包含多个自然语言理解任务。

## CLUE任务列表

1. **AFQMC** - 蚂蚁金融语义相似度
   - 判断两个句子语义是否相似
   - 二分类任务

2. **TNEWS** - 今日头条中文新闻分类
   - 15个类别的短文本分类
   - 多分类任务

3. **IFLYTEK** - 科大讯飞长文本分类
   - 119个应用类别分类
   - 长文本多分类任务

4. **CMNLI** - 中文自然语言推理
   - 判断句子对的逻辑关系（蕴含/中立/矛盾）
   - 三分类任务

5. **WSC** - 中文指代消解
   - 判断代词的指代对象
   - 二分类任务

6. **CSL** - 中文科学文献关键词识别
   - 判断关键词是否为真实关键词
   - 二分类任务

## 快速开始

### 1. 安装依赖

```bash
cd /workspace/light-eval
pip install requests tqdm pandas numpy
```

### 2. 下载CLUE数据集

```bash
# 下载所有CLUE任务数据
python src/clue/download_clue.py --data_dir data/clue --task all

# 或下载特定任务
python src/clue/download_clue.py --data_dir data/clue --task afqmc
```

### 3. 运行评估

#### 方式一：使用Shell脚本

```bash
cd scripts
chmod +x run_clue.sh run_clue_quick.sh

# 完整评估
./run_clue.sh /path/to/model data/clue all

# 快速测试（验证代码）
./run_clue_quick.sh /path/to/model data/clue
```

#### 方式二：直接运行Python脚本

```bash
cd src

# 基本用法
python eval_clue.py \
    --pretrained_path /path/to/model \
    --llama_config /path/to/config.json \
    --tokenizer_path /path/to/tokenizer.model \
    --data_dir data/clue \
    --tasks all

# 评估特定任务
python eval_clue.py \
    --pretrained_path /path/to/model \
    --llama_config /path/to/config.json \
    --tokenizer_path /path/to/tokenizer.model \
    --data_dir data/clue \
    --tasks afqmc tnews cmnli

# 快速测试模式
python eval_clue.py \
    --pretrained_path /path/to/model \
    --llama_config /path/to/config.json \
    --tokenizer_path /path/to/tokenizer.model \
    --data_dir data/clue \
    --tasks afqmc \
    --max_eval_samples 10 \
    --ntrain 2
```

## 参数说明

### 数据参数
- `--data_dir`: CLUE数据集目录，默认`data/clue`
- `--tasks`: 要评估的任务，可选`all`或具体任务名，如`afqmc tnews`
- `--ntrain`: Few-shot示例数量，默认5
- `--max_eval_samples`: 每个任务最大评估样本数（用于快速测试）

### 模型参数
- `--pretrained_path`: 预训练模型路径
- `--llama_type`: 模型类型，默认`llama`
- `--llama_config`: 模型配置文件路径
- `--tokenizer_path`: 分词器路径
- `--max_seq_len`: 最大输入序列长度，默认2048
- `--quant`: 是否使用4bit量化

### 生成参数
- `--temperature`: 生成温度，默认0.1
- `--top_p`: Top-p采样参数，默认0.9
- `--max_gen_len`: 最大生成长度，默认256

## 输出结果

评估结果将保存在`results/{model_name}/clue/evaluation_results.json`，格式如下：

```json
{
  "afqmc": {
    "accuracy": 0.7234
  },
  "tnews": {
    "accuracy": 0.5678
  },
  "cmnli": {
    "accuracy": 0.6543
  },
  "average": 0.6485
}
```

## 自定义任务

如需添加新的CLUE任务，可以在`src/clue/clue_tasks.py`中：

1. 创建新的任务类，继承`CLUETask`
2. 实现必要的方法：
   - `load_data()`: 加载数据
   - `format_example()`: 格式化样本
   - `extract_answer()`: 提取答案
   - `compute_metric()`: 计算指标
3. 在`TASK_REGISTRY`中注册新任务

## 注意事项

1. **内存要求**：大模型评估需要充足的GPU内存
2. **时间成本**：完整评估所有任务可能需要数小时
3. **Few-shot设置**：可以通过调整`ntrain`参数控制few-shot示例数量
4. **快速验证**：使用`--max_eval_samples`参数进行快速验证

## 故障排除

### 问题1：下载数据失败
- 检查网络连接
- 尝试使用代理或镜像源
- 手动下载数据并解压到指定目录

### 问题2：内存不足
- 使用`--quant`参数启用4bit量化
- 减少`--max_seq_len`参数
- 使用更小的模型

### 问题3：评估速度慢
- 使用`--max_eval_samples`限制评估样本数
- 只评估部分任务而非全部
- 使用多GPU并行评估

## 引用

如果使用本代码，请引用CLUE：

```bibtex
@article{xu2020clue,
  title={CLUE: A Chinese Language Understanding Evaluation Benchmark},
  author={Xu, Liang and others},
  journal={arXiv preprint arXiv:2004.05986},
  year={2020}
}
```

## 联系方式

如有问题或建议，请提交Issue或联系维护者。