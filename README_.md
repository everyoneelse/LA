# 小规模Pipeline验证实验

这是一个用于验证多领域预训练数据处理流程的小规模实验框架。在将24B新闻数据、代码数据和数学数据应用到900M参数模型之前，先用小规模数据验证整个pipeline的正确性。

## 🎯 实验目标

- ✅ 验证数据处理流程的正确性
- ✅ 测试不同领域数据的混合效果  
- ✅ 优化超参数和配比策略
- ✅ 评估训练稳定性和收敛性
- ✅ 为大规模训练提供参数指导

## 📊 实验设计

### 数据规模
- **总数据量**: 100M tokens (约为目标的1/240)
- **新闻数据**: 60M tokens (60%)
- **代码数据**: 25M tokens (25%)  
- **数学数据**: 15M tokens (15%)

### 模型配置
- **模型规模**: 125M参数 (DialoGPT-small)
- **训练步数**: 50K steps
- **批次大小**: 8
- **学习率**: 5e-5

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 一键运行完整实验
./run_pilot.sh

# 或分步运行
python3 pilot_experiment.py  # 运行训练实验
python3 data_analysis.py     # 分析结果
```

### 3. 查看结果

实验完成后，检查以下文件：
- `pilot_results/experiment_report.json` - 详细实验报告
- `pilot_results/training_curves.png` - 训练曲线图
- `pilot_results/data_distribution.png` - 数据分布分析
- `pilot_results/summary_report.md` - 总结报告

## 📁 项目结构

```
workspace/
├── pilot_experiment.py      # 主实验脚本
├── data_analysis.py         # 数据分析脚本
├── run_pilot.sh            # 一键运行脚本
├── requirements.txt        # 依赖包列表
├── README.md              # 项目说明
├── pilot_results/         # 实验结果目录
│   ├── experiment_report.json
│   ├── training_curves.png
│   ├── data_distribution.png
│   └── summary_report.md
├── pilot_data/           # 实验数据目录
└── logs/                # 日志文件目录
```

## 🔧 配置说明

### ExperimentConfig 参数

```python
@dataclass
class ExperimentConfig:
    # 数据配置
    total_tokens: int = 100_000_000    # 总token数
    news_ratio: float = 0.60           # 新闻数据比例
    code_ratio: float = 0.25           # 代码数据比例
    math_ratio: float = 0.15           # 数学数据比例
    
    # 模型配置
    model_name: str = "microsoft/DialoGPT-small"
    max_length: int = 512              # 最大序列长度
    
    # 训练配置
    batch_size: int = 8                # 批次大小
    learning_rate: float = 5e-5        # 学习率
    num_train_steps: int = 50_000      # 训练步数
```

## 📈 关键指标

### 训练指标
- **训练损失**: 监控模型学习进度
- **验证损失**: 评估泛化能力
- **困惑度**: 语言模型质量指标
- **梯度范数**: 训练稳定性指标

### 数据指标
- **领域分布**: 各领域数据比例
- **文本长度**: 平均长度和分布
- **数据质量**: 完整性、一致性评分
- **词汇统计**: 高频词和领域特征词

## 🎛️ 超参数调优指南

基于小规模实验结果，调整以下参数：

### 学习率
```python
# 如果loss下降太慢
learning_rate = 1e-4

# 如果loss震荡太大
learning_rate = 2e-5

# 如果梯度范数过大
learning_rate = 1e-5
```

### 数据配比
```python
# 如果代码能力不足
code_ratio = 0.35
news_ratio = 0.50

# 如果数学推理较弱
math_ratio = 0.25
news_ratio = 0.55
```

### 批次大小
```python
# GPU内存充足时
batch_size = 16

# 内存不足时
batch_size = 4
gradient_accumulation_steps = 4
```

## 🔍 结果分析

### 成功指标
- ✅ 训练损失稳定下降
- ✅ 验证损失不发散
- ✅ 梯度范数保持稳定
- ✅ 各领域数据均衡学习

### 问题诊断
- ❌ **损失不下降**: 学习率过小或数据质量问题
- ❌ **损失震荡**: 学习率过大或批次大小不当
- ❌ **过拟合**: 验证损失上升，需要正则化
- ❌ **梯度爆炸**: 梯度范数突增，需要梯度裁剪

## 🚀 扩展到大规模

验证成功后，应用到24B全量数据：

1. **数据处理**
   - 使用相同的预处理pipeline
   - 应用验证过的数据配比
   - 实施分布式数据加载

2. **模型训练**
   - 扩展到900M参数模型
   - 使用优化后的超参数
   - 实施多GPU/多节点训练

3. **监控策略**
   - 密切关注loss变化
   - 监控各领域学习进度
   - 设置自动保存和恢复

## 🤝 贡献指南

欢迎提交改进建议和bug报告：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

MIT License - 详见LICENSE文件

## 📞 支持

如有问题，请：
- 查看实验日志文件
- 检查requirements.txt依赖
- 确认GPU/CPU资源充足
- 联系开发团队获取支持

---

**祝您实验顺利！🎉**
