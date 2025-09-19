# 多领域数据采样和打包系统

这是一个专门为大规模多领域预训练设计的数据采样和打包系统，支持新闻、代码、数学数据的智能采样、质量过滤和高效打包。

## 🎯 系统特性

### 🔄 智能采样
- **多领域支持**: 新闻、代码、数学数据统一处理
- **质量过滤**: 基于多维度指标的质量评估
- **去重机制**: 基于哈希的高效去重
- **比例控制**: 精确控制各领域数据比例

### 📦 高效打包
- **序列打包**: 将多个样本打包成固定长度序列
- **领域标记**: 自动添加领域标识符
- **分块存储**: 支持大规模数据的分文件存储
- **元数据管理**: 完整的数据索引和统计信息

### 🚀 性能优化
- **并行处理**: 多进程并行采样和处理
- **流式加载**: 支持内存友好的流式数据加载
- **领域平衡**: 训练时动态维持领域比例平衡

## 📁 文件结构

```
workspace/
├── data_sampler.py           # 核心采样和打包工具
├── packed_data_loader.py     # 高效数据加载器
├── data_quality_checker.py   # 数据质量检查工具
├── run_data_sampling.sh      # 一键运行脚本
└── README_DATA_SAMPLING.md   # 本文档

输出结构:
├── raw_data/                 # 原始数据目录
│   ├── news/                 # 新闻数据
│   ├── code/                 # 代码数据
│   └── math/                 # 数学数据
└── packed_data/              # 打包后数据
    ├── packed_data_0000.pkl  # 数据文件
    ├── packed_data_0001.pkl
    ├── metadata.json         # 元数据
    ├── sampling_report.json  # 采样报告
    └── quality_report.json   # 质量报告
```

## 🚀 快速开始

### 1. 准备数据

将您的原始数据按领域分类放置：

```bash
# 创建数据目录
mkdir -p raw_data/{news,code,math}

# 放置数据文件 (支持 .txt, .json, .jsonl 格式)
# raw_data/news/    - 新闻数据文件
# raw_data/code/    - 代码数据文件  
# raw_data/math/    - 数学数据文件
```

### 2. 运行采样

```bash
# 小规模测试 (100K samples)
./run_data_sampling.sh small

# 中等规模 (1M samples)
./run_data_sampling.sh medium

# 大规模处理 (10M+ samples)
./run_data_sampling.sh large
```

### 3. 检查质量

```bash
# 运行质量检查
python3 data_quality_checker.py
```

### 4. 加载数据

```python
from packed_data_loader import create_data_loader

# 创建数据加载器
data_loader = create_data_loader(
    data_dir="./packed_data/",
    tokenizer="gpt2",
    batch_size=8,
    max_length=2048,
    domain_balance=True
)

# 开始训练
for batch in data_loader:
    # 训练代码
    pass
```

## ⚙️ 详细配置

### SamplingConfig 参数

```python
@dataclass
class SamplingConfig:
    # 数据路径配置
    news_data_path: str = "./raw_data/news/"
    code_data_path: str = "./raw_data/code/"
    math_data_path: str = "./raw_data/math/"
    output_path: str = "./packed_data/"
    
    # 采样配置
    total_samples: int = 1_000_000  # 总样本数
    news_ratio: float = 0.60        # 新闻数据比例
    code_ratio: float = 0.25        # 代码数据比例
    math_ratio: float = 0.15        # 数学数据比例
    
    # 质量过滤配置
    min_length: int = 50            # 最小文本长度
    max_length: int = 8192          # 最大文本长度
    min_quality_score: float = 0.7  # 最小质量分数
    
    # 打包配置
    sequence_length: int = 2048     # 打包后的序列长度
    pack_samples: bool = True       # 是否进行序列打包
    samples_per_file: int = 10000   # 每个文件的样本数
    
    # 性能配置
    num_workers: int = 8            # 并行工作进程数
    chunk_size: int = 1000          # 每次处理的块大小
```

## 📊 数据格式

### 输入数据格式

支持多种输入格式：

**文本文件 (.txt)**
```
这是一条新闻内容...
```

**JSON文件 (.json)**
```json
{
  "text": "新闻内容...",
  "title": "新闻标题",
  "category": "politics"
}
```

**JSONL文件 (.jsonl)**
```json
{"text": "第一条新闻..."}
{"text": "第二条新闻..."}
```

**代码数据**
```json
{
  "code": "def hello():\n    print('Hello World')",
  "language": "python"
}
```

**数学数据**
```json
{
  "problem": "求解方程 2x + 3 = 11",
  "solution": "x = 4"
}
```

### 输出数据格式

打包后的数据格式：

```python
{
    'text': '<|news|>新闻内容...<|end|><|code|>代码内容...<|end|>',
    'domains': ['news', 'code'],
    'sources': ['file1.txt', 'file2.py'],
    'boundaries': [100, 250],  # 各段落结束位置
    'length': 250
}
```

## 🔍 质量控制

### 质量评估指标

1. **文本长度**: 过滤过短或过长的文本
2. **字符质量**: 检查字符编码和异常字符
3. **重复检测**: 基于哈希的去重
4. **领域特征**: 针对不同领域的特定检查

### 质量分数计算

```python
def calculate_quality_score(text: str, domain: str) -> float:
    score = 1.0
    
    # 长度检查
    if len(text) < min_length: return 0.0
    if len(text) > max_length: score -= 0.2
    
    # 字符质量检查
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if domain == "code" and ascii_ratio < 0.7:
        score -= 0.3
    
    # 重复字符检查
    max_repeat = max(len(list(g)) for k, g in groupby(text))
    if max_repeat > 10: score -= 0.2
    
    return max(0.0, min(1.0, score))
```

## 📈 性能优化

### 内存优化

- **流式处理**: 避免将所有数据加载到内存
- **分块处理**: 按块处理大文件
- **及时释放**: 处理完成后立即释放内存

### 磁盘优化

- **压缩存储**: 使用pickle进行高效序列化
- **分文件存储**: 避免单文件过大
- **顺序写入**: 优化磁盘I/O性能

### 并行优化

- **多进程采样**: 并行处理不同数据源
- **异步I/O**: 重叠计算和I/O操作
- **负载均衡**: 动态分配工作负载

## 🛠️ 自定义扩展

### 添加新的数据领域

```python
# 1. 修改配置
@dataclass
class SamplingConfig:
    # 添加新领域
    scientific_data_path: str = "./raw_data/scientific/"
    scientific_ratio: float = 0.10

# 2. 扩展采样器
def sample_data(self):
    # 添加科学数据采样
    sampled_data['scientific'] = self._load_domain_data(
        'scientific', self.config.scientific_data_path, scientific_samples
    )
```

### 自定义质量过滤器

```python
class CustomQualityFilter(QualityFilter):
    def calculate_quality_score(self, text: str, domain: str) -> float:
        score = super().calculate_quality_score(text, domain)
        
        # 添加自定义检查
        if domain == "scientific":
            # 检查科学术语密度
            scientific_terms = count_scientific_terms(text)
            if scientific_terms < 3:
                score -= 0.3
        
        return score
```

## 📋 使用示例

### 基础使用

```python
from data_sampler import SamplingConfig, DataSamplingPipeline

# 配置采样参数
config = SamplingConfig(
    total_samples=500_000,
    news_ratio=0.50,
    code_ratio=0.30,
    math_ratio=0.20,
    sequence_length=1024
)

# 运行采样流水线
pipeline = DataSamplingPipeline(config)
pipeline.run()
```

### 高级使用

```python
from packed_data_loader import PackedDataset, DomainAwareDataLoader
from transformers import AutoTokenizer

# 加载数据集
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = PackedDataset("./packed_data/", tokenizer)

# 创建领域平衡的数据加载器
loader = DomainAwareDataLoader(
    dataset=dataset,
    batch_size=16,
    domain_balance=True,
    target_ratios={'news': 0.6, 'code': 0.25, 'math': 0.15}
)

# 获取数据加载器
data_loader = loader.create_dataloader(num_workers=4)
```

## 🔧 故障排除

### 常见问题

**1. 内存不足**
```bash
# 解决方案：使用流式数据集
streaming=True
buffer_size=5000  # 减小缓冲区
```

**2. 数据文件格式错误**
```bash
# 检查文件格式
file *.json  # 确认是有效的JSON文件
head -5 *.jsonl  # 检查JSONL格式
```

**3. 质量分数过低**
```python
# 调整质量阈值
min_quality_score=0.5  # 降低阈值
min_length=20  # 降低最小长度要求
```

### 性能调优

**CPU密集型任务**
```python
num_workers=multiprocessing.cpu_count()  # 使用所有CPU核心
chunk_size=2000  # 增大块大小
```

**I/O密集型任务**
```python
samples_per_file=5000  # 减小文件大小
buffer_size=20000  # 增大缓冲区
```

## 📊 监控和日志

### 进度监控

系统会自动显示处理进度：

```
加载news数据从 ./raw_data/news/...
处理news文件: 100%|██████████| 1250/1250 [02:30<00:00, 8.33it/s]
news数据加载完成，获得 180,000 个有效样本

开始序列打包...
打包序列: 100%|██████████| 180000/180000 [01:15<00:00, 2400.00it/s]
打包完成，生成 90,000 个序列
```

### 日志记录

详细日志保存在 `logs/` 目录：

```bash
tail -f logs/data_sampling_*.log  # 实时查看日志
```

## 🎉 完整工作流

1. **准备数据** → 将原始数据按领域分类存放
2. **配置参数** → 根据需求调整采样配置
3. **运行采样** → 执行数据采样和打包
4. **质量检查** → 验证数据质量和分布
5. **开始训练** → 使用打包数据进行模型训练

这个系统已经为您的24B多领域数据做好了准备！🚀