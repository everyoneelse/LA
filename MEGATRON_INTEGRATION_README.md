# Megatron-LM 数据集成功能说明

## 概述

本项目已集成了 Megatron-LM indexed dataset 的读取功能，可以同时处理 Parquet 文件和 Megatron 格式的数据集。

## 主要功能

### 1. 自动文件类型检测

脚本会自动检测数据文件的类型：
- **Parquet 文件**: `.parquet` 后缀的文件
- **Megatron 数据集**: 具有 `.idx` 和 `.bin` 文件对的数据集
- **未知类型**: 其他格式的文件会被跳过

### 2. 统一的数据处理接口

无论是 Parquet 还是 Megatron 格式，都使用相同的 `pack_tokens()` 函数进行处理。

### 3. Megatron 数据处理特点

- 直接读取已经 tokenized 的数据
- 支持大型数据集的内存映射访问
- 保持与原有 Parquet 处理流程的一致性

## 使用方法

### 安装依赖

```bash
# 安装 Megatron-LM
pip install megatron-lm

# 或者从源码安装
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

### 准备数据

确保你的 Megatron 数据集包含以下文件：
```
your_dataset.idx  # 索引文件
your_dataset.bin  # 数据文件
```

### 运行处理脚本

```python
from pack_tokens_enhanced import scan_for_datasets, process_with_progress
from accessory.model.tokenizer import Tokenizer

# 配置参数
max_len = 1024
tokenizer = Tokenizer('./internlm2-chat-126m/tokenizer.model')

# 扫描数据目录（支持混合格式）
data_dirs = ['CCI-DATA', 'MEGATRON-DATA']  # 可以包含不同格式的数据
files = scan_for_datasets(data_dirs)

# 处理所有文件
save_dir = "processed_tokens"
process_with_progress(files, save_dir, tokenizer, num_workers=24)
```

### 测试功能

运行测试脚本验证集成是否正常：

```bash
python test_megatron_integration.py
```

## 代码结构

### 核心函数

1. **`detect_file_type(filename)`**: 自动检测文件类型
2. **`read_megatron_dataset(dataset_prefix)`**: 读取 Megatron 数据集
3. **`pack_tokens_megatron()`**: 处理 Megatron 格式数据
4. **`pack_tokens_parquet()`**: 处理 Parquet 格式数据
5. **`pack_tokens()`**: 统一入口函数
6. **`scan_for_datasets()`**: 扫描并发现数据集文件

### 处理流程

```
输入文件 → 文件类型检测 → 选择处理函数 → Token 打包 → 保存结果
     ↓              ↓              ↓           ↓         ↓
   混合格式    parquet/megatron   专门处理    统一格式   .pkl文件
```

## 注意事项

### Token 兼容性

由于 Megatron 数据集中的 tokens 是使用 Qwen tokenizer 生成的，而你使用的是 InternLM2 tokenizer，存在以下考虑：

1. **直接使用**: 当前实现直接使用 Megatron 中的 token IDs
2. **重新编码**: 如果需要完全兼容，可能需要先解码再重新编码
3. **词汇表映射**: 可以创建两个 tokenizer 之间的映射关系

### 性能优化

- 使用内存映射 (`mmap`) 访问大型 Megatron 数据集
- 支持多进程并行处理
- 分段写入避免内存溢出

### 错误处理

- 自动跳过损坏或无法读取的文档
- 提供详细的错误信息和进度反馈
- 支持断点续传（跳过已处理的文件）

## 扩展功能

### 自定义 Token 转换

如果需要在 Qwen 和 InternLM2 tokenizer 之间进行转换，可以修改 `pack_tokens_megatron()` 函数：

```python
def convert_tokens(qwen_tokens, qwen_tokenizer, internlm_tokenizer):
    """将 Qwen tokens 转换为 InternLM2 tokens"""
    # 解码为文本
    text = qwen_tokenizer.decode(qwen_tokens)
    # 重新编码
    new_tokens = internlm_tokenizer.encode(text, bos=True, eos=True)
    return new_tokens
```

### 数据统计

脚本会输出详细的处理统计信息：
- 处理的文档数量
- 生成的 token 总数
- 各种文件类型的分布

## 故障排除

### 常见问题

1. **ImportError: No module named 'megatron'**
   - 安装 Megatron-LM: `pip install megatron-lm`

2. **FileNotFoundError: .idx or .bin file not found**
   - 确保 Megatron 数据集的 `.idx` 和 `.bin` 文件都存在

3. **Memory Error**
   - 减少 `flush_segments` 参数
   - 降低并行进程数量

4. **Token 不兼容**
   - 检查 tokenizer 词汇表
   - 考虑实现 token 转换逻辑

### 调试建议

1. 使用测试脚本验证各个组件
2. 检查数据文件的完整性
3. 监控内存和磁盘使用情况
4. 查看详细的错误日志

## 示例输出

```
Scanning directory: CCI-DATA
Found 15 parquet files
Found 3 megatron dataset files
Total files found: 18
  1. CCI-DATA/data1.parquet (parquet)
  2. CCI-DATA/data2.parquet (parquet)
  3. CCI-DATA/output_file (megatron)
  ... and 15 more files

Processing Megatron dataset: CCI-DATA/output_file
Megatron 数据集大小: 50000
Processing CCI-DATA/output_file: 100%|██████████| 50000/50000 [05:30<00:00, 151.52it/s]
Total tokens so far: 52428800
Final total tokens: 52428800
CCI-DATA/packed_tokens/output_file_megatron.pkl finished
```

这样的集成方案让你可以无缝处理不同格式的训练数据，同时保持代码的简洁性和可维护性。