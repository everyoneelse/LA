# Megatron-LM 数据处理集成 - 安装和使用指南

## 快速开始

我已经成功将 Megatron-LM 的 indexed_dataset 读取功能集成到你的 packed dataset 生成代码中。现在你可以同时处理 Parquet 文件和 Megatron 格式的数据。

## 文件说明

- `pack_tokens_enhanced.py` - 增强版的主处理脚本（支持 Megatron + Parquet）
- `simple_test.py` - 简化的测试脚本
- `test_megatron_integration.py` - 完整的功能测试脚本
- `MEGATRON_INTEGRATION_README.md` - 详细的技术文档

## 安装依赖

### 1. 基础 Python 包

```bash
# 如果可以使用 pip
pip install pandas pyarrow tqdm sentencepiece

# 或者使用系统包管理器（Ubuntu/Debian）
sudo apt update
sudo apt install python3-pandas python3-pyarrow python3-tqdm
```

### 2. Megatron-LM

```bash
# 方法1：从 PyPI 安装
pip install megatron-lm

# 方法2：从源码安装（推荐）
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

### 3. SentencePiece（用于 tokenizer）

```bash
pip install sentencepiece
# 或者
sudo apt install python3-sentencepiece
```

## 核心功能

### 自动文件类型检测

脚本会自动识别数据文件类型：

```python
def detect_file_type(filename):
    if filename.endswith('.parquet'):
        return 'parquet'
    elif os.path.exists(filename + '.idx') and os.path.exists(filename + '.bin'):
        return 'megatron'
    return 'unknown'
```

### 统一处理接口

无论是什么格式，都使用相同的函数：

```python
# 自动检测并处理
result = pack_tokens(filename, save_dir, tokenizer)
```

## 使用方法

### 基本使用

```python
from pack_tokens_enhanced import scan_for_datasets, process_with_progress
from accessory.model.tokenizer import Tokenizer

# 配置参数
max_len = 1024
tokenizer = Tokenizer('./internlm2-chat-126m/tokenizer.model')

# 扫描数据目录（支持混合格式）
data_dirs = ['CCI-DATA', 'MEGATRON-DATA']
files = scan_for_datasets(data_dirs)

# 处理所有文件
save_dir = "processed_tokens"
os.makedirs(save_dir, exist_ok=True)
process_with_progress(files, save_dir, tokenizer, num_workers=24)
```

### 处理单个文件

```python
# 处理单个 Parquet 文件
pack_tokens('data.parquet', save_dir, tokenizer)

# 处理单个 Megatron 数据集
pack_tokens('output_file', save_dir, tokenizer)  # 需要 output_file.idx 和 output_file.bin
```

## 数据格式说明

### Parquet 文件
- 包含 `content` 列的文本数据
- 会使用 InternLM2 tokenizer 进行编码

### Megatron 数据集
- 由 `.idx` 和 `.bin` 文件组成
- 包含已经 tokenized 的数据（使用 Qwen tokenizer）
- 直接读取 token IDs

## 重要注意事项

### Token 兼容性问题

由于 Megatron 数据使用 Qwen tokenizer，而你使用 InternLM2 tokenizer，存在兼容性考虑：

1. **当前实现**: 直接使用 Megatron 中的 token IDs
2. **完全兼容方案**: 需要 token 转换（见下文）

### Token 转换方案（可选）

如果需要完全兼容，可以实现以下转换逻辑：

```python
def convert_qwen_to_internlm2(qwen_tokens, qwen_tokenizer, internlm_tokenizer):
    """将 Qwen tokens 转换为 InternLM2 tokens"""
    try:
        # 解码为文本
        text = qwen_tokenizer.decode(qwen_tokens)
        # 重新编码
        new_tokens = internlm_tokenizer.encode(text, bos=True, eos=True)
        return new_tokens
    except:
        # 如果转换失败，返回原始 tokens
        return qwen_tokens
```

## 性能优化

- **内存映射**: Megatron 数据使用 mmap 访问，支持大文件
- **多进程处理**: 支持并行处理多个文件
- **分段写入**: 避免内存溢出
- **断点续传**: 跳过已处理的文件

## 测试你的环境

运行测试脚本检查环境：

```bash
python3 simple_test.py
```

如果看到以下输出表示一切正常：
```
✓ 所有核心组件都可用，可以正常使用集成功能
```

## 故障排除

### 1. ModuleNotFoundError: No module named 'megatron'

**解决方案**:
```bash
pip install megatron-lm
```

### 2. ModuleNotFoundError: No module named 'sentencepiece'

**解决方案**:
```bash
pip install sentencepiece
```

### 3. ModuleNotFoundError: No module named 'pandas'

**解决方案**:
```bash
pip install pandas pyarrow
```

### 4. 虚拟环境问题

如果系统不允许全局安装包，创建虚拟环境：

```bash
# 安装 venv 支持
sudo apt install python3-venv

# 创建虚拟环境
python3 -m venv myenv
source myenv/bin/activate

# 安装依赖
pip install pandas pyarrow tqdm sentencepiece megatron-lm
```

### 5. Tokenizer 路径问题

确保 tokenizer 文件存在：
```bash
ls -la ./internlm2-chat-126m/tokenizer.model
```

如果不存在，需要下载或指定正确路径。

## 示例输出

成功运行后你会看到类似输出：

```
Scanning directory: CCI-DATA
Found 15 parquet files
Found 3 megatron dataset files
Total files found: 18

需要处理 18 个文件
Processing files: 100%|██████████| 18/18 [15:30<00:00, 51.67s/file]

Processing Megatron dataset: output_file
Megatron 数据集大小: 50000
Total tokens so far: 52428800
Final total tokens: 52428800
output_file_megatron.pkl finished
```

## 下一步

1. 安装所需依赖
2. 准备你的数据文件（Parquet 和/或 Megatron 格式）
3. 运行 `pack_tokens_enhanced.py`
4. 检查输出的 `.pkl` 文件

现在你可以无缝处理两种数据格式，大大提高了数据处理的灵活性！