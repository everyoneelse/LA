# OpenSeek-Pretrain-100B 数据加载器

这个项目提供了加载 OpenSeek-Pretrain-100B 数据集的完整解决方案。该数据集使用 `.bin` 和 `.idx` 文件格式，这是 Megatron-LM 框架的标准索引数据格式。

## 文件说明

### 核心文件

1. **`load_openseek_data.py`** - 完整的数据加载器实现
   - 包含 `IndexedDataset` 类
   - 支持读取 `.bin` 和 `.idx` 文件格式
   - 提供详细的数据分析功能

2. **`simple_data_loader.py`** - 简化版加载器
   - 支持 Megatron-LM 官方工具
   - 提供自定义加载器回退选项

3. **`flexible_data_loader.py`** - 灵活的交互式加载器
   - 自动搜索数据文件
   - 交互式文件选择
   - 支持命令行参数

4. **`openseek_data_usage.py`** - 完整的使用指南
   - 数据分析和统计
   - PyTorch 集成示例
   - 性能优化建议

## 快速开始

### 1. 基本使用

```python
from load_openseek_data import IndexedDataset

# 加载数据集（假设您的数据文件为 data_prefix.bin 和 data_prefix.idx）
dataset = IndexedDataset('/path/to/your/data_prefix')

# 获取数据集信息
print(f"文档总数: {len(dataset)}")
print(f"数据类型: {dataset.dtype}")

# 读取第一个文档
first_doc = dataset[0]
print(f"第一个文档长度: {len(first_doc)}")
print(f"前10个token: {first_doc[:10]}")

# 关闭数据集
dataset.close()
```

### 2. 交互式使用

```bash
# 在指定目录搜索并选择数据文件
python3 flexible_data_loader.py --data-dir /path/to/your/data

# 直接指定数据文件前缀
python3 flexible_data_loader.py --data-prefix /path/to/your/data_prefix

# 仅列出找到的数据文件
python3 flexible_data_loader.py --list-only --data-dir /path/to/your/data
```

### 3. 使用 Megatron-LM 官方工具

```python
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

# 需要先安装: pip install megatron-lm
dataset = make_indexed_dataset('/path/to/your/data_prefix', 'mmap', False)
```

## 数据格式说明

OpenSeek-Pretrain-100B 数据集使用以下格式：

- **`.bin` 文件**: 包含实际的 token 数据（二进制格式）
- **`.idx` 文件**: 包含索引信息，用于快速定位每个文档的位置和长度

### 索引文件结构
```
- Magic number (8 bytes)
- Version (8 bytes) 
- Data type code (1 byte)
- Document count (8 bytes)
- Document lengths (4 bytes × document_count)
- Document offsets (8 bytes × document_count)
```

## PyTorch 集成

```python
import torch
from torch.utils.data import Dataset, DataLoader

class OpenSeekDataset(Dataset):
    def __init__(self, data_prefix):
        from load_openseek_data import IndexedDataset
        self.dataset = IndexedDataset(data_prefix)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return torch.tensor(data, dtype=torch.long)

# 使用示例
pytorch_dataset = OpenSeekDataset('/path/to/your/data_prefix')
dataloader = DataLoader(pytorch_dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    print(f"批次形状: {batch.shape}")
    break
```

## 性能优化建议

1. **内存映射**: 使用 `mmap` 模式处理大文件，避免将整个文件加载到内存
2. **批量读取**: 使用 `get_batch()` 方法批量读取多个文档
3. **合理的批次大小**: 根据您的硬件配置设置合适的批次大小
4. **多进程**: 在 DataLoader 中使用 `num_workers` 参数启用多进程数据加载

## 故障排除

### 常见问题

1. **文件未找到错误**
   ```
   FileNotFoundError: Binary file not found: /path/to/file.bin
   ```
   - 检查文件路径是否正确
   - 确保 `.bin` 和 `.idx` 文件都存在
   - 使用 `flexible_data_loader.py --list-only` 搜索数据文件

2. **依赖缺失**
   ```
   ModuleNotFoundError: No module named 'numpy'
   ```
   - 安装必要依赖: `sudo apt install python3-numpy`
   - 或使用虚拟环境: `pip install numpy`

3. **内存不足**
   - 使用内存映射模式
   - 减小批次大小
   - 使用数据流式处理

### 调试步骤

1. 检查数据文件完整性
2. 验证文件权限
3. 测试小批次数据加载
4. 检查系统内存使用情况

## 示例脚本运行

```bash
# 测试基本功能
python3 simple_data_loader.py

# 完整功能演示
python3 openseek_data_usage.py

# 交互式数据加载
python3 flexible_data_loader.py
```

## 注意事项

1. 数据文件通常很大（几GB到几TB），确保有足够的存储空间
2. 首次加载可能需要一些时间来读取索引文件
3. 建议在SSD上存储数据文件以获得更好的I/O性能
4. 对于非常大的数据集，考虑使用分布式数据加载

## 支持的数据类型

- `uint8`, `int8`, `int16`, `int32`, `int64`
- `float32`, `float64`, `uint16`

## 扩展功能

- 支持自定义 tokenizer 解码
- 数据统计和分析
- 批量数据处理
- 与深度学习框架集成

如有问题，请检查错误日志并参考故障排除部分。