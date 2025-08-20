#!/usr/bin/env python3
"""
集成示例：展示如何将 Rust tokenizer 集成到现有项目中
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def integrate_with_existing_code():
    """
    展示如何将 Rust tokenizer 集成到现有的代码中
    """
    print("="*60)
    print("集成 Rust Tokenizer 到现有项目")
    print("="*60)
    
    print("\n方法 1: 直接替换导入")
    print("-"*40)
    print("""
原始代码:
    from accessory.model.tokenizer import Tokenizer
    
修改为:
    from rust_tokenizer import RustTokenizerWrapper as Tokenizer
    
这样可以保持 API 完全兼容，无需修改其他代码。
    """)
    
    print("\n方法 2: 条件导入（渐进式迁移）")
    print("-"*40)
    print("""
import os

# 通过环境变量控制使用哪个 tokenizer
USE_RUST_TOKENIZER = os.getenv('USE_RUST_TOKENIZER', 'false').lower() == 'true'

if USE_RUST_TOKENIZER:
    try:
        from rust_tokenizer import RustTokenizerWrapper as Tokenizer
        print("Using Rust tokenizer for better performance")
    except ImportError:
        from accessory.model.tokenizer import Tokenizer
        print("Rust tokenizer not available, using Python implementation")
else:
    from accessory.model.tokenizer import Tokenizer
    """)
    
    print("\n方法 3: 创建适配器类")
    print("-"*40)
    print("""
class TokenizerAdapter:
    '''统一的 tokenizer 接口'''
    
    def __init__(self, model_path: str, use_rust: bool = True):
        if use_rust:
            try:
                from rust_tokenizer import RustTokenizerWrapper
                self._impl = RustTokenizerWrapper(model_path)
                self.backend = "rust"
            except ImportError:
                from accessory.model.tokenizer import Tokenizer
                self._impl = Tokenizer(model_path)
                self.backend = "python"
        else:
            from accessory.model.tokenizer import Tokenizer
            self._impl = Tokenizer(model_path)
            self.backend = "python"
    
    def encode(self, text: str, bos: bool = False, eos: bool = False):
        return self._impl.encode(text, bos, eos)
    
    def decode(self, ids):
        return self._impl.decode(ids)
    
    # ... 其他方法的代理
    """)


def performance_comparison_example():
    """
    性能对比示例
    """
    print("\n" + "="*60)
    print("性能对比示例代码")
    print("="*60)
    
    print("""
import time
import statistics

def benchmark_tokenizer(tokenizer, texts, name="Tokenizer"):
    '''对 tokenizer 进行基准测试'''
    
    # 预热
    for _ in range(10):
        tokenizer.encode("warmup text", bos=True, eos=True)
    
    # 测试单个编码
    start = time.perf_counter()
    for text in texts:
        tokenizer.encode(text, bos=True, eos=True)
    single_time = time.perf_counter() - start
    
    print(f"{name}:")
    print(f"  Single encoding: {single_time:.3f}s for {len(texts)} texts")
    print(f"  Throughput: {len(texts)/single_time:.0f} texts/sec")
    
    # 如果支持批量编码
    if hasattr(tokenizer, 'batch_encode'):
        start = time.perf_counter()
        tokenizer.batch_encode(texts, bos=True, eos=True)
        batch_time = time.perf_counter() - start
        print(f"  Batch encoding: {batch_time:.3f}s")
        print(f"  Speedup: {single_time/batch_time:.1f}x")

# 使用示例
test_texts = ["Sample text " + str(i) for i in range(1000)]

# Python tokenizer
from accessory.model.tokenizer import Tokenizer as PythonTokenizer
py_tokenizer = PythonTokenizer("model.bin")
benchmark_tokenizer(py_tokenizer, test_texts, "Python Tokenizer")

# Rust tokenizer
from rust_tokenizer import RustTokenizerWrapper
rust_tokenizer = RustTokenizerWrapper("model.bin")
benchmark_tokenizer(rust_tokenizer, test_texts, "Rust Tokenizer")
    """)


def migration_guide():
    """
    迁移指南
    """
    print("\n" + "="*60)
    print("迁移指南")
    print("="*60)
    
    print("""
1. 安装 Rust 工具链
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

2. 构建 Rust tokenizer
   cd rust_tokenizer
   pip install maturin
   maturin develop --release

3. 验证兼容性
   - 运行测试套件确保功能正常
   - 对比输出结果确保一致性

4. 渐进式迁移
   - 先在开发环境测试
   - 使用环境变量控制切换
   - 监控性能指标
   - 逐步推广到生产环境

5. 性能调优
   - 使用批量操作 API
   - 启用并行处理
   - 调整 batch size

6. 问题排查
   - 检查 Rust 编译日志
   - 使用 RUST_LOG=debug 查看详细信息
   - 对比 Python 和 Rust 版本的输出
    """)


def main():
    """主函数"""
    print("Rust Tokenizer 集成示例\n")
    
    # 展示集成方法
    integrate_with_existing_code()
    
    # 展示性能对比
    performance_comparison_example()
    
    # 展示迁移指南
    migration_guide()
    
    print("\n" + "="*60)
    print("更多信息请参考 README.md")
    print("="*60)


if __name__ == "__main__":
    main()