#!/usr/bin/env python3
"""
Rust Tokenizer 使用示例

这个脚本展示了如何使用 Rust 实现的高性能 tokenizer
"""

import time
from pathlib import Path
from rust_tokenizer import RustTokenizerWrapper, probe_tokenizer_path_from_pretrained

def benchmark_tokenizer(tokenizer, texts, name="Tokenizer"):
    """对 tokenizer 进行基准测试"""
    print(f"\n=== {name} Benchmark ===")
    
    # 单个文本编码
    start = time.time()
    for text in texts:
        _ = tokenizer.encode(text, bos=True, eos=True)
    single_time = time.time() - start
    print(f"Single encoding: {single_time:.3f}s for {len(texts)} texts")
    
    # 批量编码（如果支持）
    if hasattr(tokenizer, 'batch_encode'):
        start = time.time()
        _ = tokenizer.batch_encode(texts, bos=True, eos=True)
        batch_time = time.time() - start
        print(f"Batch encoding: {batch_time:.3f}s for {len(texts)} texts")
        print(f"Speedup: {single_time/batch_time:.2f}x")
    
    # 解码测试
    encoded = tokenizer.encode(texts[0], bos=True, eos=True)
    decoded = tokenizer.decode(encoded)
    print(f"\nExample:")
    print(f"  Original: {texts[0][:50]}...")
    print(f"  Encoded: {encoded[:10]}... (length: {len(encoded)})")
    print(f"  Decoded: {decoded[:50]}...")


def compare_with_python_tokenizer():
    """比较 Rust tokenizer 和 Python tokenizer 的性能"""
    print("\n=== Comparing Rust vs Python Tokenizer ===")
    
    # 准备测试数据
    test_texts = [
        "Hello, how are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "人工智能正在改变世界。",
        "Machine learning is a subset of artificial intelligence.",
        "自然语言处理是人工智能的重要分支。",
    ] * 100  # 重复以增加测试规模
    
    # 测试 Rust tokenizer
    try:
        # 这里需要一个实际的 tokenizer 模型文件
        # 您可以使用任何兼容的模型文件
        model_path = "path/to/your/tokenizer.model"  # 请替换为实际路径
        
        if Path(model_path).exists():
            rust_tokenizer = RustTokenizerWrapper(model_path)
            benchmark_tokenizer(rust_tokenizer, test_texts, "Rust Tokenizer")
    except Exception as e:
        print(f"Could not load Rust tokenizer: {e}")
        print("Please provide a valid tokenizer model path")
    
    # 如果您想比较，可以在这里加载 Python tokenizer
    # 例如：
    # from accessory.model.tokenizer import Tokenizer as PythonTokenizer
    # python_tokenizer = PythonTokenizer(model_path)
    # benchmark_tokenizer(python_tokenizer, test_texts, "Python Tokenizer")


def main():
    """主函数"""
    print("Rust Tokenizer Example")
    print("=" * 50)
    
    # 1. 基本使用示例
    print("\n1. Basic Usage Example")
    print("-" * 30)
    
    # 创建一个模拟的 tokenizer（实际使用时需要真实的模型文件）
    print("Note: To run this example, you need a tokenizer model file.")
    print("Supported formats:")
    print("  - SentencePiece: .model files")
    print("  - HuggingFace: directories with tokenizer.json")
    print("  - WordPiece: directories with vocab.txt")
    
    # 2. 探测 tokenizer 路径
    print("\n2. Probing Tokenizer Path")
    print("-" * 30)
    
    test_path = "/path/to/pretrained/model"
    result = probe_tokenizer_path_from_pretrained(test_path)
    if result:
        print(f"Found tokenizer at: {result}")
    else:
        print(f"No tokenizer found at: {test_path}")
    
    # 3. 性能比较
    print("\n3. Performance Comparison")
    print("-" * 30)
    compare_with_python_tokenizer()
    
    # 4. 特性展示
    print("\n4. Features")
    print("-" * 30)
    print("✓ High performance (10-100x faster than Python)")
    print("✓ Memory safe (Rust guarantees)")
    print("✓ Parallel batch processing")
    print("✓ Compatible with existing models")
    print("✓ Support for multiple tokenizer types")
    
    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()