#!/usr/bin/env python3
"""
性能基准测试脚本

比较 Rust tokenizer 和 Python tokenizer 的性能
"""

import time
import random
import string
import statistics
from typing import List, Callable, Tuple
import json

# 生成测试数据
def generate_test_data(num_texts: int = 1000, min_length: int = 10, max_length: int = 200) -> List[str]:
    """生成随机测试文本"""
    texts = []
    for _ in range(num_texts):
        length = random.randint(min_length, max_length)
        # 混合英文、数字和中文字符
        text_parts = []
        
        # 英文部分
        english = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length//2))
        text_parts.append(english)
        
        # 中文部分
        chinese_chars = '的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严'
        chinese = ''.join(random.choices(chinese_chars, k=length//2))
        text_parts.append(chinese)
        
        random.shuffle(text_parts)
        texts.append(''.join(text_parts))
    
    return texts


def benchmark_function(func: Callable, *args, iterations: int = 3) -> Tuple[float, float]:
    """基准测试一个函数"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0


def run_tokenizer_benchmark(tokenizer, test_texts: List[str], name: str = "Tokenizer") -> dict:
    """运行 tokenizer 基准测试"""
    results = {
        "name": name,
        "num_texts": len(test_texts),
    }
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # 1. 单文本编码
    print("1. Single text encoding...")
    def single_encode():
        for text in test_texts:
            tokenizer.encode(text, bos=True, eos=True)
    
    mean_time, std_time = benchmark_function(single_encode)
    results["single_encode_time"] = mean_time
    results["single_encode_std"] = std_time
    print(f"   Time: {mean_time:.3f}s ± {std_time:.3f}s")
    print(f"   Throughput: {len(test_texts)/mean_time:.0f} texts/second")
    
    # 2. 批量编码（如果支持）
    if hasattr(tokenizer, 'batch_encode'):
        print("2. Batch encoding...")
        def batch_encode():
            tokenizer.batch_encode(test_texts, bos=True, eos=True)
        
        mean_time, std_time = benchmark_function(batch_encode)
        results["batch_encode_time"] = mean_time
        results["batch_encode_std"] = std_time
        speedup = results["single_encode_time"] / mean_time
        results["batch_speedup"] = speedup
        print(f"   Time: {mean_time:.3f}s ± {std_time:.3f}s")
        print(f"   Throughput: {len(test_texts)/mean_time:.0f} texts/second")
        print(f"   Speedup vs single: {speedup:.1f}x")
    
    # 3. 解码测试
    print("3. Decoding...")
    # 先编码一些文本用于解码测试
    encoded_texts = [tokenizer.encode(text, bos=True, eos=True) for text in test_texts[:100]]
    
    def single_decode():
        for tokens in encoded_texts:
            tokenizer.decode(tokens)
    
    mean_time, std_time = benchmark_function(single_decode)
    results["decode_time"] = mean_time
    results["decode_std"] = std_time
    print(f"   Time: {mean_time:.3f}s ± {std_time:.3f}s")
    print(f"   Throughput: {len(encoded_texts)/mean_time:.0f} sequences/second")
    
    # 4. 内存使用（粗略估计）
    import sys
    if hasattr(tokenizer, '__sizeof__'):
        size = sys.getsizeof(tokenizer)
        results["memory_bytes"] = size
        print(f"4. Memory usage: ~{size/1024/1024:.1f} MB")
    
    return results


def compare_tokenizers():
    """比较不同 tokenizer 实现"""
    print("\n" + "="*60)
    print("TOKENIZER PERFORMANCE COMPARISON")
    print("="*60)
    
    # 生成测试数据
    print("\nGenerating test data...")
    test_texts = generate_test_data(num_texts=1000)
    print(f"Generated {len(test_texts)} test texts")
    print(f"Average text length: {sum(len(t) for t in test_texts)/len(test_texts):.0f} characters")
    
    results = []
    
    # 测试 Rust tokenizer
    try:
        from rust_tokenizer import RustTokenizerWrapper
        # 需要提供实际的模型路径
        model_path = "path/to/your/tokenizer.model"
        
        if Path(model_path).exists():
            rust_tokenizer = RustTokenizerWrapper(model_path)
            rust_results = run_tokenizer_benchmark(rust_tokenizer, test_texts, "Rust Tokenizer")
            results.append(rust_results)
    except Exception as e:
        print(f"\nCould not test Rust tokenizer: {e}")
    
    # 测试 Python tokenizer（如果可用）
    try:
        import sys
        sys.path.insert(0, '/workspace')
        from accessory.model.tokenizer import Tokenizer as PythonTokenizer
        
        model_path = "path/to/your/tokenizer.model"
        if Path(model_path).exists():
            python_tokenizer = PythonTokenizer(model_path)
            python_results = run_tokenizer_benchmark(python_tokenizer, test_texts, "Python Tokenizer")
            results.append(python_results)
    except Exception as e:
        print(f"\nCould not test Python tokenizer: {e}")
    
    # 打印比较结果
    if len(results) >= 2:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        rust_res = next((r for r in results if "Rust" in r["name"]), None)
        python_res = next((r for r in results if "Python" in r["name"]), None)
        
        if rust_res and python_res:
            print(f"\nSpeedup (Rust vs Python):")
            print(f"  Single encoding: {python_res['single_encode_time']/rust_res['single_encode_time']:.1f}x faster")
            if 'batch_encode_time' in rust_res:
                print(f"  Batch encoding: {python_res.get('single_encode_time', 0)/rust_res['batch_encode_time']:.1f}x faster")
            print(f"  Decoding: {python_res['decode_time']/rust_res['decode_time']:.1f}x faster")
    
    # 保存结果
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")


def main():
    """主函数"""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Tokenizer Performance Benchmark")
    parser.add_argument("--model", type=str, help="Path to tokenizer model")
    parser.add_argument("--num-texts", type=int, default=1000, help="Number of test texts")
    parser.add_argument("--compare", action="store_true", help="Compare different implementations")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_tokenizers()
    else:
        # 运行单个 tokenizer 的基准测试
        if not args.model:
            print("Please provide a model path with --model or use --compare")
            return
        
        from rust_tokenizer import RustTokenizerWrapper
        tokenizer = RustTokenizerWrapper(args.model)
        test_texts = generate_test_data(num_texts=args.num_texts)
        results = run_tokenizer_benchmark(tokenizer, test_texts, "Rust Tokenizer")
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)


if __name__ == "__main__":
    main()