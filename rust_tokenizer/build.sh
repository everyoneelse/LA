#!/bin/bash

# 构建 Rust tokenizer 的脚本

echo "Building Rust Tokenizer..."

# 检查是否安装了 maturin
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# 开发模式构建（快速，用于测试）
build_dev() {
    echo "Building in development mode..."
    maturin develop --release
}

# 生产模式构建（优化，用于发布）
build_prod() {
    echo "Building in production mode..."
    maturin build --release
}

# 运行测试
run_tests() {
    echo "Running tests..."
    python -m pytest tests/ -v
}

# 基准测试
run_benchmarks() {
    echo "Running benchmarks..."
    python benchmarks/benchmark.py
}

# 解析命令行参数
case "$1" in
    dev)
        build_dev
        ;;
    prod)
        build_prod
        ;;
    test)
        build_dev
        run_tests
        ;;
    bench)
        build_dev
        run_benchmarks
        ;;
    all)
        build_dev
        run_tests
        run_benchmarks
        ;;
    *)
        echo "Usage: $0 {dev|prod|test|bench|all}"
        echo "  dev   - Build in development mode"
        echo "  prod  - Build in production mode"
        echo "  test  - Build and run tests"
        echo "  bench - Build and run benchmarks"
        echo "  all   - Build, test, and benchmark"
        exit 1
        ;;
esac

echo "Done!"