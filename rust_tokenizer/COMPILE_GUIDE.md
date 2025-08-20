# Rust Tokenizer 编译详细指南

## 目录
1. [编译基础概念](#编译基础概念)
2. [编译器配置](#编译器配置)
3. [优化策略](#优化策略)
4. [交叉编译](#交叉编译)
5. [故障排除](#故障排除)
6. [性能调优](#性能调优)

## 编译基础概念

### Rust 编译流程

```mermaid
graph LR
    A[Rust 源码] --> B[词法分析]
    B --> C[语法分析]
    C --> D[语义分析]
    D --> E[MIR 生成]
    E --> F[优化]
    F --> G[LLVM IR]
    G --> H[机器码]
    H --> I[链接]
    I --> J[最终二进制]
```

### 编译产物说明

```
target/
├── debug/          # Debug 模式产物
│   ├── deps/       # 依赖库
│   ├── build/      # 构建脚本输出
│   └── librust_tokenizer.so  # 动态库
├── release/        # Release 模式产物
│   └── librust_tokenizer.so  # 优化后的动态库
└── wheels/         # Python wheel 包
```

## 编译器配置

### 1. Cargo.toml 完整配置

```toml
[package]
name = "rust_tokenizer"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <email@example.com>"]
license = "MIT"
description = "High-performance tokenizer in Rust"

[lib]
name = "rust_tokenizer"
# cdylib: C动态库，用于 Python 绑定
# rlib: Rust 库，用于 Rust 项目
crate-type = ["cdylib", "rlib"]

# 基础依赖
[dependencies]
tokenizers = { version = "0.19", features = ["http", "progressbar"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
thiserror = "1.0"

# Python 绑定
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py38"] }

# 性能优化
rayon = { version = "1.10", optional = true }
mimalloc = { version = "0.1", optional = true, default-features = false }
parking_lot = { version = "0.12", optional = true }

# SIMD 优化
packed_simd = { version = "0.3", optional = true }

# 日志
log = "0.4"
env_logger = "0.11"

# 开发依赖
[dev-dependencies]
criterion = "0.5"  # 基准测试
proptest = "1.0"   # 属性测试
quickcheck = "1.0" # 快速检查

# 特性标志
[features]
default = ["parallel", "mimalloc"]
parallel = ["rayon"]
simd = ["packed_simd"]
python = ["pyo3"]
mimalloc = ["dep:mimalloc"]
parking_lot = ["dep:parking_lot"]

# 编译配置文件
[profile.dev]
opt-level = 1       # 基础优化
debug = true        # 调试信息
debug-assertions = true
overflow-checks = true
incremental = true

[profile.release]
opt-level = 3       # 最大优化
lto = "fat"        # 完整的链接时优化
codegen-units = 1   # 单代码生成单元
strip = true        # 移除符号
debug = false
panic = "abort"     # panic 时直接终止
overflow-checks = false

# 针对大小优化的配置
[profile.release-small]
inherits = "release"
opt-level = "z"     # 优化大小
lto = true
codegen-units = 1
strip = true

# 基准测试配置
[profile.bench]
inherits = "release"
debug = true        # 保留调试信息用于分析
lto = false        # 加快编译速度

# 自定义性能配置
[profile.performance]
inherits = "release"
lto = "fat"
codegen-units = 1
opt-level = 3
strip = false       # 保留符号用于性能分析
```

### 2. 编译器环境变量

```bash
# 基础配置
export RUST_BACKTRACE=1           # 启用回溯
export RUST_LOG=debug              # 日志级别

# 编译优化
export RUSTFLAGS="-C target-cpu=native -C link-arg=-fuse-ld=lld"
# -C target-cpu=native: 针对当前 CPU 优化
# -C link-arg=-fuse-ld=lld: 使用 LLVM 链接器（更快）

# 针对特定架构优化
export RUSTFLAGS="-C target-feature=+avx2,+fma"  # Intel AVX2
export RUSTFLAGS="-C target-feature=+neon"       # ARM NEON

# 增量编译
export CARGO_INCREMENTAL=1

# 并行编译
export CARGO_BUILD_JOBS=8         # 使用 8 个线程
export RAYON_NUM_THREADS=4        # Rayon 运行时线程数

# 缓存配置
export CARGO_HOME="$HOME/.cargo"
export CARGO_TARGET_DIR="/tmp/rust-target"  # 使用 RAM disk 加速
```

### 3. .cargo/config.toml 配置

```toml
# .cargo/config.toml
[build]
jobs = 8                    # 并行任务数
target-dir = "target"       # 目标目录
incremental = true          # 增量编译
rustflags = ["-C", "target-cpu=native"]

[target.x86_64-unknown-linux-gnu]
linker = "clang"           # 使用 Clang 链接器
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

[target.x86_64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]

[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]

# 注册表镜像（中国大陆加速）
[source.crates-io]
replace-with = 'tuna'

[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"

# 别名
[alias]
b = "build --release"
t = "test --release"
br = "bench"
```

## 优化策略

### 1. 编译时优化

```bash
# 基础 Release 编译
cargo build --release

# 使用所有 CPU 核心
cargo build --release -j $(nproc)

# 针对当前 CPU 架构优化
RUSTFLAGS="-C target-cpu=native" cargo build --release

# 启用 AVX2 指令集（Intel/AMD）
RUSTFLAGS="-C target-feature=+avx2" cargo build --release

# 使用 Thin LTO（更快的编译，稍差的优化）
cargo build --release --config profile.release.lto='"thin"'

# 使用 PGO（配置文件引导优化）
./pgo_build.sh  # 见下方脚本
```

### 2. PGO 优化脚本

```bash
#!/bin/bash
# pgo_build.sh - 配置文件引导优化构建

# 步骤 1: 清理之前的构建
cargo clean

# 步骤 2: 构建用于收集配置文件的版本
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" \
    cargo build --release

# 步骤 3: 运行典型工作负载收集数据
echo "收集性能数据..."
python benchmarks/benchmark.py --num-texts 10000

# 步骤 4: 合并配置文件
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# 步骤 5: 使用配置文件重新构建
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" \
    cargo build --release

echo "PGO 优化构建完成！"
```

### 3. 内存分配器优化

```rust
// src/lib.rs
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// 或使用 jemalloc
#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
```

## 交叉编译

### 1. 安装目标架构

```bash
# 列出所有可用目标
rustup target list

# 安装常用目标
rustup target add x86_64-unknown-linux-gnu      # Linux x64
rustup target add aarch64-unknown-linux-gnu     # Linux ARM64
rustup target add x86_64-apple-darwin           # macOS x64
rustup target add aarch64-apple-darwin          # macOS ARM64 (M1/M2)
rustup target add x86_64-pc-windows-gnu         # Windows x64
rustup target add wasm32-unknown-unknown        # WebAssembly
```

### 2. 交叉编译示例

```bash
# Linux x64 -> Linux ARM64
cargo build --release --target aarch64-unknown-linux-gnu

# macOS -> Linux
cargo build --release --target x86_64-unknown-linux-gnu

# 使用 Docker 进行交叉编译
docker run --rm -v "$(pwd)":/workspace \
    -w /workspace \
    messense/rust-musl-cross:x86_64-musl \
    cargo build --release --target x86_64-unknown-linux-musl
```

### 3. 多平台 wheel 构建

```bash
# 使用 maturin 构建多平台 wheel
# Linux (manylinux)
docker run --rm -v $(pwd):/io \
    ghcr.io/pyo3/maturin build --release \
    --compatibility manylinux2014

# macOS universal2 (Intel + Apple Silicon)
maturin build --release --universal2

# Windows
maturin build --release --target x86_64-pc-windows-msvc
```

## 故障排除

### 常见编译错误及解决方案

#### 1. 链接错误

```bash
# 错误: cannot find -lpython3.10
# 解决: 安装 Python 开发包
sudo apt-get install python3.10-dev  # Ubuntu/Debian
conda install python=3.10            # Conda

# 错误: undefined reference to `__cxa_thread_atexit_impl'
# 解决: 更新 gcc/g++
sudo apt-get update && sudo apt-get install gcc g++
```

#### 2. 内存不足

```bash
# 错误: signal: 9, SIGKILL: kill
# 解决: 减少并行度或增加交换空间
cargo build --release -j 2  # 只用 2 个线程

# 增加交换空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. 依赖冲突

```bash
# 清理缓存
cargo clean
rm -rf ~/.cargo/registry/cache
rm -rf ~/.cargo/git

# 更新依赖
cargo update

# 查看依赖树
cargo tree -d  # 显示重复依赖
```

## 性能调优

### 1. 编译时间优化

```toml
# Cargo.toml
[profile.dev]
opt-level = 1  # 轻度优化，编译更快
split-debuginfo = "unpacked"  # macOS: 更快的调试信息生成

[profile.dev.package."*"]
opt-level = 3  # 依赖使用最大优化
```

### 2. 二进制大小优化

```toml
# Cargo.toml
[profile.release]
opt-level = "z"     # 优化大小
lto = true          # 链接时优化
codegen-units = 1   # 单代码单元
strip = true        # 移除符号
panic = "abort"     # 更小的 panic 处理
```

### 3. 运行时性能优化

```rust
// 使用 #[inline] 提示
#[inline(always)]
fn hot_function() { }

// 使用 likely/unlikely 分支预测
use std::intrinsics::{likely, unlikely};

if likely(condition) {
    // 常见路径
} else {
    // 罕见路径
}

// 避免边界检查
unsafe {
    *array.get_unchecked(index)
}
```

### 4. 基准测试

```bash
# 运行基准测试
cargo bench

# 使用 criterion 进行详细分析
cargo bench -- --save-baseline before
# 进行修改...
cargo bench -- --baseline before

# 生成火焰图
cargo install flamegraph
cargo flamegraph --bench my_benchmark
```

## 持续集成配置

### GitHub Actions

```yaml
# .github/workflows/rust.yml
name: Rust CI

on: [push, pull_request]

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta, nightly]
    
    runs-on: ${{ matrix.os }}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Cache cargo
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Install Rust
      uses: dtolnay/rust-toolchain@master
      with:
        toolchain: ${{ matrix.rust }}
    
    - name: Build
      run: cargo build --release --all-features
    
    - name: Test
      run: cargo test --release --all-features
    
    - name: Benchmark
      run: cargo bench --no-run
```

## 总结

1. **开发阶段**: 使用 `cargo build` 或 `maturin develop`
2. **测试阶段**: 使用 `cargo build --release` 进行性能测试
3. **生产部署**: 使用完整优化 + PGO + 合适的内存分配器
4. **分发**: 使用 `maturin build --release` 生成 wheel

记住：
- 始终在 Release 模式下进行性能测试
- 使用 `target-cpu=native` 获得最佳本地性能
- 考虑二进制大小 vs 性能的权衡
- 定期更新依赖以获得最新优化