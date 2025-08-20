# Rust Tokenizer 安装与编译指南

## 一、安装 Rust 依赖

### 方法 1：使用 Conda 安装（推荐）

Conda 提供了方便的 Rust 工具链管理，特别适合在科学计算环境中使用。

```bash
# 1. 创建新的 conda 环境（可选）
conda create -n rust-tokenizer python=3.10
conda activate rust-tokenizer

# 2. 安装 Rust 工具链
conda install -c conda-forge rust

# 3. 验证安装
rustc --version
cargo --version

# 4. 安装 maturin（Python-Rust 绑定构建工具）
conda install -c conda-forge maturin

# 或者使用 pip
pip install maturin
```

#### 使用 conda-forge 的优势：
- ✅ 自动处理依赖关系
- ✅ 与 conda 环境完美集成
- ✅ 便于版本管理和环境隔离
- ✅ 跨平台一致性

### 方法 2：使用 rustup 官方安装器

如果您需要最新版本或更多控制，可以使用官方安装器：

```bash
# Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 按照提示完成安装，通常选择默认选项即可
# 安装完成后，重新加载环境变量
source $HOME/.cargo/env

# Windows (PowerShell)
# 下载并运行: https://win.rustup.rs/
```

### 方法 3：使用系统包管理器

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install rustc cargo

# Fedora
sudo dnf install rust cargo

# macOS (使用 Homebrew)
brew install rust

# Arch Linux
sudo pacman -S rust
```

## 二、安装 Python 依赖

```bash
# 基础依赖
pip install maturin pytest pytest-benchmark

# 可选：用于对比测试
pip install transformers sentencepiece tokenizers

# 开发依赖
pip install black isort mypy
```

## 三、Rust 编译说明

### 3.1 编译模式

Rust 提供两种主要编译模式：

#### Debug 模式（开发）
```bash
# 快速编译，包含调试信息，未优化
cargo build
maturin develop
```
- 编译速度：快（~10秒）
- 运行性能：慢
- 文件大小：大
- 调试信息：完整
- 适用场景：开发、调试

#### Release 模式（生产）
```bash
# 优化编译，性能最佳
cargo build --release
maturin develop --release
```
- 编译速度：慢（~1-2分钟）
- 运行性能：快（10-100x 提升）
- 文件大小：小
- 调试信息：最少
- 适用场景：生产、基准测试

### 3.2 编译优化选项

在 `Cargo.toml` 中配置优化选项：

```toml
# 开发模式优化（更快的开发体验）
[profile.dev]
opt-level = 1  # 基础优化，编译更快

# 发布模式优化（最佳性能）
[profile.release]
opt-level = 3      # 最高优化级别
lto = true         # 链接时优化
codegen-units = 1  # 单代码生成单元，更好的优化
strip = true       # 移除符号信息，减小体积
panic = "abort"    # 使用 abort 而非 unwind，性能更好

# 基准测试专用配置
[profile.bench]
inherits = "release"
debug = true  # 保留调试信息用于性能分析
```

### 3.3 目标架构编译

```bash
# 查看支持的目标架构
rustup target list

# 添加目标架构
rustup target add x86_64-unknown-linux-gnu
rustup target add aarch64-unknown-linux-gnu  # ARM64

# 交叉编译
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target aarch64-apple-darwin  # Apple Silicon
```

### 3.4 特性标志（Features）

在 `Cargo.toml` 中定义特性：

```toml
[features]
default = ["parallel"]
parallel = ["rayon"]  # 并行处理
simd = []            # SIMD 优化
cuda = []            # CUDA 支持（需要额外配置）
```

编译时启用特性：
```bash
# 启用特定特性
cargo build --release --features "parallel,simd"

# 禁用默认特性
cargo build --release --no-default-features

# 启用所有特性
cargo build --release --all-features
```

## 四、完整编译流程

### 4.1 首次编译

```bash
# 1. 进入项目目录
cd /workspace/rust_tokenizer

# 2. 安装 Rust（如果还没安装）
conda install -c conda-forge rust maturin

# 3. 编译并安装到当前 Python 环境
maturin develop --release

# 4. 验证安装
python -c "from rust_tokenizer import RustTokenizerWrapper; print('Success!')"
```

### 4.2 使用提供的构建脚本

```bash
# 开发构建（快速）
./build.sh dev

# 生产构建（优化）
./build.sh prod

# 构建并测试
./build.sh test

# 构建并运行基准测试
./build.sh bench
```

### 4.3 构建 wheel 包

```bash
# 构建当前平台的 wheel
maturin build --release

# 构建通用 wheel（manylinux）
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release

# 输出位置：target/wheels/
```

## 五、环境变量配置

### 5.1 Rust 编译环境变量

```bash
# 设置 Rust 编译器优化
export RUSTFLAGS="-C target-cpu=native"  # 针对当前 CPU 优化

# 设置并行编译
export CARGO_BUILD_JOBS=8  # 使用 8 个线程编译

# 设置缓存目录
export CARGO_HOME="$HOME/.cargo"
export CARGO_TARGET_DIR="/tmp/rust-target"  # 使用 SSD 加速

# 启用增量编译
export CARGO_INCREMENTAL=1

# 调试信息
export RUST_BACKTRACE=1  # 显示完整错误栈
export RUST_LOG=debug     # 显示调试日志
```

### 5.2 Conda 环境配置

```bash
# 创建环境配置文件
cat > environment.yml << EOF
name: rust-tokenizer
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - rust=1.75
  - maturin>=1.4
  - pip
  - pytest
  - numpy
  - pip:
    - tokenizers
    - transformers
    - sentencepiece
EOF

# 使用配置文件创建环境
conda env create -f environment.yml
conda activate rust-tokenizer
```

## 六、常见问题解决

### 6.1 编译错误

```bash
# 清理编译缓存
cargo clean

# 更新依赖
cargo update

# 检查依赖树
cargo tree

# 详细编译输出
RUST_BACKTRACE=full cargo build --verbose
```

### 6.2 链接错误

```bash
# Linux: 安装开发工具
sudo apt-get install build-essential pkg-config libssl-dev

# macOS: 安装 Xcode 命令行工具
xcode-select --install

# 设置链接器（Linux）
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc
```

### 6.3 性能调试

```bash
# 生成性能分析数据
cargo build --release
perf record --call-graph=dwarf ./target/release/your_binary
perf report

# 使用 flamegraph
cargo install flamegraph
cargo flamegraph
```

## 七、Docker 构建

```dockerfile
# Dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY . .

# 构建 Release 版本
RUN cargo build --release

# 多阶段构建，减小镜像体积
FROM python:3.10-slim

COPY --from=builder /app/target/release/librust_tokenizer.so /usr/local/lib/
COPY python/ /app/python/

WORKDIR /app
RUN pip install maturin
RUN maturin develop --release

CMD ["python", "-c", "from rust_tokenizer import RustTokenizerWrapper"]
```

构建 Docker 镜像：
```bash
docker build -t rust-tokenizer .
docker run -it rust-tokenizer
```

## 八、CI/CD 集成

### GitHub Actions 示例

```yaml
# .github/workflows/build.yml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true
    
    - name: Install dependencies
      run: |
        pip install maturin pytest
    
    - name: Build
      run: maturin develop --release
    
    - name: Test
      run: pytest tests/
```

## 九、性能优化建议

1. **使用 CPU 特定优化**
   ```bash
   RUSTFLAGS="-C target-cpu=native" maturin develop --release
   ```

2. **启用 LTO（链接时优化）**
   已在 Cargo.toml 中配置

3. **使用 PGO（配置文件引导优化）**
   ```bash
   # 步骤 1: 构建用于收集配置文件的版本
   RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release
   
   # 步骤 2: 运行典型工作负载
   ./run_benchmarks.sh
   
   # 步骤 3: 使用配置文件重新构建
   RUSTFLAGS="-Cprofile-use=/tmp/pgo-data" cargo build --release
   ```

4. **内存分配器优化**
   ```toml
   # Cargo.toml
   [dependencies]
   mimalloc = { version = "*", default-features = false }
   ```

## 十、验证安装

创建测试脚本 `test_installation.py`：

```python
#!/usr/bin/env python3

def test_rust_tokenizer():
    try:
        from rust_tokenizer import RustTokenizerWrapper
        print("✅ Rust tokenizer 导入成功")
        
        # 测试基本功能
        print("✅ 所有功能正常")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if test_rust_tokenizer() else 1)
```

运行验证：
```bash
python test_installation.py
```

---

**提示**: 
- 对于生产环境，始终使用 `--release` 标志
- 定期更新 Rust 工具链：`rustup update`
- 使用 `cargo audit` 检查安全漏洞
- 考虑使用 `sccache` 加速重复编译