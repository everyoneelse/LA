#!/bin/bash

# Rust Tokenizer 自动安装脚本
# 支持多种安装方式，自动检测环境

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    echo $OS
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查 Rust 是否已安装
check_rust() {
    if command_exists rustc && command_exists cargo; then
        RUST_VERSION=$(rustc --version | cut -d' ' -f2)
        print_success "Rust 已安装 (版本: $RUST_VERSION)"
        return 0
    else
        return 1
    fi
}

# 使用 Conda 安装 Rust
install_rust_conda() {
    print_info "使用 Conda 安装 Rust..."
    
    if ! command_exists conda; then
        print_error "未找到 Conda，请先安装 Anaconda 或 Miniconda"
        return 1
    fi
    
    # 检查是否在 conda 环境中
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        print_info "创建新的 Conda 环境..."
        conda create -n rust-tokenizer python=3.10 -y
        eval "$(conda shell.bash hook)"
        conda activate rust-tokenizer
    fi
    
    print_info "安装 Rust 和相关工具..."
    conda install -c conda-forge rust maturin -y
    
    if check_rust; then
        print_success "Rust 通过 Conda 安装成功"
        return 0
    else
        print_error "Rust 安装失败"
        return 1
    fi
}

# 使用 rustup 安装 Rust
install_rust_rustup() {
    print_info "使用 rustup 官方安装器安装 Rust..."
    
    if [[ $(detect_os) == "windows" ]]; then
        print_info "请访问 https://rustup.rs 下载 Windows 安装器"
        return 1
    else
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi
    
    if check_rust; then
        print_success "Rust 通过 rustup 安装成功"
        return 0
    else
        print_error "Rust 安装失败"
        return 1
    fi
}

# 使用系统包管理器安装 Rust
install_rust_system() {
    local OS=$(detect_os)
    print_info "使用系统包管理器安装 Rust..."
    
    case $OS in
        linux)
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y rustc cargo build-essential pkg-config libssl-dev
            elif command_exists dnf; then
                sudo dnf install -y rust cargo gcc openssl-devel
            elif command_exists pacman; then
                sudo pacman -S --noconfirm rust
            else
                print_error "不支持的 Linux 发行版"
                return 1
            fi
            ;;
        macos)
            if command_exists brew; then
                brew install rust
            else
                print_error "请先安装 Homebrew"
                return 1
            fi
            ;;
        *)
            print_error "不支持的操作系统"
            return 1
            ;;
    esac
    
    if check_rust; then
        print_success "Rust 通过系统包管理器安装成功"
        return 0
    else
        print_error "Rust 安装失败"
        return 1
    fi
}

# 安装 Python 依赖
install_python_deps() {
    print_info "安装 Python 依赖..."
    
    # 升级 pip
    pip install --upgrade pip
    
    # 安装必需的包
    pip install maturin pytest pytest-benchmark
    
    # 可选包（用于对比测试）
    pip install transformers sentencepiece tokenizers 2>/dev/null || true
    
    print_success "Python 依赖安装完成"
}

# 编译 Rust tokenizer
build_tokenizer() {
    print_info "编译 Rust tokenizer..."
    
    cd /workspace/rust_tokenizer
    
    # 选择编译模式
    echo "选择编译模式:"
    echo "1) Debug (快速编译，用于开发)"
    echo "2) Release (优化编译，用于生产)"
    read -p "请选择 [1-2]: " choice
    
    case $choice in
        1)
            print_info "使用 Debug 模式编译..."
            maturin develop
            ;;
        2)
            print_info "使用 Release 模式编译..."
            maturin develop --release
            ;;
        *)
            print_info "默认使用 Release 模式..."
            maturin develop --release
            ;;
    esac
    
    print_success "编译完成"
}

# 验证安装
verify_installation() {
    print_info "验证安装..."
    
    python -c "
try:
    from rust_tokenizer import RustTokenizerWrapper
    print('✅ Rust tokenizer 导入成功')
    print('✅ 安装验证通过')
except ImportError as e:
    print(f'❌ 导入失败: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "安装验证成功！"
        return 0
    else
        print_error "安装验证失败"
        return 1
    fi
}

# 显示安装信息
show_info() {
    echo ""
    echo "======================================"
    echo "   Rust Tokenizer 安装完成"
    echo "======================================"
    echo ""
    echo "使用方法:"
    echo "  from rust_tokenizer import RustTokenizerWrapper"
    echo "  tokenizer = RustTokenizerWrapper('path/to/model')"
    echo ""
    echo "运行示例:"
    echo "  python example.py"
    echo ""
    echo "运行测试:"
    echo "  pytest tests/"
    echo ""
    echo "运行基准测试:"
    echo "  python benchmarks/benchmark.py"
    echo ""
}

# 主安装流程
main() {
    echo "======================================"
    echo "   Rust Tokenizer 自动安装脚本"
    echo "======================================"
    echo ""
    
    # 检测操作系统
    OS=$(detect_os)
    print_info "检测到操作系统: $OS"
    
    # 检查或安装 Rust
    if ! check_rust; then
        echo ""
        echo "选择 Rust 安装方式:"
        echo "1) 使用 Conda (推荐)"
        echo "2) 使用 rustup 官方安装器"
        echo "3) 使用系统包管理器"
        echo "4) 跳过 (已手动安装)"
        read -p "请选择 [1-4]: " rust_choice
        
        case $rust_choice in
            1)
                install_rust_conda || install_rust_rustup
                ;;
            2)
                install_rust_rustup
                ;;
            3)
                install_rust_system
                ;;
            4)
                print_info "跳过 Rust 安装"
                ;;
            *)
                print_info "默认使用 Conda 安装"
                install_rust_conda || install_rust_rustup
                ;;
        esac
    fi
    
    # 再次检查 Rust
    if ! check_rust; then
        print_error "Rust 未正确安装，请手动安装后重试"
        exit 1
    fi
    
    # 安装 Python 依赖
    install_python_deps
    
    # 编译 tokenizer
    build_tokenizer
    
    # 验证安装
    if verify_installation; then
        show_info
    else
        print_error "安装过程中出现错误，请检查错误信息"
        exit 1
    fi
}

# 运行主函数
main