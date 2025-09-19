#!/bin/bash

# 数据采样和打包脚本
# 用于处理24B新闻、代码、数学数据

set -e  # 出错时退出

echo "🚀 开始大规模数据采样和打包..."
echo "=================================================="

# 检查Python环境
echo "📋 检查环境依赖..."
python3 -c "import numpy, tqdm; print('✅ 基础依赖检查通过')" || {
    echo "❌ 缺少必要依赖，请运行: pip install numpy tqdm"
    exit 1
}

# 检查数据目录
echo "📁 检查数据目录..."
DATA_DIRS=("./raw_data/news/" "./raw_data/code/" "./raw_data/math/")

for dir in "${DATA_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "⚠️  数据目录不存在: $dir"
        echo "请创建目录并放入相应数据文件"
        mkdir -p "$dir"
        echo "已创建目录: $dir"
    else
        file_count=$(find "$dir" -type f \( -name "*.txt" -o -name "*.json" -o -name "*.jsonl" \) | wc -l)
        echo "✅ $dir: 发现 $file_count 个数据文件"
    fi
done

# 创建输出目录
echo "📁 创建输出目录..."
mkdir -p packed_data
mkdir -p logs

# 设置日志文件
LOG_FILE="logs/data_sampling_$(date +%Y%m%d_%H%M%S).log"
echo "📝 日志文件: $LOG_FILE"

# 检查可用内存
echo "💾 检查系统资源..."
AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
echo "可用内存: ${AVAILABLE_MEM}GB"

if (( $(echo "$AVAILABLE_MEM < 8" | bc -l) )); then
    echo "⚠️  可用内存较少，建议增加swap或使用流式处理"
fi

# 检查磁盘空间
AVAILABLE_DISK=$(df -h . | awk 'NR==2 {print $4}')
echo "可用磁盘空间: $AVAILABLE_DISK"

echo "=================================================="

# 运行数据采样
echo "🔄 开始数据采样和打包..."
echo "这可能需要较长时间，请耐心等待..."

# 使用不同的配置选项
SAMPLING_MODE=${1:-"small"}  # small, medium, large

case $SAMPLING_MODE in
    "small")
        echo "📊 运行小规模测试 (100K samples)"
        python3 -c "
from data_sampler import SamplingConfig, DataSamplingPipeline
import multiprocessing as mp

config = SamplingConfig(
    news_data_path='./raw_data/news/',
    code_data_path='./raw_data/code/',
    math_data_path='./raw_data/math/',
    output_path='./packed_data/',
    total_samples=100_000,
    sequence_length=1024,
    samples_per_file=5000,
    num_workers=min(4, mp.cpu_count())
)

pipeline = DataSamplingPipeline(config)
pipeline.run()
" 2>&1 | tee "$LOG_FILE"
        ;;
    
    "medium")
        echo "📊 运行中等规模处理 (1M samples)"
        python3 -c "
from data_sampler import SamplingConfig, DataSamplingPipeline
import multiprocessing as mp

config = SamplingConfig(
    news_data_path='./raw_data/news/',
    code_data_path='./raw_data/code/',
    math_data_path='./raw_data/math/',
    output_path='./packed_data/',
    total_samples=1_000_000,
    sequence_length=2048,
    samples_per_file=10_000,
    num_workers=min(8, mp.cpu_count())
)

pipeline = DataSamplingPipeline(config)
pipeline.run()
" 2>&1 | tee "$LOG_FILE"
        ;;
    
    "large")
        echo "📊 运行大规模处理 (10M+ samples)"
        python3 -c "
from data_sampler import SamplingConfig, DataSamplingPipeline
import multiprocessing as mp

config = SamplingConfig(
    news_data_path='./raw_data/news/',
    code_data_path='./raw_data/code/',
    math_data_path='./raw_data/math/',
    output_path='./packed_data/',
    total_samples=10_000_000,
    sequence_length=2048,
    samples_per_file=20_000,
    num_workers=mp.cpu_count(),
    chunk_size=2000
)

pipeline = DataSamplingPipeline(config)
pipeline.run()
" 2>&1 | tee "$LOG_FILE"
        ;;
    
    *)
        echo "❌ 未知模式: $SAMPLING_MODE"
        echo "请使用: small, medium, large"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo "✅ 数据采样和打包成功完成!"
    
    # 显示结果统计
    echo ""
    echo "📊 处理结果:"
    echo "----------------------------------------"
    
    if [ -f "packed_data/metadata.json" ]; then
        echo "📄 元数据信息:"
        python3 -c "
import json
with open('packed_data/metadata.json', 'r') as f:
    meta = json.load(f)
print(f'  总序列数: {meta[\"total_sequences\"]:,}')
print(f'  文件数: {meta[\"num_files\"]}')
print(f'  序列长度: {meta[\"sequence_length\"]}')
print(f'  每文件样本数: {meta[\"samples_per_file\"]:,}')
"
    fi
    
    if [ -f "packed_data/sampling_report.json" ]; then
        echo "📈 采样统计:"
        python3 -c "
import json
with open('packed_data/sampling_report.json', 'r') as f:
    report = json.load(f)

domain_stats = report['domain_statistics']
for domain, stats in domain_stats.items():
    print(f'  {domain}: {stats[\"count\"]:,} 样本, 平均长度: {stats[\"avg_length\"]:.1f}')

packed_stats = report['packed_statistics']
print(f'  总字符数: {packed_stats[\"total_characters\"]:,}')
print(f'  平均序列长度: {packed_stats[\"avg_sequence_length\"]:.1f}')
"
    fi
    
    # 显示文件大小
    echo "💾 输出文件:"
    du -sh packed_data/*
    
    echo ""
    echo "🎉 数据处理完成!"
    echo "现在可以使用 packed_data_loader.py 加载数据进行训练"
    
else
    echo "❌ 数据采样失败，请检查日志: $LOG_FILE"
    echo ""
    echo "常见问题排查:"
    echo "1. 检查数据目录是否存在且包含文件"
    echo "2. 检查磁盘空间是否充足"
    echo "3. 检查内存是否足够"
    echo "4. 查看详细错误信息: tail -50 $LOG_FILE"
    exit 1
fi

echo "=================================================="
echo "下一步建议:"
echo "1. 使用 python3 packed_data_loader.py 测试数据加载"
echo "2. 检查 packed_data/sampling_report.json 了解数据分布"
echo "3. 根据需要调整采样参数重新运行"
echo "4. 开始模型训练"
echo "=================================================="