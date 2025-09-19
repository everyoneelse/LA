#!/bin/bash

# 小规模Pipeline验证实验运行脚本
# 作者: Assistant
# 日期: $(date)

set -e  # 出错时退出

echo "🚀 开始小规模Pipeline验证实验..."
echo "=================================================="

# 检查Python环境
echo "📋 检查环境依赖..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
    echo "❌ PyTorch未安装，请先安装PyTorch"
    exit 1
}

python3 -c "import transformers; print(f'Transformers版本: {transformers.__version__}')" || {
    echo "❌ Transformers未安装，请运行: pip install transformers"
    exit 1
}

# 创建必要目录
echo "📁 创建工作目录..."
mkdir -p pilot_results
mkdir -p pilot_data
mkdir -p logs

# 设置日志文件
LOG_FILE="logs/pilot_experiment_$(date +%Y%m%d_%H%M%S).log"

echo "📝 日志文件: $LOG_FILE"
echo "=================================================="

# 运行实验
echo "🧪 运行小规模验证实验..."
python3 pilot_experiment.py 2>&1 | tee "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ 实验运行成功!"
    
    # 运行数据分析
    echo "📊 开始数据分析..."
    python3 data_analysis.py 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ 分析完成!"
        echo ""
        echo "📄 生成的文件:"
        echo "  - 实验报告: pilot_results/experiment_report.json"
        echo "  - 训练曲线: pilot_results/training_curves.png"
        echo "  - 数据分布: pilot_results/data_distribution.png"
        echo "  - 总结报告: pilot_results/summary_report.md"
        echo "  - 完整日志: $LOG_FILE"
        echo ""
        echo "🎉 小规模验证实验全部完成!"
        echo "现在可以基于这些结果优化参数，然后扩展到全量数据。"
    else
        echo "❌ 数据分析失败，请检查日志"
        exit 1
    fi
else
    echo "❌ 实验失败，请检查日志: $LOG_FILE"
    exit 1
fi

echo "=================================================="
echo "下一步建议:"
echo "1. 查看 pilot_results/summary_report.md 了解关键发现"
echo "2. 根据训练曲线调整学习率和批次大小"
echo "3. 基于数据分布优化领域配比"
echo "4. 准备扩展到24B全量数据"
echo "=================================================="