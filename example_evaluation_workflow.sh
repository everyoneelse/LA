#!/bin/bash
# 999M模型评估工作流程示例
# 这个脚本展示了一个完整的评估流程

set -e  # 遇到错误立即退出

echo "================================================"
echo "  999M模型完整评估工作流程"
echo "================================================"
echo ""

# ============================================
# 第一部分：环境准备
# ============================================
echo "📋 第一步：环境准备"
echo "================================================"

# 请根据你的实际情况设置这些路径
MODEL_PATH="${MODEL_PATH:-/path/to/your/999m/model}"
LLAMA_CONFIG="${LLAMA_CONFIG:-/path/to/params.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/tokenizer.model}"

echo "模型路径: $MODEL_PATH"
echo "配置文件: $LLAMA_CONFIG"
echo "分词器: $TOKENIZER_PATH"
echo ""

# 检查路径是否有效
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    echo ""
    echo "请设置正确的路径："
    echo "  export MODEL_PATH=/your/actual/model/path"
    echo "  export LLAMA_CONFIG=/your/actual/config/path"
    echo "  export TOKENIZER_PATH=/your/actual/tokenizer/path"
    exit 1
fi

# 创建结果目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="evaluation_workflow_${TIMESTAMP}"
mkdir -p "$WORK_DIR"

echo "✅ 工作目录已创建: $WORK_DIR"
echo ""

# ============================================
# 第二部分：快速诊断测试
# ============================================
echo "📊 第二步：快速诊断测试 (约5分钟)"
echo "================================================"
echo "这将快速测试模型的基本功能..."
echo ""

python text_completion_test.py \
    --pretrained_path "$MODEL_PATH" \
    --llama_config "$LLAMA_CONFIG" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --max_gen_len 100 \
    --dtype bf16 \
    <<EOF
解释什么是深度学习
quit
EOF

echo "✅ 快速诊断完成"
echo ""

# ============================================
# 第三部分：困惑度评估（最重要的基础指标）
# ============================================
echo "📈 第三步：困惑度评估 (约10分钟)"
echo "================================================"
echo "困惑度是衡量语言模型质量的基础指标..."
echo ""

python comprehensive_model_evaluation.py \
    --pretrained_path "$MODEL_PATH" \
    --llama_config "$LLAMA_CONFIG" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --eval_types perplexity \
    --output_dir "$WORK_DIR/perplexity_test" \
    --dtype bf16

echo "✅ 困惑度评估完成"
echo ""

# 读取困惑度结果
PERPLEXITY_RESULT=$(ls -t "$WORK_DIR"/perplexity_test/eval_*/perplexity.json | head -1)
if [ -f "$PERPLEXITY_RESULT" ]; then
    echo "困惑度结果:"
    cat "$PERPLEXITY_RESULT" | grep "average_perplexity" || echo "无法读取结果"
    echo ""
fi

# ============================================
# 第四部分：基础能力全面评估
# ============================================
echo "🎯 第四步：基础能力全面评估 (约30分钟)"
echo "================================================"
echo "测试模型在各种任务上的表现..."
echo ""

python comprehensive_model_evaluation.py \
    --pretrained_path "$MODEL_PATH" \
    --llama_config "$LLAMA_CONFIG" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --eval_types basic quality speed \
    --output_dir "$WORK_DIR/comprehensive_eval" \
    --dtype bf16

echo "✅ 基础能力评估完成"
echo ""

# ============================================
# 第五部分：性能优化测试（可选）
# ============================================
echo "⚡ 第五步：性能优化测试 (可选)"
echo "================================================"
read -p "是否测试量化性能？(y/n): " test_quant

if [ "$test_quant" = "y" ]; then
    echo "测试4bit量化推理性能..."
    echo ""
    
    python comprehensive_model_evaluation.py \
        --pretrained_path "$MODEL_PATH" \
        --llama_config "$LLAMA_CONFIG" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --eval_types speed \
        --output_dir "$WORK_DIR/quant_test" \
        --quant \
        --dtype bf16
    
    echo "✅ 量化测试完成"
else
    echo "⏭️ 跳过量化测试"
fi
echo ""

# ============================================
# 第六部分：标准Benchmark测试（可选）
# ============================================
echo "📚 第六步：标准Benchmark测试 (可选)"
echo "================================================"
echo "标准benchmark测试通常需要2-4小时"
echo "可选的测试："
echo "  1. GSM8K - 数学推理能力"
echo "  2. MMLU - 多任务知识理解"
echo "  3. C-Eval - 中文综合能力"
echo "  4. 跳过"
echo ""
read -p "请选择 (1-4): " benchmark_choice

case $benchmark_choice in
    1|2|3)
        echo "准备运行benchmark测试..."
        echo "注意：这可能需要几个小时"
        read -p "确认继续？(y/n): " confirm
        
        if [ "$confirm" = "y" ]; then
            export MODEL_PATH="$MODEL_PATH"
            export LLAMA_CONFIG="$LLAMA_CONFIG"
            export TOKENIZER_PATH="$TOKENIZER_PATH"
            
            # 这里会调用benchmark测试脚本
            echo "提示: 使用以下命令手动运行benchmark:"
            echo "bash run_benchmark_suite.sh"
        fi
        ;;
    4)
        echo "⏭️ 跳过benchmark测试"
        ;;
esac
echo ""

# ============================================
# 第七部分：生成评估报告
# ============================================
echo "📋 第七步：生成评估报告"
echo "================================================"

# 汇总所有评估结果
SUMMARY_FILE="$WORK_DIR/evaluation_summary.md"

cat > "$SUMMARY_FILE" <<EOF
# 999M模型评估总结报告

**评估时间**: $(date +"%Y-%m-%d %H:%M:%S")
**模型路径**: $MODEL_PATH

## 评估概览

本次评估包含以下测试：
EOF

# 添加困惑度结果
if [ -f "$PERPLEXITY_RESULT" ]; then
    echo "- ✅ 困惑度评估" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "### 困惑度结果" >> "$SUMMARY_FILE"
    echo "\`\`\`json" >> "$SUMMARY_FILE"
    cat "$PERPLEXITY_RESULT" >> "$SUMMARY_FILE"
    echo "\`\`\`" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
fi

# 添加其他评估结果的链接
cat >> "$SUMMARY_FILE" <<EOF

## 详细结果

所有详细结果保存在以下目录：

- 困惑度评估: \`$WORK_DIR/perplexity_test/\`
- 综合评估: \`$WORK_DIR/comprehensive_eval/\`

## 下一步建议

1. 查看详细的评估报告：
   \`\`\`bash
   cat $WORK_DIR/comprehensive_eval/eval_*/evaluation_report.md
   \`\`\`

2. 根据评估结果查看优化建议：
   \`\`\`bash
   cat $WORK_DIR/comprehensive_eval/eval_*/optimization_suggestions.json
   \`\`\`

3. 参考评估指南制定优化计划：
   \`\`\`bash
   cat MODEL_EVALUATION_GUIDE.md
   \`\`\`

## 快速参考

- 📖 完整指南: \`MODEL_EVALUATION_GUIDE.md\`
- 🎯 快速参考: \`QUICK_REFERENCE.md\`
- ⚙️ 配置说明: \`eval_config_999m.json\`

EOF

echo "✅ 评估报告已生成: $SUMMARY_FILE"
echo ""

# ============================================
# 第八部分：显示结果和建议
# ============================================
echo "================================================"
echo "  🎉 评估完成！"
echo "================================================"
echo ""
echo "📁 所有结果保存在: $WORK_DIR"
echo ""
echo "📄 重要文件："
echo "  - 评估总结: $SUMMARY_FILE"
echo "  - 详细报告: $WORK_DIR/comprehensive_eval/eval_*/evaluation_report.md"
echo "  - 优化建议: $WORK_DIR/comprehensive_eval/eval_*/optimization_suggestions.json"
echo ""
echo "🔍 查看总结报告："
echo "  cat $SUMMARY_FILE"
echo ""
echo "📚 参考资源："
echo "  - 完整评估指南: cat MODEL_EVALUATION_GUIDE.md"
echo "  - 快速参考卡片: cat QUICK_REFERENCE.md"
echo ""
echo "💡 下一步建议："
echo "  1. 仔细阅读评估报告，了解模型的强项和弱项"
echo "  2. 查看优化建议，确定改进重点"
echo "  3. 参考评估指南，制定具体的优化计划"
echo "  4. 实施改进后，重新运行评估验证效果"
echo ""
echo "🚀 开始优化你的模型吧！"
echo "================================================"

# 在终端显示总结
cat "$SUMMARY_FILE"
