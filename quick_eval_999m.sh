#!/bin/bash
# 999M模型快速评估脚本

echo "================================================"
echo "  999M模型快速评估工具"
echo "================================================"
echo ""

# 检查必要的环境变量
if [ -z "$MODEL_PATH" ]; then
    echo "⚠️  请设置 MODEL_PATH 环境变量"
    echo "示例: export MODEL_PATH=/path/to/your/999m/model"
    exit 1
fi

if [ -z "$LLAMA_CONFIG" ]; then
    echo "⚠️  请设置 LLAMA_CONFIG 环境变量"
    echo "示例: export LLAMA_CONFIG=/path/to/params.json"
    exit 1
fi

if [ -z "$TOKENIZER_PATH" ]; then
    echo "⚠️  请设置 TOKENIZER_PATH 环境变量"
    echo "示例: export TOKENIZER_PATH=/path/to/tokenizer.model"
    exit 1
fi

# 默认参数
EVAL_TYPES=${EVAL_TYPES:-"basic perplexity quality speed"}
OUTPUT_DIR=${OUTPUT_DIR:-"./evaluation_results"}
NPROC=${NPROC:-1}
MASTER_PORT=${MASTER_PORT:-23456}

echo "配置信息："
echo "  模型路径: $MODEL_PATH"
echo "  配置文件: $LLAMA_CONFIG"
echo "  分词器: $TOKENIZER_PATH"
echo "  评估类型: $EVAL_TYPES"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行评估
echo "开始评估..."
echo ""

torchrun --nproc-per-node="$NPROC" --master_port "$MASTER_PORT" \
    comprehensive_model_evaluation.py \
    --pretrained_path "$MODEL_PATH" \
    --llama_config "$LLAMA_CONFIG" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --eval_types $EVAL_TYPES \
    --dtype bf16 \
    --max_seq_len 4096 \
    --max_gen_len 256 \
    2>&1 | tee "$OUTPUT_DIR"/evaluation.log

echo ""
echo "================================================"
echo "✅ 评估完成！"
echo "📁 结果保存在: $OUTPUT_DIR"
echo "================================================"
