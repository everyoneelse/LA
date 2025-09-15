#!/bin/bash
# CLUE预训练模型快速测试脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# 默认参数
MODEL_PATH=${1:-"/path/to/pretrained/model"}
DATA_DIR=${2:-"data/clue"}

# 模型配置 - 需要根据实际模型修改
LLAMA_TYPE="llama"
LLAMA_CONFIG="/path/to/params.json"
TOKENIZER_PATH="/path/to/tokenizer.model"

# 快速测试参数
TASKS="afqmc cmnli"  # 只测试两个最简单的任务
EVAL_MODE="both"     # 同时测试zero-shot和few-shot
MAX_EVAL_SAMPLES=20  # 每个任务只评估20个样本
NUM_SHOTS=2         # Few-shot使用2个示例

echo "=========================================="
echo "CLUE预训练模型快速测试"
echo "=========================================="
echo "测试任务: $TASKS"
echo "每任务样本数: $MAX_EVAL_SAMPLES"
echo "Few-shot示例数: $NUM_SHOTS"
echo "=========================================="

# 运行快速测试
python -u ../src/eval_clue_pretrain.py \
    --llama_type $LLAMA_TYPE \
    --llama_config $LLAMA_CONFIG \
    --tokenizer_path $TOKENIZER_PATH \
    --pretrained_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --tasks $TASKS \
    --evaluation_mode $EVAL_MODE \
    --num_shots $NUM_SHOTS \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --max_seq_len 1024 \
    --max_gen_len 64 \
    --temperature 0.1 \
    --top_p 0.9 \
    --device cuda \
    --model_parallel_size 1 \
    --overwrite \
    2>&1 | tee clue_pretrain_quick_$(date +%Y%m%d_%H%M%S).log

echo "=========================================="
echo "快速测试完成！"
echo "=========================================="