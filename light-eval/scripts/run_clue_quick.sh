#!/bin/bash
# CLUE Benchmark快速测试脚本（用于验证代码）

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
NTRAIN=2  # 减少few-shot示例
MAX_EVAL_SAMPLES=10  # 每个任务只评估10个样本
TASKS="afqmc tnews"  # 只测试两个任务

# 运行快速测试
echo "开始CLUE快速测试..."
echo "只评估任务: $TASKS"
echo "每个任务评估样本数: $MAX_EVAL_SAMPLES"

python -u ../src/eval_clue.py \
    --llama_type $LLAMA_TYPE \
    --llama_config $LLAMA_CONFIG \
    --tokenizer_path $TOKENIZER_PATH \
    --pretrained_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --tasks $TASKS \
    --ntrain $NTRAIN \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --max_seq_len 1024 \
    --max_gen_len 128 \
    --temperature 0.1 \
    --top_p 0.9 \
    --device cuda \
    --model_parallel_size 1 \
    --overwrite \
    2>&1 | tee clue_quick_test_$(date +%Y%m%d_%H%M%S).log

echo "快速测试完成！"