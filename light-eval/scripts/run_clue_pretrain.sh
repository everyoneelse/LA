#!/bin/bash
# CLUE预训练模型专用评估脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# 默认参数
MODEL_PATH=${1:-"/path/to/pretrained/model"}
DATA_DIR=${2:-"data/clue"}
EVAL_MODE=${3:-"both"}  # zero-shot, few-shot, both

# 模型配置 - 需要根据实际模型修改
LLAMA_TYPE="llama"
LLAMA_CONFIG="/path/to/params.json"
TOKENIZER_PATH="/path/to/tokenizer.model"

# 评估参数
TASKS="recommended"  # 使用推荐的适合预训练模型的任务
SEED=42
MAX_SEQ_LEN=2048
MAX_GEN_LEN=128
TEMPERATURE=0.1
TOP_P=0.9

echo "=========================================="
echo "CLUE预训练模型评估"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "评估模式: $EVAL_MODE"
echo "任务集合: $TASKS"
echo "=========================================="

# 检查数据是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "CLUE数据不存在，开始下载..."
    python ../src/clue/download_clue.py --data_dir $DATA_DIR --task all
fi

# 运行评估
python -u ../src/eval_clue_pretrain.py \
    --llama_type $LLAMA_TYPE \
    --llama_config $LLAMA_CONFIG \
    --tokenizer_path $TOKENIZER_PATH \
    --pretrained_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --tasks $TASKS \
    --evaluation_mode $EVAL_MODE \
    --seed $SEED \
    --max_seq_len $MAX_SEQ_LEN \
    --max_gen_len $MAX_GEN_LEN \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --device cuda \
    --model_parallel_size 1 \
    2>&1 | tee clue_pretrain_eval_$(date +%Y%m%d_%H%M%S).log

echo "=========================================="
echo "评估完成！"
echo "=========================================="