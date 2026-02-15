#!/bin/bash
# CLUE Benchmark评估运行脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# 默认参数
MODEL_PATH=${1:-"/path/to/pretrained/model"}
DATA_DIR=${2:-"data/clue"}
TASKS=${3:-"all"}  # 可以指定特定任务，如 "afqmc tnews" 或 "all"

# 模型配置
LLAMA_TYPE="llama"
LLAMA_CONFIG="/path/to/params.json"
TOKENIZER_PATH="/path/to/tokenizer.model"

# 评估参数
NTRAIN=5  # Few-shot示例数量
MAX_SEQ_LEN=2048
MAX_GEN_LEN=256
TEMPERATURE=0.1
TOP_P=0.9

# 检查数据是否存在，如果不存在则下载
if [ ! -d "$DATA_DIR" ]; then
    echo "CLUE数据不存在，开始下载..."
    python ../src/clue/download_clue.py --data_dir $DATA_DIR --task all
fi

# 运行评估
python -u ../src/eval_clue.py \
    --llama_type $LLAMA_TYPE \
    --llama_config $LLAMA_CONFIG \
    --tokenizer_path $TOKENIZER_PATH \
    --pretrained_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --tasks $TASKS \
    --ntrain $NTRAIN \
    --max_seq_len $MAX_SEQ_LEN \
    --max_gen_len $MAX_GEN_LEN \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --device cuda \
    --model_parallel_size 1 \
    2>&1 | tee clue_eval_$(date +%Y%m%d_%H%M%S).log

echo "评估完成！"