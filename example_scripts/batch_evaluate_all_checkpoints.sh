#!/bin/bash

# 示例脚本：批量评估训练输出目录中所有checkpoint的性能

# 训练输出目录（包含所有checkpoint）
OUTPUT_DIR="/path/to/training/output"
TOKENIZER_PATH="../tokenizer.model"

# 验证数据配置
VAL_DATA_META_PATH="/path/to/new_val/PretrainMetaVal.json"
VAL_DATA_ROOT="/path/to/new_val/data"

# 评估配置
BATCH_SIZE=4
MAX_WORDS=2048
PRECISION="bf16"
DEVICE="cuda"
NUM_WORKERS=5

# 结果输出目录
RESULTS_DIR="${OUTPUT_DIR}/evaluation_results"

# Checkpoint过滤选项（可选）
MIN_ITER=10000     # 最小迭代次数
MAX_ITER=100000    # 最大迭代次数
ITER_STEP=10000    # 迭代步长（只评估符合此步长的checkpoint）

# 运行批量评估
python batch_evaluate_checkpoints.py \
    --output_dir ${OUTPUT_DIR} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --val_data_meta_path ${VAL_DATA_META_PATH} \
    --val_data_root ${VAL_DATA_ROOT} \
    --packed_data \
    --batch_size ${BATCH_SIZE} \
    --max_words ${MAX_WORDS} \
    --precision ${PRECISION} \
    --device ${DEVICE} \
    --num_workers ${NUM_WORKERS} \
    --results_dir ${RESULTS_DIR} \
    --plot_results \
    --min_iter ${MIN_ITER} \
    --max_iter ${MAX_ITER} \
    --iter_step ${ITER_STEP}