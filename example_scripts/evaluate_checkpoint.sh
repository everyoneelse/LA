#!/bin/bash

# 示例脚本：评估单个checkpoint在新验证数据集上的性能

# 模型和checkpoint配置
CHECKPOINT_PATH="/path/to/output/epoch1-iter50000"  # 或者具体的.pth文件路径
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

# 输出配置
OUTPUT_FILE="./checkpoint_eval_results.json"

# 运行评估
python evaluate_checkpoint.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --val_data_meta_path ${VAL_DATA_META_PATH} \
    --val_data_root ${VAL_DATA_ROOT} \
    --packed_data \
    --batch_size ${BATCH_SIZE} \
    --max_words ${MAX_WORDS} \
    --precision ${PRECISION} \
    --device ${DEVICE} \
    --num_workers ${NUM_WORKERS} \
    --output_file ${OUTPUT_FILE} \
    --verbose