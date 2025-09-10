#!/bin/bash

# 示例脚本：使用独立验证数据集进行packed模式预训练

# 基本配置
MODEL_TYPE="llama"
TOKENIZER_PATH="../tokenizer.model"
MAX_WORDS=2048
BATCH_SIZE=4
ACCUM_ITER=4
LR=0.001
MIN_LR=0.0001
WARMUP_ITERS=20000
LR_DECAY_ITERS=1800000

# 训练数据配置
TRAIN_DATA_META_PATH="/path/to/train/PretrainMetaPacked.json"
TRAIN_DATA_ROOT="/path/to/train/data"

# 验证数据配置（独立的验证数据集）
VAL_DATA_META_PATH="/path/to/val/PretrainMetaVal.json"
VAL_DATA_ROOT="/path/to/val/data"

# 输出配置
OUTPUT_DIR="./output_packed_separate_val"
SAVE_FREQ=5000
VAL_FREQ=10000

# 分布式配置
MODEL_PARALLEL_SIZE=1
DATA_PARALLEL="fsdp"
PRECISION="bf16"

# 运行训练
torchrun --nproc_per_node=8 \
    accessory/main_pretrain.py \
    --llama_type ${MODEL_TYPE} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --data_meta_path ${TRAIN_DATA_META_PATH} \
    --data_root ${TRAIN_DATA_ROOT} \
    --val_data_meta_path ${VAL_DATA_META_PATH} \
    --val_data_root ${VAL_DATA_ROOT} \
    --packed_data \
    --max_words ${MAX_WORDS} \
    --batch_size ${BATCH_SIZE} \
    --accum_iter ${ACCUM_ITER} \
    --lr ${LR} \
    --min_lr ${MIN_LR} \
    --warmup_iters ${WARMUP_ITERS} \
    --lr_decay_iters ${LR_DECAY_ITERS} \
    --output_dir ${OUTPUT_DIR} \
    --save_freq ${SAVE_FREQ} \
    --val_freq ${VAL_FREQ} \
    --model_parallel_size ${MODEL_PARALLEL_SIZE} \
    --data_parallel ${DATA_PARALLEL} \
    --precision ${PRECISION} \
    --checkpointing \
    --num_workers 5 \
    --pin_mem