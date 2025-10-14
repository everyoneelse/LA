#!/bin/bash

# 学习率调度策略使用示例
# 展示如何在当前repo中使用不同的学习率调度策略

llama_config="$1"
tokenizer_path="$2"
data_meta_path="$3"
data_root="$4"

data_parallel=fsdp
model_parallel=1

# =============================================================================
# 1. 标准Warmup + Cosine Decay（当前默认策略）
# =============================================================================
exp_name="pretrain/warmup_cosine"
echo "Running experiment: $exp_name"
mkdir -p output/"$exp_name"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" main_pretrain.py \
--output_dir output/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--max_words 2048 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 \
--clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--lr_schedule warmup_cosine \
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

# =============================================================================
# 2. 多周期Cosine调度（推荐用于长期训练）
# =============================================================================
exp_name="pretrain/multi_cycle_cosine"
echo "Running experiment: $exp_name"
mkdir -p output/"$exp_name"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" main_pretrain.py \
--output_dir output/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--max_words 2048 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 \
--clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--lr_schedule multi_cycle_cosine \
--cycle_length 50000 \
--cycle_decay_factor 0.8 \
--cycle_warmup_iters 1000 \
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

# =============================================================================
# 3. 纯Cosine调度（适合小模型或良好初始化）
# =============================================================================
exp_name="pretrain/pure_cosine"
echo "Running experiment: $exp_name"
mkdir -p output/"$exp_name"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" main_pretrain.py \
--output_dir output/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--max_words 2048 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 \
--clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--lr_schedule pure_cosine \
--total_iters 400000 \
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

# =============================================================================
# 4. 线性衰减调度（计算简单，稳定收敛）
# =============================================================================
exp_name="pretrain/linear_decay"
echo "Running experiment: $exp_name"
mkdir -p output/"$exp_name"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" main_pretrain.py \
--output_dir output/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--max_words 2048 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 \
--clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--lr_schedule linear_decay \
--total_iters 400000 \
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

# =============================================================================
# 5. 指数衰减调度（快速收敛）
# =============================================================================
exp_name="pretrain/exponential_decay"
echo "Running experiment: $exp_name"
mkdir -p output/"$exp_name"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" main_pretrain.py \
--output_dir output/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--max_words 2048 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 \
--clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--lr_schedule exponential_decay \
--exp_decay_rate 0.96 \
--exp_decay_steps 10000 \
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

# =============================================================================
# 6. 多阶段调度（灵活配置）
# =============================================================================
exp_name="pretrain/multi_stage"
echo "Running experiment: $exp_name"
mkdir -p output/"$exp_name"

torchrun --nproc_per_node=8 --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
--nnodes="$WORLD_SIZE" --node_rank="$RANK" main_pretrain.py \
--output_dir output/"$exp_name" \
--batch_size 4 --accum_iter 16 --num_workers 4 \
--max_words 2048 \
--lr 0.0001 --min_lr 0.00001 --warmup_iters 5000 --lr_decay_iters 400000 \
--clip_grad 2 --weight_decay 0.02 \
--data_parallel "$data_parallel" --model_parallel_size "$model_parallel" \
--llama_type llama --llama_config "$llama_config" --tokenizer_path "$tokenizer_path" \
--data_meta_path "$data_meta_path" --data_root "$data_root" \
--lr_schedule multi_stage \
2>&1 | tee -a output/"$exp_name"/output"$RANK".log

echo "所有学习率调度实验已启动完成！"