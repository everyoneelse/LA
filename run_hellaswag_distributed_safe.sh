#!/bin/bash

# 分布式训练安全启动脚本 - 解决HellaSwag评估中的死锁问题
# Usage: ./run_hellaswag_distributed_safe.sh [args...]

# 设置环境变量以优化分布式训练
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_TIMEOUT=1800  # 30分钟超时
export CUDA_LAUNCH_BLOCKING=0  # 允许异步执行
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 设置NCCL参数以避免死锁
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# 内存和CUDA优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# 获取GPU数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "使用 $NUM_GPUS 个GPU进行分布式训练"

# 设置合理的批次大小（分布式模式下减小以避免死锁）
DEFAULT_BATCH_SIZE=2  # 保守的批次大小
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}

echo "批次大小设置为: $BATCH_SIZE"
echo "超时设置: $NCCL_TIMEOUT 秒"

# 启动分布式训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    light-eval/src/eval_hellaswag.py \
    --batch_size=$BATCH_SIZE \
    "$@"

# 检查退出状态
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "分布式训练失败，退出码: $EXIT_CODE"
    echo "可能的解决方案："
    echo "1. 减小批次大小: export BATCH_SIZE=1"
    echo "2. 减少GPU数量: export CUDA_VISIBLE_DEVICES=0,1"
    echo "3. 检查NCCL配置和网络设置"
    echo "4. 查看上面的日志输出寻找具体错误"
fi

exit $EXIT_CODE