#!/bin/bash
# 标准Benchmark测试套件运行脚本

echo "================================================"
echo "  标准Benchmark测试套件"
echo "================================================"
echo ""

# 检查必要的环境变量
if [ -z "$MODEL_PATH" ]; then
    echo "⚠️  请设置 MODEL_PATH 环境变量"
    exit 1
fi

if [ -z "$LLAMA_CONFIG" ]; then
    echo "⚠️  请设置 LLAMA_CONFIG 环境变量"
    exit 1
fi

if [ -z "$TOKENIZER_PATH" ]; then
    echo "⚠️  请设置 TOKENIZER_PATH 环境变量"
    exit 1
fi

# 配置参数
MODEL_NAME=$(basename "$MODEL_PATH")
RESULTS_DIR="benchmark_results/${MODEL_NAME}"
mkdir -p "$RESULTS_DIR"

echo "配置信息："
echo "  模型路径: $MODEL_PATH"
echo "  模型名称: $MODEL_NAME"
echo "  配置文件: $LLAMA_CONFIG"
echo "  结果目录: $RESULTS_DIR"
echo ""

# 选择要运行的测试
echo "请选择要运行的benchmark测试:"
echo "1. GSM8K (数学推理)"
echo "2. MMLU (多任务语言理解)"
echo "3. C-Eval (中文综合能力)"
echo "4. HumanEval (代码生成)"
echo "5. 全部测试"
echo ""
read -p "请输入选项 (1-5): " choice

run_gsm8k() {
    echo ""
    echo "========================================"
    echo "运行 GSM8K 测试..."
    echo "========================================"
    
    cd light-eval || exit
    
    # 创建临时脚本
    cat > run_gsm8k_temp.sh <<EOF
#!/bin/bash
task=gsm8k
pretrained_type=meta_ori
pretrained_path=$MODEL_PATH
llama_config=$LLAMA_CONFIG
tokenizer_path=$TOKENIZER_PATH
data_dir='data/gsm8k'

nproc_per_node=1
master_port=23456

exp_name=$MODEL_NAME
mkdir -p logs/"\$exp_name"

torchrun --nproc-per-node="\$nproc_per_node" --master_port "\$master_port" src/eval_"\$task".py \\
    --pretrained_type "\$pretrained_type" \\
    --llama_config "\$llama_config" \\
    --tokenizer_path "\$tokenizer_path" \\
    --pretrained_path "\$pretrained_path" \\
    --data_dir "\$data_dir" \\
    2>&1 | tee logs/"\$exp_name"/"\$task".log
EOF
    
    chmod +x run_gsm8k_temp.sh
    bash run_gsm8k_temp.sh
    rm run_gsm8k_temp.sh
    
    cd ..
    
    # 复制结果
    if [ -d "light-eval/results/$MODEL_NAME/gsm8k" ]; then
        cp -r "light-eval/results/$MODEL_NAME/gsm8k" "$RESULTS_DIR/"
        echo "✅ GSM8K 测试完成，结果已保存"
    fi
}

run_mmlu() {
    echo ""
    echo "========================================"
    echo "运行 MMLU 测试..."
    echo "========================================"
    
    cd light-eval || exit
    
    cat > run_mmlu_temp.sh <<EOF
#!/bin/bash
task=mmlu
pretrained_type=meta_ori
pretrained_path=$MODEL_PATH
llama_config=$LLAMA_CONFIG
tokenizer_path=$TOKENIZER_PATH

nproc_per_node=1
master_port=23456

exp_name=$MODEL_NAME
mkdir -p logs/"\$exp_name"

torchrun --nproc-per-node="\$nproc_per_node" --master_port "\$master_port" src/eval_"\$task".py \\
    --pretrained_type "\$pretrained_type" \\
    --llama_config "\$llama_config" \\
    --tokenizer_path "\$tokenizer_path" \\
    --pretrained_path "\$pretrained_path" \\
    2>&1 | tee logs/"\$exp_name"/"\$task".log
EOF
    
    chmod +x run_mmlu_temp.sh
    bash run_mmlu_temp.sh
    rm run_mmlu_temp.sh
    
    cd ..
    
    if [ -d "light-eval/results/$MODEL_NAME/mmlu" ]; then
        cp -r "light-eval/results/$MODEL_NAME/mmlu" "$RESULTS_DIR/"
        echo "✅ MMLU 测试完成，结果已保存"
    fi
}

run_ceval() {
    echo ""
    echo "========================================"
    echo "运行 C-Eval 测试..."
    echo "========================================"
    
    cd light-eval || exit
    
    cat > run_ceval_temp.sh <<EOF
#!/bin/bash
task=ceval
pretrained_type=meta_ori
pretrained_path=$MODEL_PATH
llama_config=$LLAMA_CONFIG
tokenizer_path=$TOKENIZER_PATH

nproc_per_node=1
master_port=23456

exp_name=$MODEL_NAME
mkdir -p logs/"\$exp_name"

torchrun --nproc-per-node="\$nproc_per_node" --master_port "\$master_port" src/eval_"\$task".py \\
    --pretrained_type "\$pretrained_type" \\
    --llama_config "\$llama_config" \\
    --tokenizer_path "\$tokenizer_path" \\
    --pretrained_path "\$pretrained_path" \\
    2>&1 | tee logs/"\$exp_name"/"\$task".log
EOF
    
    chmod +x run_ceval_temp.sh
    bash run_ceval_temp.sh
    rm run_ceval_temp.sh
    
    cd ..
    
    if [ -d "light-eval/results/$MODEL_NAME/ceval" ]; then
        cp -r "light-eval/results/$MODEL_NAME/ceval" "$RESULTS_DIR/"
        echo "✅ C-Eval 测试完成，结果已保存"
    fi
}

run_humaneval() {
    echo ""
    echo "========================================"
    echo "运行 HumanEval 测试..."
    echo "========================================"
    
    cd light-eval || exit
    
    cat > run_humaneval_temp.sh <<EOF
#!/bin/bash
task=humaneval
pretrained_type=meta_ori
pretrained_path=$MODEL_PATH
llama_config=$LLAMA_CONFIG
tokenizer_path=$TOKENIZER_PATH

nproc_per_node=1
master_port=23456

exp_name=$MODEL_NAME
mkdir -p logs/"\$exp_name"

torchrun --nproc-per-node="\$nproc_per_node" --master_port "\$master_port" src/eval_"\$task".py \\
    --pretrained_type "\$pretrained_type" \\
    --llama_config "\$llama_config" \\
    --tokenizer_path "\$tokenizer_path" \\
    --pretrained_path "\$pretrained_path" \\
    2>&1 | tee logs/"\$exp_name"/"\$task".log
EOF
    
    chmod +x run_humaneval_temp.sh
    bash run_humaneval_temp.sh
    rm run_humaneval_temp.sh
    
    cd ..
    
    if [ -d "light-eval/results/$MODEL_NAME/humaneval" ]; then
        cp -r "light-eval/results/$MODEL_NAME/humaneval" "$RESULTS_DIR/"
        echo "✅ HumanEval 测试完成，结果已保存"
    fi
}

# 根据选择运行测试
case $choice in
    1)
        run_gsm8k
        ;;
    2)
        run_mmlu
        ;;
    3)
        run_ceval
        ;;
    4)
        run_humaneval
        ;;
    5)
        run_gsm8k
        run_mmlu
        run_ceval
        run_humaneval
        ;;
    *)
        echo "无效的选项"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "✅ 测试完成！"
echo "📁 结果保存在: $RESULTS_DIR"
echo "================================================"
echo ""
echo "下一步建议："
echo "1. 查看详细结果文件"
echo "2. 分析模型强项和弱项"
echo "3. 根据评估指南制定优化计划"
echo "4. 运行综合评估: bash quick_eval_999m.sh"
