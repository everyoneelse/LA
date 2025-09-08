#!/bin/bash
# 文本补全测试启动脚本
# 请根据您的实际情况修改以下参数

# 模型路径 - 请替换为您的实际模型路径
MODEL_PATH="/path/to/your/model"

# 模型类型 - 根据您的模型选择
MODEL_TYPE="llama2_7B"

# 分词器路径 - 通常在模型目录中
TOKENIZER_PATH="/path/to/tokenizer.model"

# 可选: 模型配置文件
CONFIG_PATH="accessory/configs/model/finetune/sg/llamaAdapter.json"

# 生成参数
MAX_GEN_LEN=128
TEMPERATURE=0.1
TOP_P=0.75

# 运行测试
python text_completion_test.py \
    --pretrained_path $MODEL_PATH \
    --llama_type $MODEL_TYPE \
    --tokenizer_path $TOKENIZER_PATH \
    --llama_config $CONFIG_PATH \
    --max_gen_len $MAX_GEN_LEN \
    --temperature $TEMPERATURE \
    --top_p $TOP_P

# 如果显存不足，可以添加 --quant 参数:
# python text_completion_test.py \
#     --pretrained_path $MODEL_PATH \
#     --llama_type $MODEL_TYPE \
#     --tokenizer_path $TOKENIZER_PATH \
#     --quant
