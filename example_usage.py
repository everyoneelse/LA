#!/usr/bin/env python3
"""
文本补全测试使用示例
演示如何使用 text_completion_test.py 进行模型测试
"""

import subprocess
import os

def show_usage_examples():
    """显示使用示例"""
    print("=" * 70)
    print("🚀 文本补全测试脚本使用指南")
    print("=" * 70)
    
    print("\n📋 基本用法:")
    print("python text_completion_test.py --pretrained_path <模型路径> [其他参数]")
    
    print("\n🔧 常用参数:")
    print("  --pretrained_path     : 预训练模型路径 (必需)")
    print("  --llama_type         : 模型类型 (如: llama2_7B)")
    print("  --llama_config       : 模型配置文件路径")
    print("  --tokenizer_path     : tokenizer路径")
    print("  --max_gen_len        : 最大生成长度 (默认: 128)")
    print("  --temperature        : 温度参数 (默认: 0.1)")
    print("  --top_p              : top-p参数 (默认: 0.75)")
    print("  --dtype              : 数据类型 (bf16/fp16, 默认: bf16)")
    print("  --quant              : 启用4bit量化")
    
    print("\n📝 使用示例:")
    
    # 示例1: 基本用法
    print("\n1️⃣  基本测试 (需要替换为您的实际模型路径):")
    example1 = """python text_completion_test.py \\
    --pretrained_path /path/to/your/model \\
    --llama_type llama2_7B \\
    --tokenizer_path /path/to/tokenizer.model"""
    print(example1)
    
    # 示例2: 使用配置文件
    print("\n2️⃣  使用配置文件:")
    example2 = """python text_completion_test.py \\
    --pretrained_path /path/to/your/model \\
    --llama_type llama2_7B \\
    --llama_config accessory/configs/model/finetune/sg/llamaAdapter.json \\
    --tokenizer_path /path/to/tokenizer.model"""
    print(example2)
    
    # 示例3: 调整生成参数
    print("\n3️⃣  调整生成参数:")
    example3 = """python text_completion_test.py \\
    --pretrained_path /path/to/your/model \\
    --llama_type llama2_7B \\
    --max_gen_len 256 \\
    --temperature 0.7 \\
    --top_p 0.9"""
    print(example3)
    
    # 示例4: 启用量化
    print("\n4️⃣  启用4bit量化 (节省显存):")
    example4 = """python text_completion_test.py \\
    --pretrained_path /path/to/your/model \\
    --llama_type llama2_7B \\
    --quant"""
    print(example4)
    
    print("\n🎯 测试模式:")
    print("运行脚本后，您可以选择:")
    print("  1. 交互式测试 - 逐个输入文本进行补全")
    print("  2. 批量测试   - 使用预定义的测试用例")
    
    print("\n💡 交互式测试命令:")
    print("  quit/exit     : 退出程序")
    print("  params        : 查看当前生成参数")
    print("  set_temp <值> : 设置温度参数")
    print("  set_top_p <值>: 设置top_p参数")
    print("  set_max_len <值>: 设置最大生成长度")
    
    print("\n📁 常见模型路径结构:")
    print("模型目录通常包含:")
    print("  ├── consolidated.*.pth  (模型权重文件)")
    print("  ├── tokenizer.model     (分词器)")
    print("  └── params.json         (参数配置)")
    
    print("\n⚠️  注意事项:")
    print("1. 确保有足够的GPU显存 (推荐8GB+)")
    print("2. 如果显存不足，可以使用 --quant 参数启用量化")
    print("3. 首次运行可能需要下载依赖包")
    print("4. 分布式运行需要设置相应的环境变量")

def check_environment():
    """检查环境配置"""
    print("\n" + "=" * 70)
    print("🔍 环境检查")
    print("=" * 70)
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("❌ CUDA不可用")
    except ImportError:
        print("❌ PyTorch未安装")
    
    # 检查依赖包
    required_packages = [
        'torch', 'transformers', 'fairscale', 'flash_attn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
    
    # 检查模型目录结构
    print(f"\n📁 当前工作目录: {os.getcwd()}")
    print("请确保您的模型文件位于可访问的路径中")

def create_sample_script():
    """创建示例启动脚本"""
    script_content = '''#!/bin/bash
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
python text_completion_test.py \\
    --pretrained_path $MODEL_PATH \\
    --llama_type $MODEL_TYPE \\
    --tokenizer_path $TOKENIZER_PATH \\
    --llama_config $CONFIG_PATH \\
    --max_gen_len $MAX_GEN_LEN \\
    --temperature $TEMPERATURE \\
    --top_p $TOP_P

# 如果显存不足，可以添加 --quant 参数:
# python text_completion_test.py \\
#     --pretrained_path $MODEL_PATH \\
#     --llama_type $MODEL_TYPE \\
#     --tokenizer_path $TOKENIZER_PATH \\
#     --quant
'''
    
    with open('/workspace/run_completion_test.sh', 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod('/workspace/run_completion_test.sh', 0o755)
    print("✅ 已创建示例启动脚本: run_completion_test.sh")

if __name__ == "__main__":
    show_usage_examples()
    check_environment()
    create_sample_script()
    
    print("\n" + "=" * 70)
    print("🎉 准备工作完成!")
    print("请根据上述指南配置您的模型路径，然后运行测试脚本。")
    print("=" * 70)