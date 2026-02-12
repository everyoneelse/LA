#!/usr/bin/env python3
"""
示例脚本：从4卡checkpoint恢复到2卡继续训练

使用方法：
python resume_training_different_gpus.py \
    --original_checkpoint /path/to/4gpu/checkpoint \
    --new_model_parallel_size 2 \
    --output_dir /path/to/new/output
"""

import os
import sys
import argparse
from pathlib import Path

# 添加accessory路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'accessory'))

def validate_checkpoint_compatibility(checkpoint_path, target_mp_size):
    """
    验证checkpoint是否可以从原始MP size转换到目标MP size
    """
    from accessory.util.tensor_parallel import infer_checkpoint_format_and_mp_size
    
    try:
        format_name, original_mp_size = infer_checkpoint_format_and_mp_size(checkpoint_path)
        print(f"检测到checkpoint格式: {format_name}")
        print(f"原始model parallel size: {original_mp_size}")
        print(f"目标model parallel size: {target_mp_size}")
        
        # 检查兼容性
        if original_mp_size % target_mp_size == 0:
            print("✅ 兼容性检查通过: 支持从{}卡合并到{}卡".format(original_mp_size, target_mp_size))
            return True, original_mp_size, format_name
        elif target_mp_size % original_mp_size == 0:
            print("✅ 兼容性检查通过: 支持从{}卡分割到{}卡".format(original_mp_size, target_mp_size))
            return True, original_mp_size, format_name
        else:
            print("❌ 兼容性检查失败: {}和{}之间无法直接转换".format(original_mp_size, target_mp_size))
            print("建议的转换路径: 先转换为公共因子，再转换为目标大小")
            return False, original_mp_size, format_name
            
    except Exception as e:
        print(f"❌ Checkpoint验证失败: {e}")
        return False, None, None

def generate_resume_command(args, original_mp_size):
    """
    生成恢复训练的命令
    """
    # 计算新的batch size以保持相同的effective batch size
    # effective_batch_size = batch_size * accum_iter * world_size
    # 假设原来的配置
    original_batch_size = 16  # 默认值，实际应该从checkpoint中读取
    new_batch_size = original_batch_size * original_mp_size // args.new_model_parallel_size
    
    command_template = """
# 恢复训练命令 (从{original_mp}卡恢复到{new_mp}卡)
torchrun --nproc_per_node={new_mp} accessory/main_finetune.py \\
    --model_parallel_size {new_mp} \\
    --batch_size {new_batch_size} \\
    --resume {checkpoint_path} \\
    --output_dir {output_dir} \\
    --data_config /path/to/your/data_config.yaml \\
    --llama_config /path/to/your/llama_config.json \\
    --tokenizer_path /path/to/tokenizer.model \\
    --epochs 10 \\
    --lr 1e-4 \\
    --weight_decay 0.02 \\
    --precision bf16 \\
    --data_parallel fsdp \\
    --checkpointing
""".format(
        original_mp=original_mp_size,
        new_mp=args.new_model_parallel_size,
        new_batch_size=new_batch_size,
        checkpoint_path=args.original_checkpoint,
        output_dir=args.output_dir
    )
    
    return command_template.strip()

def check_gpu_memory_requirements(original_mp_size, new_mp_size):
    """
    估算GPU内存需求变化
    """
    memory_factor = original_mp_size / new_mp_size
    print(f"\n📊 内存需求分析:")
    print(f"每卡内存需求大约增加 {memory_factor:.1f}x")
    
    if memory_factor > 2:
        print("⚠️  警告: 内存需求显著增加，请确保GPU内存充足")
        print("建议: 考虑减少batch_size或启用gradient checkpointing")
    elif memory_factor > 1.5:
        print("⚠️  注意: 内存需求适度增加，建议监控GPU内存使用")
    else:
        print("✅ 内存需求变化在合理范围内")

def main():
    parser = argparse.ArgumentParser(description='从不同GPU数量的checkpoint恢复训练')
    parser.add_argument('--original_checkpoint', required=True, 
                       help='原始checkpoint路径')
    parser.add_argument('--new_model_parallel_size', type=int, required=True,
                       help='新的model parallel size (GPU数量)')
    parser.add_argument('--output_dir', required=True,
                       help='新的输出目录')
    parser.add_argument('--dry_run', action='store_true',
                       help='只生成命令，不执行')
    
    args = parser.parse_args()
    
    print("🔍 检查checkpoint兼容性...")
    is_compatible, original_mp_size, format_name = validate_checkpoint_compatibility(
        args.original_checkpoint, args.new_model_parallel_size
    )
    
    if not is_compatible:
        print("❌ 无法直接转换，请检查GPU数量配置")
        return 1
    
    # 检查内存需求
    check_gpu_memory_requirements(original_mp_size, args.new_model_parallel_size)
    
    # 生成恢复命令
    print(f"\n📝 生成的恢复训练命令:")
    command = generate_resume_command(args, original_mp_size)
    print(command)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存命令到文件
    command_file = os.path.join(args.output_dir, 'resume_command.sh')
    with open(command_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# 从{}卡checkpoint恢复到{}卡训练\n".format(original_mp_size, args.new_model_parallel_size))
        f.write("# 生成时间: $(date)\n\n")
        f.write(command)
    
    os.chmod(command_file, 0o755)
    print(f"\n💾 命令已保存到: {command_file}")
    
    if not args.dry_run:
        print(f"\n🚀 开始执行恢复训练...")
        print("注意: 实际执行前请检查所有路径和参数是否正确")
        # 这里可以添加实际执行逻辑
        # os.system(command)
    
    print("\n✅ 操作完成!")
    print("\n📋 重要提醒:")
    print("1. 确保新的GPU配置有足够内存")
    print("2. 检查所有文件路径是否正确")
    print("3. 根据实际情况调整batch_size和其他超参数")
    print("4. 建议先在小数据集上测试")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())