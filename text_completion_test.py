#!/usr/bin/env python3
"""
文本补全测试脚本
用于测试训练好的模型的文本补全能力
"""

import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

from accessory.model.meta import MetaModel
import argparse
import torch
import torch.distributed as dist
import numpy as np
import random
from accessory.util import misc
from fairscale.nn.model_parallel import initialize as fs_init
from accessory.data.alpaca import format_prompt


def get_args_parser():
    parser = argparse.ArgumentParser('文本补全测试', add_help=False)
    
    # 模型参数
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                        help='预训练模型检查点目录')
    parser.add_argument('--llama_type', default=None, type=str, metavar='MODEL',
                        help='llama模型类型')
    parser.add_argument('--llama_config', default=None, type=str, nargs="*",
                        help='llama模型配置路径')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='tokenizer.model路径')
    
    # 生成参数
    parser.add_argument('--max_seq_len', type=int, default=4096, help='最大序列长度')
    parser.add_argument('--max_gen_len', type=int, default=128, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.1, help='温度参数')
    parser.add_argument('--top_p', type=float, default=0.75, help='top-p参数')
    
    # 设备和精度
    parser.add_argument('--device', default='cuda', help='推理设备')
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
                        help="模型权重和推理的数据类型")
    parser.add_argument('--quant', action='store_true', help="启用量化")
    
    # 分布式
    parser.add_argument('--dist_on_itp', action='store_true')
    
    return parser


class TextCompletionTester:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.target_dtype = None
        self._setup_model()
    
    def _setup_model(self):
        """设置模型"""
        # 设置随机种子
        random.seed(0)
        torch.random.manual_seed(0)
        np.random.seed(0)
        
        # 初始化分布式
        misc.init_distributed_mode(self.args)
        fs_init.initialize_model_parallel(dist.get_world_size())
        
        # 设置数据类型
        self.target_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.args.dtype]
        
        # 加载模型
        print(f"正在从 {self.args.pretrained_path} 加载模型...")
        self.model = MetaModel.from_pretrained(
            self.args.pretrained_path, 
            self.args.llama_type, 
            self.args.llama_config, 
            self.args.tokenizer_path,
            with_visual=False, 
            max_seq_len=self.args.max_seq_len,
            mp_group=fs_init.get_model_parallel_group(),
            dtype=self.target_dtype, 
            device="cpu" if self.args.quant else "cuda"
        )
        
        # 量化处理
        if self.args.quant:
            print("将模型量化为4bit!")
            from accessory.util.quant import quantize
            from transformers.utils.quantization_config import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig.from_dict(
                config_dict={
                    "load_in_8bit": False, 
                    "load_in_4bit": True, 
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16
                },
                return_unused_kwargs=False,
            )
            quantize(self.model, quantization_config)
        
        print("模型加载完成!")
        print("Model = %s" % str(self.model))
        self.model.bfloat16().cuda()
    
    @torch.inference_mode()
    def complete_text(self, prompt_text, system_prompt="alpaca"):
        """
        文本补全函数
        
        Args:
            prompt_text: 输入的文本提示
            system_prompt: 系统提示类型
        
        Returns:
            补全后的文本
        """
        # 格式化提示
        if system_prompt == "alpaca":
            formatted_prompt = format_prompt({"instruction": prompt_text, "input": ""}, system_prompt)
        else:
            formatted_prompt = prompt_text
        
        print(f"\n输入提示: {prompt_text}")
        print(f"格式化后的提示: {formatted_prompt}")
        print("=" * 50)
        
        # 分布式同步
        if dist.is_initialized():
            dist.barrier()
            dist.broadcast_object_list([formatted_prompt, None, self.args.max_gen_len, self.args.temperature, self.args.top_p])
        
        # 生成文本
        if self.args.quant:
            results = self.model.generate(
                [formatted_prompt], 
                None,  # image
                max_gen_len=self.args.max_gen_len, 
                temperature=self.args.temperature, 
                top_p=self.args.top_p
            )
        else:
            with torch.cuda.amp.autocast(dtype=self.target_dtype):
                results = self.model.generate(
                    [formatted_prompt], 
                    None,  # image
                    max_gen_len=self.args.max_gen_len, 
                    temperature=self.args.temperature, 
                    top_p=self.args.top_p
                )
        
        generated_text = results[0].strip()
        return generated_text
    
    def interactive_test(self):
        """交互式测试"""
        print("\n" + "=" * 60)
        print("🚀 文本补全交互式测试")
        print("=" * 60)
        print("输入 'quit' 或 'exit' 退出程序")
        print("输入 'params' 查看当前生成参数")
        print("输入 'set_temp <值>' 设置温度参数")
        print("输入 'set_top_p <值>' 设置top_p参数")
        print("输入 'set_max_len <值>' 设置最大生成长度")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n请输入文本提示 >>> ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("再见!")
                    break
                
                if user_input.lower() == 'params':
                    print(f"当前参数:")
                    print(f"  温度 (temperature): {self.args.temperature}")
                    print(f"  Top-p: {self.args.top_p}")
                    print(f"  最大生成长度: {self.args.max_gen_len}")
                    continue
                
                if user_input.lower().startswith('set_temp '):
                    try:
                        new_temp = float(user_input.split()[1])
                        self.args.temperature = new_temp
                        print(f"温度已设置为: {new_temp}")
                    except:
                        print("温度设置错误，请输入有效数值")
                    continue
                
                if user_input.lower().startswith('set_top_p '):
                    try:
                        new_top_p = float(user_input.split()[1])
                        self.args.top_p = new_top_p
                        print(f"Top-p已设置为: {new_top_p}")
                    except:
                        print("Top-p设置错误，请输入有效数值")
                    continue
                
                if user_input.lower().startswith('set_max_len '):
                    try:
                        new_max_len = int(user_input.split()[1])
                        self.args.max_gen_len = new_max_len
                        print(f"最大生成长度已设置为: {new_max_len}")
                    except:
                        print("最大生成长度设置错误，请输入有效整数")
                    continue
                
                if not user_input:
                    print("请输入有效的文本提示")
                    continue
                
                # 进行文本补全
                result = self.complete_text(user_input)
                
                print(f"\n🤖 模型补全结果:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                continue
    
    def batch_test(self, test_prompts):
        """批量测试"""
        print("\n" + "=" * 60)
        print("📝 批量文本补全测试")
        print("=" * 60)
        
        results = []
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n测试 {i}/{len(test_prompts)}")
            result = self.complete_text(prompt)
            results.append({
                'prompt': prompt,
                'completion': result
            })
            
            print(f"🤖 补全结果:")
            print("-" * 40)
            print(result)
            print("-" * 40)
        
        return results


def worker_func():
    """分布式工作进程函数"""
    while True:
        if dist.is_initialized():
            dist.barrier()
            input_data = [None for _ in range(5)]
            dist.broadcast_object_list(input_data)
            _prompt, image, max_gen_len, gen_t, top_p = input_data
            # 这里可以添加工作进程的处理逻辑


def main():
    args = get_args_parser().parse_args()
    
    # 检查必要参数
    if args.pretrained_path == '/path/to/pretrained':
        print("⚠️  请设置正确的模型路径 --pretrained_path")
        print("示例: python text_completion_test.py --pretrained_path /your/model/path")
        return
    
    # 创建测试器
    try:
        tester = TextCompletionTester(args)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 如果是主进程，运行测试
    if not dist.is_initialized() or dist.get_rank() == 0:
        # 预定义的测试提示
        test_prompts = [
            "人工智能的发展历程可以分为",
            "深度学习的核心思想是",
            "自然语言处理技术在现代社会的应用包括",
            "机器学习算法的分类主要有",
            "大语言模型的训练过程需要"
        ]
        
        print("选择测试模式:")
        print("1. 交互式测试")
        print("2. 批量测试")
        
        while True:
            choice = input("请选择 (1/2): ").strip()
            if choice == '1':
                tester.interactive_test()
                break
            elif choice == '2':
                tester.batch_test(test_prompts)
                break
            else:
                print("请输入 1 或 2")
    else:
        # 其他进程作为工作进程
        worker_func()


if __name__ == "__main__":
    main()