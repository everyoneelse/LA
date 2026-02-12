#!/usr/bin/env python3
"""
受控文本生成脚本
解决预训练模型输出不停止的问题，添加自定义停止条件
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
import re
from accessory.util import misc
from fairscale.nn.model_parallel import initialize as fs_init
from accessory.data.alpaca import format_prompt
from typing import List, Optional, Callable, Dict, Any


def get_args_parser():
    parser = argparse.ArgumentParser('受控文本生成测试', add_help=False)
    
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
    parser.add_argument('--max_gen_len', type=int, default=15, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    parser.add_argument('--top_p', type=float, default=0.6, help='top-p参数')
    
    # 设备和精度
    parser.add_argument('--device', default='cuda', help='推理设备')
    parser.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16",
                        help="模型权重和推理的数据类型")
    parser.add_argument('--quant', action='store_true', help="启用量化")
    
    # 分布式
    parser.add_argument('--dist_on_itp', action='store_true')
    
    return parser


class ControlledGenerator:
    """受控文本生成器，支持自定义停止条件"""
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.target_dtype = None
        self._setup_model()
        
        # 预定义的停止模式
        self.stop_patterns = {
            'phone_number': [
                r'\d{11}',  # 11位手机号
                r'\d{3}-\d{4}-\d{4}',  # 带连字符的手机号
                r'\d{3}\s\d{4}\s\d{4}',  # 带空格的手机号
            ],
            'email': [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 邮箱地址
            ],
            'punctuation': [
                r'[。！？；]',  # 中文标点符号
                r'[.!?;]',  # 英文标点符号
            ],
            'newline': [
                r'\n',  # 换行符
            ]
        }
    
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
        self.model.bfloat16().cuda()
    
    def check_stop_condition(self, text: str, stop_patterns: List[str]) -> bool:
        """检查是否满足停止条件"""
        for pattern in stop_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def extract_content_by_pattern(self, text: str, patterns: List[str]) -> Optional[str]:
        """根据模式提取内容"""
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None
    
    @torch.inference_mode()
    def controlled_generate(
        self, 
        prompt_text: str, 
        stop_after_patterns: List[str] = None,
        additional_stop_symbols: List[str] = None,
        max_attempts: int = 5
    ) -> Dict[str, Any]:
        """
        受控文本生成
        
        Args:
            prompt_text: 输入提示文本
            stop_after_patterns: 在匹配这些正则表达式模式后停止生成
            additional_stop_symbols: 额外的停止符号
            max_attempts: 最大尝试次数
        
        Returns:
            包含生成结果和元信息的字典
        """
        if stop_after_patterns is None:
            stop_after_patterns = []
        if additional_stop_symbols is None:
            additional_stop_symbols = []
        
        print(f"\n输入提示: {prompt_text}")
        print(f"停止模式: {stop_after_patterns}")
        print(f"额外停止符号: {additional_stop_symbols}")
        print("=" * 50)
        
        results = []
        best_result = None
        
        for attempt in range(max_attempts):
            print(f"尝试 {attempt + 1}/{max_attempts}")
            
            # 分布式同步
            if dist.is_initialized():
                dist.barrier()
                dist.broadcast_object_list([prompt_text, None, self.args.max_gen_len, 
                                          self.args.temperature, self.args.top_p])
            
            # 生成文本
            if self.args.quant:
                generated = self.model.generate(
                    [prompt_text], 
                    None,  # image
                    max_gen_len=self.args.max_gen_len, 
                    temperature=self.args.temperature, 
                    top_p=self.args.top_p,
                    additional_stop_symbols=additional_stop_symbols
                )
            else:
                with torch.cuda.amp.autocast(dtype=self.target_dtype):
                    generated = self.model.generate(
                        [prompt_text], 
                        None,  # image
                        max_gen_len=self.args.max_gen_len, 
                        temperature=self.args.temperature, 
                        top_p=self.args.top_p,
                        additional_stop_symbols=additional_stop_symbols
                    )
            
            generated_text = generated[0].strip()
            
            # 检查是否满足停止条件
            stop_found = False
            extracted_content = None
            
            if stop_after_patterns:
                for pattern in stop_after_patterns:
                    match = re.search(pattern, generated_text)
                    if match:
                        # 截取到匹配位置
                        stop_pos = match.end()
                        truncated_text = generated_text[:stop_pos]
                        extracted_content = match.group()
                        stop_found = True
                        
                        result = {
                            'attempt': attempt + 1,
                            'generated_text': truncated_text,
                            'extracted_content': extracted_content,
                            'pattern_matched': pattern,
                            'stop_found': True,
                            'full_generation': generated_text
                        }
                        results.append(result)
                        
                        print(f"✅ 找到匹配模式 '{pattern}': {extracted_content}")
                        print(f"截取后的文本: {truncated_text}")
                        
                        if best_result is None or len(truncated_text) < len(best_result['generated_text']):
                            best_result = result
                        break
            
            if not stop_found:
                result = {
                    'attempt': attempt + 1,
                    'generated_text': generated_text,
                    'extracted_content': None,
                    'pattern_matched': None,
                    'stop_found': False,
                    'full_generation': generated_text
                }
                results.append(result)
                print(f"❌ 未找到匹配模式，完整生成: {generated_text}")
            
            # 如果找到了满意的结果，可以提前结束
            if stop_found and best_result:
                break
        
        # 如果没有找到匹配的结果，返回最后一次生成
        if best_result is None and results:
            best_result = results[-1]
        
        return {
            'best_result': best_result,
            'all_attempts': results,
            'total_attempts': len(results)
        }
    
    def demo_phone_extraction(self):
        """演示手机号提取"""
        print("\n" + "=" * 60)
        print("📱 手机号提取演示")
        print("=" * 60)
        
        test_prompts = [
            "卢经理 联系方式:",
            "张总监 手机号码:",
            "客服电话:",
            "销售部门负责人电话:"
        ]
        
        phone_patterns = self.stop_patterns['phone_number']
        
        for prompt in test_prompts:
            print(f"\n测试提示: {prompt}")
            result = self.controlled_generate(
                prompt_text=prompt,
                stop_after_patterns=phone_patterns,
                additional_stop_symbols=['\n', '  '],  # 在换行符或双空格后停止
                max_attempts=3
            )
            
            if result['best_result'] and result['best_result']['stop_found']:
                print(f"✅ 成功提取: {result['best_result']['extracted_content']}")
                print(f"完整输出: {result['best_result']['generated_text']}")
            else:
                print("❌ 未能成功提取手机号")
                if result['best_result']:
                    print(f"原始输出: {result['best_result']['generated_text']}")
    
    def interactive_controlled_test(self):
        """交互式受控测试"""
        print("\n" + "=" * 60)
        print("🎯 受控文本生成交互式测试")
        print("=" * 60)
        print("输入 'quit' 或 'exit' 退出程序")
        print("输入 'demo' 运行手机号提取演示")
        print("输入 'patterns' 查看可用的停止模式")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n请输入文本提示 >>> ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("再见!")
                    break
                
                if user_input.lower() == 'demo':
                    self.demo_phone_extraction()
                    continue
                
                if user_input.lower() == 'patterns':
                    print("可用的停止模式:")
                    for category, patterns in self.stop_patterns.items():
                        print(f"  {category}: {patterns}")
                    continue
                
                if not user_input:
                    print("请输入有效的文本提示")
                    continue
                
                # 询问停止条件
                print("选择停止条件类型:")
                print("1. 手机号后停止")
                print("2. 邮箱后停止") 
                print("3. 标点符号后停止")
                print("4. 换行符后停止")
                print("5. 自定义模式")
                print("6. 无特殊停止条件")
                
                choice = input("请选择 (1-6): ").strip()
                
                stop_patterns = []
                additional_stops = []
                
                if choice == '1':
                    stop_patterns = self.stop_patterns['phone_number']
                    additional_stops = ['\n', '  ']
                elif choice == '2':
                    stop_patterns = self.stop_patterns['email']
                elif choice == '3':
                    stop_patterns = self.stop_patterns['punctuation']
                elif choice == '4':
                    stop_patterns = self.stop_patterns['newline']
                elif choice == '5':
                    custom_pattern = input("请输入自定义正则表达式模式: ").strip()
                    if custom_pattern:
                        stop_patterns = [custom_pattern]
                
                # 进行受控生成
                result = self.controlled_generate(
                    prompt_text=user_input,
                    stop_after_patterns=stop_patterns,
                    additional_stop_symbols=additional_stops,
                    max_attempts=3
                )
                
                print(f"\n🤖 受控生成结果:")
                print("-" * 40)
                
                if result['best_result']:
                    best = result['best_result']
                    if best['stop_found']:
                        print(f"✅ 成功控制停止")
                        print(f"提取内容: {best['extracted_content']}")
                        print(f"完整输出: {best['generated_text']}")
                        print(f"匹配模式: {best['pattern_matched']}")
                    else:
                        print(f"❌ 未找到停止条件")
                        print(f"完整输出: {best['generated_text']}")
                    
                    print(f"总尝试次数: {result['total_attempts']}")
                else:
                    print("生成失败")
                
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                continue


def main():
    args = get_args_parser().parse_args()
    
    # 检查必要参数
    if args.pretrained_path == '/path/to/pretrained':
        print("⚠️  请设置正确的模型路径 --pretrained_path")
        print("示例: python controlled_generation.py --pretrained_path /your/model/path")
        return
    
    # 创建受控生成器
    try:
        generator = ControlledGenerator(args)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 如果是主进程，运行测试
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("选择测试模式:")
        print("1. 交互式受控测试")
        print("2. 手机号提取演示")
        
        while True:
            choice = input("请选择 (1/2): ").strip()
            if choice == '1':
                generator.interactive_controlled_test()
                break
            elif choice == '2':
                generator.demo_phone_extraction()
                break
            else:
                print("请输入 1 或 2")


if __name__ == "__main__":
    main()