#!/usr/bin/env python3
"""
训练后的Prompt测试脚本
读取训练期间保存的prompt日志，使用训练好的模型进行推理测试
"""

import json
import argparse
import torch
from accessory.model.meta import MetaModel
from accessory.util.tensor_parallel import load_tensor_parallel_model_list

def test_saved_prompts(log_file, model_path, tokenizer_path):
    """测试保存的prompts"""
    
    # 加载模型 (不使用FSDP)
    model = MetaModel.from_pretrained(
        model_path, 
        llama_type="llama2_7B",  # 根据实际情况调整
        tokenizer_path=tokenizer_path,
        with_visual=False,
        dtype=torch.bfloat16,
        device="cuda"
    )
    
    # 读取prompt日志
    with open(log_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            
            print(f"\n{'='*60}")
            print(f"Testing prompts from Epoch {entry['epoch']}, Step {entry['step']}")
            print(f"{'='*60}")
            
            prompts = entry['prompts']
            params = entry['test_params']
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                results = model.generate(
                    prompts,
                    None,  # image
                    max_gen_len=params['max_gen_len'],
                    temperature=params['temperature'],
                    top_p=params['top_p']
                )
            
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                print(f"\nPrompt {i+1}: {prompt}")
                print(f"Response: {result}")
                print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True, help="Path to prompt_test_log.jsonl")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--tokenizer_path", required=True, help="Path to tokenizer")
    
    args = parser.parse_args()
    test_saved_prompts(args.log_file, args.model_path, args.tokenizer_path)
