#!/usr/bin/env python3
"""
手机号提取示例 - 解决您遇到的具体问题
这个脚本演示如何让模型在输出手机号后停止
"""

import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

import re
import torch
from accessory.model.meta import MetaModel
from typing import List, Dict, Any


class PhoneExtractor:
    """专门用于手机号提取的类"""
    
    def __init__(self, model):
        self.model = model
        
        # 手机号正则表达式模式
        self.phone_patterns = [
            r'1[3-9]\d{9}',  # 中国手机号：以1开头，第二位3-9，总共11位
            r'\d{11}',       # 通用11位数字
            r'\d{3}-\d{4}-\d{4}',  # 带连字符格式
            r'\d{3}\s\d{4}\s\d{4}',  # 带空格格式
        ]
    
    def extract_phone_from_text(self, text: str) -> Dict[str, Any]:
        """从文本中提取手机号"""
        for pattern in self.phone_patterns:
            match = re.search(pattern, text)
            if match:
                phone_number = match.group()
                # 找到手机号在文本中的位置
                start_pos = match.start()
                end_pos = match.end()
                
                # 截取到手机号结束的位置
                truncated_text = text[:end_pos]
                
                return {
                    'success': True,
                    'phone_number': phone_number,
                    'truncated_text': truncated_text,
                    'original_text': text,
                    'pattern_used': pattern
                }
        
        return {
            'success': False,
            'phone_number': None,
            'truncated_text': text,
            'original_text': text,
            'pattern_used': None
        }
    
    @torch.inference_mode()
    def generate_and_extract_phone(
        self, 
        prompt: str, 
        max_gen_len: int = 15,
        temperature: float = 1.0,
        top_p: float = 0.6,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        生成文本并提取手机号
        
        Args:
            prompt: 输入提示
            max_gen_len: 最大生成长度
            temperature: 温度参数
            top_p: top_p参数
            max_attempts: 最大尝试次数
        
        Returns:
            提取结果
        """
        print(f"输入提示: {prompt}")
        print(f"生成参数: max_len={max_gen_len}, temp={temperature}, top_p={top_p}")
        print("-" * 50)
        
        best_result = None
        all_attempts = []
        
        for attempt in range(max_attempts):
            print(f"尝试 {attempt + 1}/{max_attempts}")
            
            # 生成文本
            results = self.model.generate(
                [prompt], 
                None,  # image
                max_gen_len=max_gen_len, 
                temperature=temperature, 
                top_p=top_p
            )
            
            generated_text = results[0].strip()
            print(f"原始生成: {generated_text}")
            
            # 提取手机号
            extraction_result = self.extract_phone_from_text(generated_text)
            extraction_result['attempt'] = attempt + 1
            all_attempts.append(extraction_result)
            
            if extraction_result['success']:
                print(f"✅ 提取到手机号: {extraction_result['phone_number']}")
                print(f"截取后文本: {extraction_result['truncated_text']}")
                
                # 选择最短的成功结果作为最佳结果
                if best_result is None or len(extraction_result['truncated_text']) < len(best_result['truncated_text']):
                    best_result = extraction_result
            else:
                print(f"❌ 未提取到手机号")
            
            print()
        
        return {
            'best_result': best_result,
            'all_attempts': all_attempts,
            'success': best_result is not None
        }


def demonstrate_solution():
    """演示解决方案"""
    print("=" * 60)
    print("📱 手机号提取解决方案演示")
    print("=" * 60)
    
    # 注意：这里需要您提供实际的模型路径
    print("⚠️  请先设置您的模型路径!")
    print("修改下面的 model_path 变量为您的实际模型路径")
    print()
    
    # 示例配置 - 请根据您的实际情况修改
    model_config = {
        'pretrained_path': '/path/to/your/model',  # 请修改为您的模型路径
        'llama_type': 'llama2_7B',  # 请根据您的模型类型修改
        'tokenizer_path': '/path/to/tokenizer.model',  # 请修改为您的tokenizer路径
    }
    
    print("如果您已经有加载好的模型，可以直接传入 PhoneExtractor")
    print("示例代码:")
    print("""
# 假设您已经有了加载好的模型
# model = your_loaded_model
# extractor = PhoneExtractor(model)

# 测试提取
test_prompts = [
    "卢经理 联系方式:",
    "张总监 手机号码:",
    "客服电话:",
]

for prompt in test_prompts:
    result = extractor.generate_and_extract_phone(
        prompt=prompt,
        max_gen_len=15,
        temperature=1.0,
        top_p=0.6,
        max_attempts=3
    )
    
    if result['success']:
        best = result['best_result']
        print(f"成功提取: {best['phone_number']}")
        print(f"清理后的输出: {best['truncated_text']}")
    else:
        print("提取失败")
""")


def create_usage_guide():
    """创建使用指南"""
    guide_content = """
# 手机号提取解决方案使用指南

## 问题描述
您的预训练模型在生成手机号后继续输出其他内容，这是正常现象。预训练模型学会了数据中的模式，
在新闻数据中，联系方式后通常跟着更多信息。

## 解决方案

### 方法1: 后处理提取（推荐）
使用正则表达式从生成的文本中提取手机号，然后截取到手机号结束位置。

```python
import re

def extract_phone_and_truncate(text):
    phone_pattern = r'1[3-9]\\d{9}'  # 中国手机号模式
    match = re.search(phone_pattern, text)
    if match:
        return text[:match.end()]
    return text

# 使用示例
generated = "13974628606\\n  2.采购代理机构信息(如有"
clean_result = extract_phone_and_truncate(generated)
print(clean_result)  # 输出: "13974628606"
```

### 方法2: 使用additional_stop_symbols
在generate方法中添加停止符号：

```python
results = model.generate(
    ["卢经理 联系方式:"], 
    None,  # image
    max_gen_len=15, 
    temperature=1, 
    top_p=0.6,
    additional_stop_symbols=['\\n', '  ', '。', '，']  # 添加停止符号
)
```

### 方法3: 调整生成参数
- 降低temperature（如0.1-0.3）使生成更确定性
- 减小max_gen_len限制生成长度
- 调整top_p值

### 方法4: 多次生成选择最佳结果
生成多次，选择最符合期望的结果。

## 长期解决方案

1. **指令微调**: 在包含明确停止指令的数据上微调模型
2. **强化学习**: 使用RLHF训练模型学会在合适位置停止
3. **提示工程**: 设计更好的提示格式

## 建议的改进方向

1. **数据准备**: 准备一些"联系方式: 手机号"格式的训练数据
2. **微调训练**: 在这些数据上进行少量微调
3. **评估指标**: 建立评估生成质量的指标

这是预训练模型的正常行为，通过后处理可以很好地解决您的问题。
"""
    
    with open('/workspace/phone_extraction_guide.md', 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("✅ 已创建使用指南: phone_extraction_guide.md")


if __name__ == "__main__":
    demonstrate_solution()
    create_usage_guide()
    
    print("\n" + "=" * 60)
    print("💡 解决方案总结:")
    print("1. 您遇到的现象完全正常")
    print("2. 推荐使用后处理方法提取手机号")
    print("3. 可以调整生成参数获得更好效果")
    print("4. 长期可考虑指令微调")
    print("=" * 60)