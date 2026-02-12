#!/usr/bin/env python3
"""
智能手机号生成器
自动截取生成内容到手机号结束
"""

import re

#!/usr/bin/env python3
"""
快速手机号提取修复脚本
直接解决您当前遇到的问题
"""

import re
from typing import Dict, List, Optional, Tuple


def extract_phone_number(text: str) -> Dict[str, any]:
    """
    从文本中提取手机号并截取到手机号结束
    
    Args:
        text: 输入文本
    
    Returns:
        包含提取结果的字典
    """
    # 中国手机号正则表达式模式
    phone_patterns = [
        r'1[3-9]\d{9}',           # 标准中国手机号
        r'\d{11}',                # 11位数字
        r'1[3-9]\d-\d{4}-\d{4}',  # 带连字符
        r'1[3-9]\d\s\d{4}\s\d{4}' # 带空格
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            phone_number = match.group()
            # 截取到手机号结束位置
            clean_text = text[:match.end()]
            
            return {
                'success': True,
                'phone_number': phone_number,
                'clean_text': clean_text,
                'original_text': text,
                'pattern': pattern
            }
    
    return {
        'success': False,
        'phone_number': None,
        'clean_text': text,
        'original_text': text,
        'pattern': None
    }


def fix_your_generation_result(generated_text: str) -> str:
    """
    修复您的生成结果
    输入: '13974628606\n  2.采购代理机构信息(如有'
    输出: '13974628606'
    """
    result = extract_phone_number(generated_text)
    if result['success']:
        return result['clean_text']
    else:
        return generated_text


def demonstrate_fix():
    """演示修复效果"""
    print("=" * 60)
    print("🔧 手机号提取修复演示")
    print("=" * 60)
    
    # 您的实际例子
    your_example = "13974628606\n  2.采购代理机构信息(如有"
    print(f"原始生成结果: {repr(your_example)}")
    
    fixed_result = fix_your_generation_result(your_example)
    print(f"修复后结果: {repr(fixed_result)}")
    
    print("\n详细信息:")
    detail = extract_phone_number(your_example)
    for key, value in detail.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-" * 40)
    print("更多测试例子:")
    
    test_cases = [
        "张经理电话：13812345678，请联系时说明来意",
        "客服热线15987654321\n营业时间：9:00-18:00",
        "联系人：李总 手机号码：13698765432 邮箱：li@company.com",
        "销售部门：139-8765-4321\n地址：北京市朝阳区",
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"原文: {test_case}")
        result = extract_phone_number(test_case)
        if result['success']:
            print(f"✅ 提取: {result['phone_number']}")
            print(f"清理: {result['clean_text']}")
        else:
            print("❌ 未找到手机号")



class SmartPhoneGenerator:
    """智能手机号生成器 - 自动截取到手机号结束"""
    
    def __init__(self, model):
        self.model = model
    
    def generate_phone_contact(self, prompt, **kwargs):
        """生成联系方式并自动截取到手机号"""
        # 使用您现有的生成代码
        results = self.model.generate(
            [prompt], 
            None,  # image
            max_gen_len=kwargs.get('max_gen_len', 15), 
            temperature=kwargs.get('temperature', 1), 
            top_p=kwargs.get('top_p', 0.6)
        )
        
        generated_text = results[0].strip()
        
        # 自动提取和截取手机号
        phone_result = extract_phone_number(generated_text)
        
        if phone_result['success']:
            return {
                'text': phone_result['clean_text'],
                'phone': phone_result['phone_number'],
                'success': True,
                'original': generated_text
            }
        else:
            return {
                'text': generated_text,
                'phone': None,
                'success': False,
                'original': generated_text
            }

# 使用示例:
# smart_gen = SmartPhoneGenerator(your_model)
# result = smart_gen.generate_phone_contact("卢经理 联系方式:")
# print(result['text'])  # 只包含到手机号的内容
