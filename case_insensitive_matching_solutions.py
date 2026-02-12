#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大小写不敏感匹配的专业解决方案
解决 item['competitor_product'].lower() in same_name_products 的问题
"""

import re
from typing import List, Dict, Union, Set
from difflib import SequenceMatcher


class CaseInsensitiveMatcher:
    """大小写不敏感匹配器"""
    
    def __init__(self, same_name_products: Union[List[str], Dict, Set[str]]):
        """
        初始化匹配器
        
        Args:
            same_name_products: 产品名称列表、字典或集合
        """
        self.original_data = same_name_products
        
        # 根据数据类型创建不同的索引结构
        if isinstance(same_name_products, dict):
            # 如果是字典，创建小写键到原始键的映射
            self.lower_to_original = {k.lower(): k for k in same_name_products.keys()}
            self.lowercase_set = set(self.lower_to_original.keys())
        elif isinstance(same_name_products, (list, set)):
            # 如果是列表或集合，创建小写值的集合
            self.lowercase_set = {item.lower() for item in same_name_products}
            self.lower_to_original = {item.lower(): item for item in same_name_products}
        else:
            raise TypeError("same_name_products must be a list, dict, or set")
    
    def exact_match(self, competitor_product: str) -> bool:
        """
        精确匹配（大小写不敏感）
        
        Args:
            competitor_product: 竞争对手产品名称
            
        Returns:
            bool: 是否匹配
        """
        return competitor_product.lower() in self.lowercase_set
    
    def partial_match(self, competitor_product: str) -> List[str]:
        """
        部分匹配（包含关系，大小写不敏感）
        
        Args:
            competitor_product: 竞争对手产品名称
            
        Returns:
            List[str]: 匹配的产品名称列表
        """
        competitor_lower = competitor_product.lower()
        matches = []
        
        for lowercase_name in self.lowercase_set:
            if competitor_lower in lowercase_name or lowercase_name in competitor_lower:
                matches.append(self.lower_to_original[lowercase_name])
        
        return matches
    
    def fuzzy_match(self, competitor_product: str, threshold: float = 0.8) -> List[tuple]:
        """
        模糊匹配（基于相似度）
        
        Args:
            competitor_product: 竞争对手产品名称
            threshold: 相似度阈值 (0-1)
            
        Returns:
            List[tuple]: (匹配的产品名称, 相似度) 的列表
        """
        competitor_lower = competitor_product.lower()
        matches = []
        
        for lowercase_name in self.lowercase_set:
            similarity = SequenceMatcher(None, competitor_lower, lowercase_name).ratio()
            if similarity >= threshold:
                matches.append((self.lower_to_original[lowercase_name], similarity))
        
        # 按相似度降序排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def regex_match(self, competitor_product: str) -> List[str]:
        """
        正则表达式匹配（大小写不敏感）
        
        Args:
            competitor_product: 竞争对手产品名称（可包含正则表达式）
            
        Returns:
            List[str]: 匹配的产品名称列表
        """
        try:
            pattern = re.compile(competitor_product, re.IGNORECASE)
            matches = []
            
            for original_name in (self.original_data if isinstance(self.original_data, (list, set)) 
                                else self.original_data.keys()):
                if pattern.search(original_name):
                    matches.append(original_name)
            
            return matches
        except re.error:
            # 如果不是有效的正则表达式，回退到精确匹配
            return [self.lower_to_original.get(competitor_product.lower())] if self.exact_match(competitor_product) else []


# 方案1: 最简单直接的解决方案
def solution_1_preprocess_data(same_name_products: Union[List[str], Dict]) -> Set[str]:
    """
    方案1: 预处理数据，将所有产品名称转换为小写
    适用于: 数据量不大，查询频繁的场景
    """
    if isinstance(same_name_products, dict):
        return {key.lower() for key in same_name_products.keys()}
    else:
        return {item.lower() for item in same_name_products}


def solution_1_check(competitor_product: str, lowercase_products: Set[str]) -> bool:
    """使用预处理后的小写数据进行匹配"""
    return competitor_product.lower() in lowercase_products


# 方案2: 使用字典映射保持原始数据
def solution_2_create_mapping(same_name_products: Union[List[str], Dict]) -> Dict[str, str]:
    """
    方案2: 创建小写到原始值的映射
    适用于: 需要保留原始数据格式的场景
    """
    if isinstance(same_name_products, dict):
        return {key.lower(): key for key in same_name_products.keys()}
    else:
        return {item.lower(): item for item in same_name_products}


def solution_2_check(competitor_product: str, mapping: Dict[str, str]) -> Union[str, None]:
    """使用映射进行匹配，返回原始值"""
    return mapping.get(competitor_product.lower())


# 方案3: 使用any()函数进行动态比较
def solution_3_dynamic_check(competitor_product: str, same_name_products: Union[List[str], Dict]) -> bool:
    """
    方案3: 动态比较（不预处理数据）
    适用于: 数据经常变化，查询不频繁的场景
    """
    competitor_lower = competitor_product.lower()
    
    if isinstance(same_name_products, dict):
        return any(competitor_lower == key.lower() for key in same_name_products.keys())
    else:
        return any(competitor_lower == item.lower() for item in same_name_products)


# 示例使用
if __name__ == "__main__":
    # 测试数据
    same_name_products_list = ["iPhone 15", "Samsung Galaxy", "Google Pixel", "OnePlus Nord"]
    same_name_products_dict = {
        "iPhone 15": {"price": 999, "brand": "Apple"},
        "Samsung Galaxy": {"price": 899, "brand": "Samsung"},
        "Google Pixel": {"price": 699, "brand": "Google"},
        "OnePlus Nord": {"price": 399, "brand": "OnePlus"}
    }
    
    test_items = [
        {"competitor_product": "iphone 15"},  # 小写
        {"competitor_product": "SAMSUNG GALAXY"},  # 大写
        {"competitor_product": "Google pixel"},  # 混合大小写
        {"competitor_product": "oneplus nord"},  # 小写
        {"competitor_product": "Huawei P50"},  # 不存在的产品
    ]
    
    print("=== 方案1: 预处理数据 ===")
    lowercase_products = solution_1_preprocess_data(same_name_products_list)
    for item in test_items:
        result = solution_1_check(item["competitor_product"], lowercase_products)
        print(f"{item['competitor_product']} -> {result}")
    
    print("\n=== 方案2: 映射保持原始数据 ===")
    mapping = solution_2_create_mapping(same_name_products_list)
    for item in test_items:
        result = solution_2_check(item["competitor_product"], mapping)
        print(f"{item['competitor_product']} -> {result}")
    
    print("\n=== 方案3: 动态比较 ===")
    for item in test_items:
        result = solution_3_dynamic_check(item["competitor_product"], same_name_products_list)
        print(f"{item['competitor_product']} -> {result}")
    
    print("\n=== 高级匹配器示例 ===")
    matcher = CaseInsensitiveMatcher(same_name_products_list)
    
    for item in test_items:
        product = item["competitor_product"]
        print(f"\n测试产品: {product}")
        
        # 精确匹配
        exact = matcher.exact_match(product)
        print(f"  精确匹配: {exact}")
        
        # 部分匹配
        partial = matcher.partial_match(product)
        print(f"  部分匹配: {partial}")
        
        # 模糊匹配
        fuzzy = matcher.fuzzy_match(product, threshold=0.6)
        print(f"  模糊匹配: {fuzzy}")
    
    print("\n=== 性能对比建议 ===")
    print("1. 方案1 (预处理): 最快，适合查询频繁的场景")
    print("2. 方案2 (映射): 平衡性能和功能，推荐使用")
    print("3. 方案3 (动态): 最灵活，适合数据经常变化的场景")
    print("4. 高级匹配器: 功能最全，适合复杂匹配需求")