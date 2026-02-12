#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速解决方案：解决大小写匹配问题
直接替换你现有的代码逻辑
"""

# 原始问题代码：
# item['competitor_product'].lower() in same_name_products  # 这样会失败

# ==== 推荐解决方案 ====

# 方案A: 如果 same_name_products 是列表或集合，一次性预处理（推荐）
def preprocess_products(same_name_products):
    """将产品列表转换为小写集合，提高查询效率"""
    return {product.lower() for product in same_name_products}

# 使用方式：
# same_name_products_lower = preprocess_products(same_name_products)
# 然后判断：item['competitor_product'].lower() in same_name_products_lower


# 方案B: 如果 same_name_products 是字典，保持键值映射
def preprocess_products_dict(same_name_products):
    """为字典创建小写键的映射"""
    return {key.lower(): value for key, value in same_name_products.items()}

# 使用方式：
# same_name_products_lower = preprocess_products_dict(same_name_products)
# 然后判断：item['competitor_product'].lower() in same_name_products_lower


# 方案C: 动态匹配（如果数据经常变化）
def case_insensitive_check(competitor_product, same_name_products):
    """动态进行大小写不敏感的匹配"""
    competitor_lower = competitor_product.lower()
    
    if isinstance(same_name_products, dict):
        return any(competitor_lower == key.lower() for key in same_name_products.keys())
    else:  # 列表或集合
        return any(competitor_lower == product.lower() for product in same_name_products)

# 使用方式：
# if case_insensitive_check(item['competitor_product'], same_name_products):


# ==== 实际使用示例 ====
if __name__ == "__main__":
    # 示例数据
    same_name_products = ["iPhone 15", "Samsung Galaxy", "Google Pixel"]
    
    # 测试数据
    items = [
        {'competitor_product': 'iphone 15'},
        {'competitor_product': 'SAMSUNG GALAXY'},
        {'competitor_product': 'google pixel'},
    ]
    
    print("=== 原始方法（会失败）===")
    for item in items:
        # 这是你原来的代码，会因为大小写问题失败
        result = item['competitor_product'].lower() in same_name_products
        print(f"{item['competitor_product']} -> {result}")
    
    print("\n=== 方案A: 预处理列表（推荐）===")
    same_name_products_lower = preprocess_products(same_name_products)
    print(f"预处理后的数据: {same_name_products_lower}")
    
    for item in items:
        result = item['competitor_product'].lower() in same_name_products_lower
        print(f"{item['competitor_product']} -> {result}")
    
    print("\n=== 方案C: 动态匹配 ===")
    for item in items:
        result = case_insensitive_check(item['competitor_product'], same_name_products)
        print(f"{item['competitor_product']} -> {result}")


# ==== 你需要修改的代码 ====
"""
# 原来的代码：
if item['competitor_product'].lower() in same_name_products:
    # 处理逻辑

# 修改为（方案A，推荐）：
# 在代码开始处预处理一次
same_name_products_lower = {product.lower() for product in same_name_products}

# 然后在判断处使用：
if item['competitor_product'].lower() in same_name_products_lower:
    # 处理逻辑

# 或者修改为（方案C，动态匹配）：
if any(item['competitor_product'].lower() == product.lower() for product in same_name_products):
    # 处理逻辑
"""