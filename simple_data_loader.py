#!/usr/bin/env python3
"""
简化版 OpenSeek 数据加载器

如果您安装了 Megatron-LM，可以使用这个简化版本
"""

def load_with_megatron():
    """
    使用 Megatron-LM 的官方工具加载数据
    需要先安装 Megatron-LM: pip install megatron-lm
    """
    try:
        from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
        
        # 数据文件路径前缀（不包含 .bin 或 .idx 扩展名）
        data_prefix = '/workspace/018_00000_text_document'
        
        # 加载数据集
        # 参数说明：
        # - data_prefix: 数据文件路径前缀
        # - 'mmap': 使用内存映射模式，适合大文件
        # - False: 不使用多进程
        dataset = make_indexed_dataset(data_prefix, 'mmap', False)
        
        if dataset is not None:
            print(f"数据集加载成功！")
            print(f"数据集大小: {len(dataset)}")
            
            # 获取第一个样本
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"第一个样本: {sample}")
                print(f"样本类型: {type(sample)}")
                print(f"样本长度: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")
            
            return dataset
        else:
            print("数据集加载失败，请检查文件路径")
            return None
            
    except ImportError:
        print("未找到 Megatron-LM，请先安装：")
        print("pip install megatron-lm")
        return None
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None


def load_with_custom_loader():
    """
    使用自定义加载器加载数据
    """
    try:
        from load_openseek_data import load_openseek_dataset
        
        # 加载数据集
        dataset = load_openseek_dataset('/workspace', '018_00000_text_document')
        
        print(f"使用自定义加载器加载成功！")
        print(f"数据集大小: {len(dataset)}")
        
        return dataset
        
    except Exception as e:
        print(f"使用自定义加载器时出错: {e}")
        return None


def main():
    """
    主函数：尝试不同的加载方法
    """
    print("=== OpenSeek-Pretrain-100B 数据加载测试 ===\n")
    
    # 方法1：尝试使用 Megatron-LM 官方工具
    print("1. 尝试使用 Megatron-LM 官方工具加载...")
    dataset_megatron = load_with_megatron()
    
    if dataset_megatron is None:
        # 方法2：使用自定义加载器
        print("\n2. 使用自定义加载器加载...")
        dataset_custom = load_with_custom_loader()
        
        if dataset_custom is not None:
            print("建议使用自定义加载器进行后续处理")
        else:
            print("所有加载方法都失败了，请检查：")
            print("1. 数据文件是否存在")
            print("2. 文件路径是否正确")
            print("3. 文件是否损坏")


if __name__ == "__main__":
    main()