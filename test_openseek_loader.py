#!/usr/bin/env python3
"""
OpenSeek 数据加载器测试脚本
"""

import sys
import os


def test_openseek_loader(data_path_prefix):
    """
    测试 OpenSeek 数据加载器
    
    Args:
        data_path_prefix: 数据文件路径前缀
    """
    print(f"=== 测试 OpenSeek 数据加载器 ===")
    print(f"数据路径前缀: {data_path_prefix}")
    
    # 检查文件是否存在
    bin_path = data_path_prefix + '.bin'
    idx_path = data_path_prefix + '.idx'
    
    if not os.path.exists(bin_path):
        print(f"❌ 二进制文件不存在: {bin_path}")
        return False
    
    if not os.path.exists(idx_path):
        print(f"❌ 索引文件不存在: {idx_path}")
        return False
    
    print(f"✅ 文件存在检查通过")
    
    try:
        # 导入优化加载器
        from openseek_optimized_loader import load_openseek_optimized
        
        print(f"\n正在加载数据集...")
        dataset = load_openseek_optimized(data_path_prefix)
        
        print(f"\n=== 基本测试 ===")
        print(f"数据集大小: {len(dataset):,} 文档")
        
        # 测试读取第一个文档
        print(f"\n测试读取第一个文档...")
        try:
            first_doc = dataset[0]
            print(f"✅ 第一个文档读取成功")
            print(f"   长度: {len(first_doc)} tokens")
            print(f"   数据类型: {first_doc.dtype}")
            print(f"   前10个tokens: {first_doc[:10].tolist()}")
            if len(first_doc) > 10:
                print(f"   后10个tokens: {first_doc[-10:].tolist()}")
            print(f"   值范围: {first_doc.min()} - {first_doc.max()}")
        except Exception as e:
            print(f"❌ 读取第一个文档失败: {e}")
            return False
        
        # 测试读取中间的文档
        print(f"\n测试读取中间文档...")
        try:
            mid_idx = len(dataset) // 2
            mid_doc = dataset[mid_idx]
            print(f"✅ 中间文档 (索引 {mid_idx}) 读取成功")
            print(f"   长度: {len(mid_doc)} tokens")
            print(f"   前5个tokens: {mid_doc[:5].tolist()}")
        except Exception as e:
            print(f"❌ 读取中间文档失败: {e}")
            return False
        
        # 测试批量读取
        print(f"\n测试批量读取...")
        try:
            batch_indices = [0, 1, 2, 3, 4]
            batch = dataset.get_batch(batch_indices)
            print(f"✅ 批量读取成功")
            print(f"   批次大小: {len(batch)}")
            for i, doc in enumerate(batch):
                print(f"   文档 {batch_indices[i]}: {len(doc)} tokens")
        except Exception as e:
            print(f"❌ 批量读取失败: {e}")
            return False
        
        # 显示统计信息
        print(f"\n=== 数据集统计信息 ===")
        try:
            stats = dataset.get_statistics()
            print(f"总文档数: {stats['total_documents']:,}")
            print(f"总tokens数: {stats['total_tokens']:,}")
            print(f"平均文档长度: {stats['avg_doc_length']:.1f}")
            print(f"文档长度范围: {stats['min_doc_length']} - {stats['max_doc_length']}")
            print(f"文档长度中位数: {stats['median_doc_length']:.1f}")
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
        
        # 随机采样测试
        print(f"\n=== 随机采样测试 ===")
        try:
            samples = dataset.sample_documents(3)
            for i, sample in enumerate(samples):
                if 'error' in sample:
                    print(f"❌ 样本 {i+1} 读取失败: {sample['error']}")
                else:
                    print(f"✅ 样本 {i+1} (索引 {sample['index']}):")
                    print(f"   长度: {sample['length']} tokens")
                    print(f"   前5个tokens: {sample['first_10_tokens'][:5]}")
        except Exception as e:
            print(f"❌ 随机采样失败: {e}")
        
        print(f"\n🎉 所有测试完成！数据加载器工作正常。")
        
        # 关闭数据集
        dataset.close()
        return True
        
    except Exception as e:
        print(f"❌ 加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python3 test_openseek_loader.py <data_path_prefix>")
        print()
        print("示例:")
        print("  python3 test_openseek_loader.py /home/hy/GPT/workspace/git/LLaMA2-Accessory-main/BAAI/OpenSeek-Pretrain-100B/arxiv/007_00000_text_document")
        print()
        print("注意: data_path_prefix 应该是不包含 .bin 或 .idx 扩展名的文件路径前缀")
        return
    
    data_path_prefix = sys.argv[1]
    success = test_openseek_loader(data_path_prefix)
    
    if success:
        print(f"\n✅ 测试成功！您可以使用以下代码加载数据:")
        print(f"```python")
        print(f"from openseek_optimized_loader import load_openseek_optimized")
        print(f"dataset = load_openseek_optimized('{data_path_prefix}')")
        print(f"first_doc = dataset[0]")
        print(f"print(f'第一个文档: {{len(first_doc)}} tokens')")
        print(f"dataset.close()")
        print(f"```")
    else:
        print(f"\n❌ 测试失败，请检查文件路径和格式")


if __name__ == "__main__":
    main()