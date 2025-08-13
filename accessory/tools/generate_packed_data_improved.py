#!/usr/bin/env python3
"""
改进版的 generate_packed_data.py
移除了对已废弃的 tqdm.set_lock 的依赖，使用现代方法处理多进程进度条

主要改进：
1. 移除了 set_lock 的使用（新版本 tqdm 已不支持）
2. 提供三种进度条显示方案供选择
3. 更清晰的进度反馈
"""

import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])

import glob
import os
import pandas as pd
from multiprocessing import Pool, Queue, Manager
from accessory.model.tokenizer import Tokenizer
import pickle
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def pack_tokens_simple(filename, save_dir, tokenizer):
    """
    简单版本：处理单个文件的 token 打包（无进度条）
    """
    logger.info(f"开始处理: {filename}")
    
    texts = pd.read_parquet(filename)['content'].tolist()
    
    l_packed_tokens = []
    _idx = 0
    _cache = [0 for _ in range(max_len)]
    
    for t in texts:
        token_split = tokenizer.encode(t, bos=True, eos=True)
        
        if token_split[0] == 1:
            token_split = token_split[1:]
        
        while _idx + len(token_split) > max_len:
            part_len = max_len - _idx
            _cache[_idx: _idx + part_len] = token_split[:part_len]
            assert _idx + part_len == max_len
            l_packed_tokens.append(_cache)
            _idx = 0
            _cache = [0 for _ in range(max_len)]
            token_split = token_split[part_len:]
        
        remaining_len = len(token_split)
        _cache[_idx:_idx + remaining_len] = token_split
        _idx += remaining_len
        assert _cache[_idx - 1] == 2
    
    l_packed_tokens.append(_cache)
    
    save_tokens_path = os.path.join(save_dir, os.path.basename(filename).split('.')[0] + '.pkl')
    with open(save_tokens_path, 'wb') as f:
        pickle.dump(l_packed_tokens, f)
    
    logger.info(f"完成处理: {save_tokens_path}")
    return save_tokens_path


def pack_tokens_with_internal_progress(args):
    """
    带内部进度条的版本：每个进程显示自己的进度
    注意：多个进度条可能会互相干扰显示
    """
    filename, save_dir, tokenizer, worker_id = args
    
    texts = pd.read_parquet(filename)['content'].tolist()
    
    l_packed_tokens = []
    _idx = 0
    _cache = [0 for _ in range(max_len)]
    
    # 每个进程创建自己的进度条，使用 position 参数避免重叠
    pbar = tqdm(total=len(texts), 
                desc=f"Worker {worker_id}: {os.path.basename(filename)}", 
                position=worker_id,
                leave=False)
    
    for t in texts:
        token_split = tokenizer.encode(t, bos=True, eos=True)
        
        if token_split[0] == 1:
            token_split = token_split[1:]
        
        while _idx + len(token_split) > max_len:
            part_len = max_len - _idx
            _cache[_idx: _idx + part_len] = token_split[:part_len]
            assert _idx + part_len == max_len
            l_packed_tokens.append(_cache)
            _idx = 0
            _cache = [0 for _ in range(max_len)]
            token_split = token_split[part_len:]
        
        remaining_len = len(token_split)
        _cache[_idx:_idx + remaining_len] = token_split
        _idx += remaining_len
        assert _cache[_idx - 1] == 2
        
        pbar.update(1)
    
    pbar.close()
    l_packed_tokens.append(_cache)
    
    save_tokens_path = os.path.join(save_dir, os.path.basename(filename).split('.')[0] + '.pkl')
    with open(save_tokens_path, 'wb') as f:
        pickle.dump(l_packed_tokens, f)
    
    return save_tokens_path


def process_method1_imap(files, save_dir, tokenizer, num_workers=48):
    """
    方法1：使用 imap_unordered 在主进程中显示总体进度
    优点：进度条稳定，不会混乱
    缺点：只显示文件级别的进度，不显示每个文件内部的处理进度
    """
    print("\n使用方法1：imap_unordered + 主进程进度条")
    
    pool = Pool(num_workers)
    process_func = partial(pack_tokens_simple, save_dir=save_dir, tokenizer=tokenizer)
    
    # 过滤出需要处理的文件
    files_to_process = []
    for file in files:
        output_path = os.path.join(save_dir, os.path.basename(file).split('.')[0] + '.pkl')
        if not os.path.exists(output_path):
            files_to_process.append(file)
    
    if not files_to_process:
        print("所有文件已处理完成")
        return
    
    print(f"需要处理 {len(files_to_process)} 个文件")
    
    # 使用 imap_unordered 处理文件，并在主进程显示进度
    with tqdm(total=len(files_to_process), desc="处理文件") as pbar:
        for result in pool.imap_unordered(process_func, files_to_process):
            pbar.update(1)
            pbar.set_postfix_str(f"完成: {os.path.basename(result)}")
    
    pool.close()
    pool.join()


def process_method2_concurrent(files, save_dir, tokenizer, num_workers=48):
    """
    方法2：使用 tqdm.contrib.concurrent 的 process_map
    优点：代码简洁，自动处理进度
    缺点：需要 tqdm 支持 contrib.concurrent 模块
    """
    print("\n使用方法2：tqdm.contrib.concurrent.process_map")
    
    try:
        from tqdm.contrib.concurrent import process_map
        
        # 过滤出需要处理的文件
        files_to_process = []
        for file in files:
            output_path = os.path.join(save_dir, os.path.basename(file).split('.')[0] + '.pkl')
            if not os.path.exists(output_path):
                files_to_process.append(file)
        
        if not files_to_process:
            print("所有文件已处理完成")
            return
        
        print(f"需要处理 {len(files_to_process)} 个文件")
        
        process_func = partial(pack_tokens_simple, save_dir=save_dir, tokenizer=tokenizer)
        
        # 使用 process_map 处理
        results = process_map(process_func, files_to_process, 
                            max_workers=num_workers, 
                            desc="处理文件", 
                            unit="文件",
                            chunksize=1)
        
        print(f"成功处理 {len(results)} 个文件")
        
    except ImportError:
        print("tqdm.contrib.concurrent 不可用，切换到方法1")
        process_method1_imap(files, save_dir, tokenizer, num_workers)


def process_method3_multiple_bars(files, save_dir, tokenizer, num_workers=48):
    """
    方法3：每个进程显示自己的进度条
    优点：可以看到每个文件的详细处理进度
    缺点：多个进度条可能会互相干扰，显示可能混乱
    """
    print("\n使用方法3：多进程多进度条（每个进程一个）")
    print("注意：多个进度条可能会互相干扰显示")
    
    pool = Pool(num_workers)
    
    # 过滤出需要处理的文件
    files_to_process = []
    for i, file in enumerate(files):
        output_path = os.path.join(save_dir, os.path.basename(file).split('.')[0] + '.pkl')
        if not os.path.exists(output_path):
            # 添加 worker_id 用于设置进度条位置
            files_to_process.append((file, save_dir, tokenizer, i % num_workers))
    
    if not files_to_process:
        print("所有文件已处理完成")
        return
    
    print(f"需要处理 {len(files_to_process)} 个文件")
    
    # 使用 map 处理
    results = pool.map(pack_tokens_with_internal_progress, files_to_process)
    
    pool.close()
    pool.join()
    
    print(f"成功处理 {len(results)} 个文件")


# ============ 主程序 ============

if __name__ == "__main__":
    # 配置参数
    max_len = 2048
    tokenizer = Tokenizer('./tokenizer.model')
    files = glob.glob('data/*.parquet')
    files.sort()
    save_dir = "packed_tokens"
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择处理方法（可以通过命令行参数或环境变量控制）
    import argparse
    parser = argparse.ArgumentParser(description='处理和打包文本数据')
    parser.add_argument('--method', type=int, default=1, choices=[1, 2, 3],
                       help='选择处理方法: 1=imap(推荐), 2=concurrent, 3=多进度条')
    parser.add_argument('--workers', type=int, default=48,
                       help='并行工作进程数')
    
    args = parser.parse_args()
    
    print(f"使用 {args.workers} 个工作进程")
    
    if args.method == 1:
        process_method1_imap(files, save_dir, tokenizer, args.workers)
    elif args.method == 2:
        process_method2_concurrent(files, save_dir, tokenizer, args.workers)
    elif args.method == 3:
        process_method3_multiple_bars(files, save_dir, tokenizer, args.workers)
    
    print("\n处理完成！")