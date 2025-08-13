import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 3)[0])

import glob
import os
import pandas as pd
from multiprocessing import Pool, Manager
from accessory.model.tokenizer import Tokenizer
import pickle
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


def pack_tokens(filename, save_dir, tokenizer, progress_queue=None):
    """
    处理单个文件的 token 打包
    """
    print(f"{filename} start")
    
    texts = pd.read_parquet(filename)['content'].tolist()
    
    l_packed_tokens = []
    _idx = 0
    _cache = [0 for _ in range(max_len)]
    
    total_texts = len(texts)
    processed = 0
    
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
        
        # 更新进度
        processed += 1
        if progress_queue is not None and processed % 100 == 0:
            # 每处理100个文本更新一次进度，减少队列通信开销
            progress_queue.put((filename, processed, total_texts))
    
    # 最后一次更新进度
    if progress_queue is not None:
        progress_queue.put((filename, processed, total_texts))
    
    l_packed_tokens.append(_cache)
    
    save_tokens_path = os.path.join(save_dir, os.path.basename(filename).split('.')[0] + '.pkl')
    with open(save_tokens_path, 'wb') as f:
        pickle.dump(l_packed_tokens, f)
    
    print(f"{save_tokens_path} finished")
    return save_tokens_path


def process_with_progress(files, save_dir, tokenizer, num_workers=48):
    """
    使用进度条处理多个文件
    方案1：使用 imap_unordered 在主进程中显示总体进度
    """
    pool = Pool(num_workers)
    
    # 创建处理函数的偏函数
    process_func = partial(pack_tokens, save_dir=save_dir, tokenizer=tokenizer, progress_queue=None)
    
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
    with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
        for result in pool.imap_unordered(process_func, files_to_process):
            pbar.update(1)
            pbar.set_postfix_str(f"Completed: {result}")
    
    pool.close()
    pool.join()


def process_with_concurrent(files, save_dir, tokenizer, num_workers=48):
    """
    方案2：使用 tqdm.contrib.concurrent 的 process_map（如果可用）
    """
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
        
        # 创建处理函数的偏函数
        process_func = partial(pack_tokens, save_dir=save_dir, tokenizer=tokenizer, progress_queue=None)
        
        # 使用 process_map 处理
        results = process_map(process_func, files_to_process, max_workers=num_workers, 
                            desc="Processing files", unit="file")
        
    except ImportError:
        print("tqdm.contrib.concurrent 不可用，使用备用方案")
        process_with_progress(files, save_dir, tokenizer, num_workers)


# Modify arguments here based on your needs
max_len = 2048
tokenizer = Tokenizer('./tokenizer.model')
files = glob.glob('data/*.parquet')
files.sort()
save_dir = "packed_tokens"
os.makedirs(save_dir, exist_ok=True)

# 选择处理方案
# 方案1：使用标准的 multiprocessing.Pool + tqdm（更稳定）
process_with_progress(files, save_dir, tokenizer, num_workers=48)

# 方案2：使用 tqdm.contrib.concurrent（如果可用，更简洁）
# process_with_concurrent(files, save_dir, tokenizer, num_workers=48)