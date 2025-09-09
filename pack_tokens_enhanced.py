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
import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore

# Megatron-LM support
try:
    from megatron.data import indexed_dataset
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False
    print("Warning: Megatron-LM not available. Only parquet files will be supported.")


def detect_file_type(filename):
    """
    检测文件类型：parquet 或 megatron indexed dataset
    """
    if filename.endswith('.parquet'):
        return 'parquet'
    elif os.path.exists(filename + '.idx') and os.path.exists(filename + '.bin'):
        return 'megatron'
    elif os.path.basename(filename).find('.') == -1:
        # 可能是 megatron dataset 的前缀
        if os.path.exists(filename + '.idx') and os.path.exists(filename + '.bin'):
            return 'megatron'
    return 'unknown'


def read_megatron_dataset(dataset_prefix):
    """
    读取 Megatron indexed dataset
    """
    if not MEGATRON_AVAILABLE:
        raise ImportError("Megatron-LM is not available. Please install it first.")
    
    try:
        dataset = indexed_dataset.make_dataset(dataset_prefix, impl='mmap')
        print(f"Megatron 数据集大小: {len(dataset)}")
        return dataset
    except Exception as e:
        print(f"Error loading Megatron dataset {dataset_prefix}: {e}")
        return None


def pack_tokens_parquet(filename, save_dir, tokenizer, progress_queue=None, *, chunk_rows: int = 10000, flush_segments: int = 5000):
    """
    处理 parquet 文件的原始函数
    """
    print(f"Processing parquet: {filename}")

    # 尝试获取总行数（仅用于进度展示，不影响逻辑）
    try:
        total_texts = pq.ParquetFile(filename).metadata.num_rows
    except Exception:
        total_texts = None

    l_packed_tokens = []          # 已完成的 max_len 段
    _idx = 0                      # 当前 cache 写入位置
    _cache = [0 for _ in range(max_len)]  # 当前正在填充的 max_len 段

    processed = 0
    total_token = 0

    save_tokens_path = os.path.join(save_dir, os.path.basename(filename).split('.')[0] + '.pkl')

    # flush 帮助函数：把已累积的段写入磁盘并清空内存
    def _flush():
        nonlocal l_packed_tokens
        if l_packed_tokens:
            # 追加写入，protocol=pickle.HIGHEST_PROTOCOL 能获得最好效率
            with open(save_tokens_path, 'ab') as f:
                pickle.dump(l_packed_tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
            l_packed_tokens = []

    # ========= 读取 parquet =========
    try:
        parquet_iter = pd.read_parquet(filename, columns=['content'], iterator=True, chunksize=chunk_rows)
    except TypeError:
        # pandas<2.0 不支持 iterator; 回退到 pyarrow（若可用），否则一次性读取
        if pq is None:
            parquet_iter = [pd.read_parquet(filename, columns=['content'])]
        else:
            parquet_file = pq.ParquetFile(filename)
            parquet_iter = (batch.to_pandas() for batch in parquet_file.iter_batches(batch_size=chunk_rows, columns=['content']))

    for df in parquet_iter:
        for t in tqdm(df['content'], desc=f"Processing {os.path.basename(filename)}"):
            token_split = tokenizer.encode(t, bos=True, eos=True)

            if token_split and token_split[0] == 1:
                token_split = token_split[1:]

            # 把 token_split 写入当前 cache，必要时截断并换行
            while _idx + len(token_split) > max_len:
                part_len = max_len - _idx
                _cache[_idx: _idx + part_len] = token_split[:part_len]
                l_packed_tokens.append(_cache)
                _idx = 0
                _cache = [0 for _ in range(max_len)]
                token_split = token_split[part_len:]

            remaining_len = len(token_split)
            _cache[_idx:_idx + remaining_len] = token_split
            _idx += remaining_len
            if remaining_len:
                assert _cache[_idx - 1] == 2

            processed += 1
            if progress_queue is not None and processed % 10000 == 0:
                progress_queue.put((filename, processed, total_texts))

            # 达到阈值就写盘并清空
            if len(l_packed_tokens) >= flush_segments:
                for lits in l_packed_tokens:
                    total_token += len(lits)
                print(f"Total tokens so far: {total_token}")
                _flush()

    # 处理完最后一个 cache
    if _idx > 0:  # 只有当cache中有内容时才添加
        l_packed_tokens.append(_cache)
    
    for lits in l_packed_tokens:
        total_token += len(lits)
    print(f"Final total tokens: {total_token}")
    _flush()

    if progress_queue is not None:
        progress_queue.put((filename, processed, total_texts))

    print(f"{save_tokens_path} finished")
    return save_tokens_path


def pack_tokens_megatron(dataset_prefix, save_dir, tokenizer, progress_queue=None, *, flush_segments: int = 5000):
    """
    处理 Megatron indexed dataset
    """
    print(f"Processing Megatron dataset: {dataset_prefix}")
    
    dataset = read_megatron_dataset(dataset_prefix)
    if dataset is None:
        return None

    l_packed_tokens = []          # 已完成的 max_len 段
    _idx = 0                      # 当前 cache 写入位置
    _cache = [0 for _ in range(max_len)]  # 当前正在填充的 max_len 段

    processed = 0
    total_token = 0
    total_docs = len(dataset)

    save_tokens_path = os.path.join(save_dir, os.path.basename(dataset_prefix) + '_megatron.pkl')

    # flush 帮助函数：把已累积的段写入磁盘并清空内存
    def _flush():
        nonlocal l_packed_tokens
        if l_packed_tokens:
            with open(save_tokens_path, 'ab') as f:
                pickle.dump(l_packed_tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
            l_packed_tokens = []

    # ========= 读取 Megatron dataset =========
    for i in tqdm(range(total_docs), desc=f"Processing {os.path.basename(dataset_prefix)}"):
        try:
            # 从 Megatron dataset 读取已经 tokenized 的数据
            doc_tokens = dataset[i]
            
            # 确保是 numpy array 或 list
            if hasattr(doc_tokens, 'numpy'):
                token_split = doc_tokens.numpy().tolist()
            elif hasattr(doc_tokens, 'tolist'):
                token_split = doc_tokens.tolist()
            else:
                token_split = list(doc_tokens)
            
            # 由于 Megatron 数据已经是 token ids，我们需要使用 internlm2 tokenizer 重新处理
            # 首先将 token ids 转换回文本（如果可能），然后用新的 tokenizer 重新编码
            # 但这里我们假设可以直接使用这些 token ids，只需要调整格式
            
            # 如果需要添加 BOS/EOS tokens，可以在这里处理
            # 注意：这里假设 Megatron 数据中的 token ids 可以直接使用
            # 实际使用时可能需要根据具体情况调整
            
            # 把 token_split 写入当前 cache，必要时截断并换行
            while _idx + len(token_split) > max_len:
                part_len = max_len - _idx
                _cache[_idx: _idx + part_len] = token_split[:part_len]
                l_packed_tokens.append(_cache[:])  # 创建副本
                _idx = 0
                _cache = [0 for _ in range(max_len)]
                token_split = token_split[part_len:]

            remaining_len = len(token_split)
            if remaining_len > 0:
                _cache[_idx:_idx + remaining_len] = token_split
                _idx += remaining_len

            processed += 1
            if progress_queue is not None and processed % 10000 == 0:
                progress_queue.put((dataset_prefix, processed, total_docs))

            # 达到阈值就写盘并清空
            if len(l_packed_tokens) >= flush_segments:
                for lits in l_packed_tokens:
                    total_token += len(lits)
                print(f"Total tokens so far: {total_token}")
                _flush()
                
        except Exception as e:
            print(f"Error processing document {i}: {e}")
            continue

    # 处理完最后一个 cache
    if _idx > 0:  # 只有当cache中有内容时才添加
        l_packed_tokens.append(_cache[:])  # 创建副本
    
    for lits in l_packed_tokens:
        total_token += len(lits)
    print(f"Final total tokens: {total_token}")
    _flush()

    if progress_queue is not None:
        progress_queue.put((dataset_prefix, processed, total_docs))

    print(f"{save_tokens_path} finished")
    return save_tokens_path


def pack_tokens(filename, save_dir, tokenizer, progress_queue=None, **kwargs):
    """
    统一的 token packing 函数，自动检测文件类型并调用相应的处理函数
    """
    file_type = detect_file_type(filename)
    
    if file_type == 'parquet':
        return pack_tokens_parquet(filename, save_dir, tokenizer, progress_queue, **kwargs)
    elif file_type == 'megatron':
        return pack_tokens_megatron(filename, save_dir, tokenizer, progress_queue, **kwargs)
    else:
        print(f"Unknown file type for {filename}, skipping...")
        return None


def process_with_progress(files, save_dir, tokenizer, num_workers=48):
    """
    使用进度条处理多个文件
    """
    pool = Pool(processes=num_workers, maxtasksperchild=1)

    # 创建处理函数的偏函数
    process_func = partial(pack_tokens, save_dir=save_dir, tokenizer=tokenizer, progress_queue=None)

    # 过滤出需要处理的文件
    files_to_process = []
    for file in files:
        file_type = detect_file_type(file)
        if file_type == 'parquet':
            output_path = os.path.join(save_dir, os.path.basename(file).split('.')[0] + '.pkl')
        elif file_type == 'megatron':
            output_path = os.path.join(save_dir, os.path.basename(file) + '_megatron.pkl')
        else:
            print(f"Skipping unknown file type: {file}")
            continue
            
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
            file_type = detect_file_type(file)
            if file_type == 'parquet':
                output_path = os.path.join(save_dir, os.path.basename(file).split('.')[0] + '.pkl')
            elif file_type == 'megatron':
                output_path = os.path.join(save_dir, os.path.basename(file) + '_megatron.pkl')
            else:
                print(f"Skipping unknown file type: {file}")
                continue
                
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


def scan_for_datasets(data_dirs):
    """
    扫描目录中的数据集文件（支持 parquet 和 megatron 格式）
    """
    all_files = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Warning: Directory {data_dir} does not exist")
            continue
            
        print(f"Scanning directory: {data_dir}")
        
        # 扫描 parquet 文件
        parquet_files = glob.glob(os.path.join(data_dir, '*.parquet'))
        all_files.extend(parquet_files)
        print(f"Found {len(parquet_files)} parquet files")
        
        # 扫描 megatron 数据集文件
        # 查找所有 .idx 文件，然后检查对应的 .bin 文件是否存在
        idx_files = glob.glob(os.path.join(data_dir, '*.idx'))
        megatron_files = []
        for idx_file in idx_files:
            prefix = idx_file[:-4]  # 移除 .idx 后缀
            if os.path.exists(prefix + '.bin'):
                megatron_files.append(prefix)
        
        all_files.extend(megatron_files)
        print(f"Found {len(megatron_files)} megatron dataset files")
    
    return sorted(all_files)


if __name__ == "__main__":
    # 配置参数
    max_len = 1024
    tokenizer = Tokenizer('./internlm2-chat-126m/tokenizer.model')
    
    # 数据目录列表（可以包含 parquet 和 megatron 数据）
    data_dirs = ['CCI-DATA']  # 可以添加更多目录
    
    # 扫描所有数据集文件
    files = scan_for_datasets(data_dirs)
    
    print(f"Total files found: {len(files)}")
    for i, file in enumerate(files[:10]):  # 显示前10个文件
        file_type = detect_file_type(file)
        print(f"  {i+1}. {file} ({file_type})")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    
    save_dir = "CCI-DATA/packed_tokens"
    os.makedirs(save_dir, exist_ok=True)

    # 选择处理方案
    # 方案1：使用标准的 multiprocessing.Pool + tqdm（更稳定）
    process_with_progress(files, save_dir, tokenizer, num_workers=24)

    # 方案2：使用 tqdm.contrib.concurrent（如果可用，更简洁）
    # process_with_concurrent(files, save_dir, tokenizer, num_workers=48)