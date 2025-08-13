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
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore


def pack_tokens(filename, save_dir, tokenizer, progress_queue=None, *, chunk_rows: int = 10000, flush_segments: int = 5000):
    """
    流式读取 parquet 并分段写盘，避免一次性占用大量内存。
    - chunk_rows:  每次从 parquet 读取的行数
    - flush_segments: 累积多少个 max_len 段后写盘并清理内存
    """
    print(f"{filename} start")

    # 尝试获取总行数（仅用于进度展示，不影响逻辑）
    try:
        total_texts = pq.ParquetFile(filename).metadata.num_rows
    except Exception:
        total_texts = None

    l_packed_tokens = []          # 已完成的 2048 段
    _idx = 0                      # 当前 cache 写入位置
    _cache = [0 for _ in range(max_len)]  # 当前正在填充的 2048 段

    processed = 0

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
        for t in df['content']:
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
                _flush()

    # 处理完最后一个 cache
    l_packed_tokens.append(_cache)
    _flush()

    if progress_queue is not None:
        progress_queue.put((filename, processed, total_texts))

    print(f"{save_tokens_path} finished")
    return save_tokens_path


def process_with_progress(files, save_dir, tokenizer, num_workers=48):
    """
    使用进度条处理多个文件
    方案1：使用 imap_unordered 在主进程中显示总体进度
    """
    pool = Pool(processes=num_workers, maxtasksperchild=1)

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