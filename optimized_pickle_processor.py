import pickle
import os
import glob
import numpy as np
import zlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp
from functools import partial

def read_all_pickle_objects(file_path):
    """读取pickle文件中的所有对象"""
    all_data = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                data_chunk = pickle.load(f)
                all_data.extend(data_chunk)
            except EOFError:
                break
    return all_data

def compress_and_save_split(args):
    """压缩并保存单个分片的数据"""
    split_data, output_path = args
    
    # 检查文件是否已存在
    if os.path.exists(output_path):
        return f"Skipped (exists): {output_path}"
    
    try:
        # 压缩存储
        compressed_data = zlib.compress(
            pickle.dumps(split_data, protocol=pickle.HIGHEST_PROTOCOL), 
            level=6  # 平衡压缩率和速度
        )
        
        # 原子性写入：先写入临时文件，然后重命名
        temp_path = output_path + '.tmp'
        with open(temp_path, 'wb') as f:
            f.write(compressed_data)
        os.rename(temp_path, output_path)
        
        return f"Saved: {output_path}"
    except Exception as e:
        return f"Error saving {output_path}: {str(e)}"

def process_single_pickle_file(pkl_file, target_dir, chunk_size=100000):
    """处理单个pickle文件"""
    basename = os.path.basename(pkl_file)
    
    try:
        # 读取数据
        l_packed_tokens = read_all_pickle_objects(pkl_file)
        
        # 分割数据
        splits = [l_packed_tokens[i:i + chunk_size] 
                  for i in range(0, len(l_packed_tokens), chunk_size)]
        
        # 准备参数列表
        save_tasks = []
        for idx, split in enumerate(splits):
            output_path = os.path.join(target_dir, basename.replace(".pkl", f"_{idx}.pkl"))
            save_tasks.append((split, output_path))
        
        return basename, save_tasks, len(l_packed_tokens)
    except Exception as e:
        return basename, None, f"Error: {str(e)}"

def optimized_pickle_processor(
    source_path="./CCI-DATA/",
    target_path="./CCI-DATA/packed_tokens_splits",
    done_files=None,
    chunk_size=100000,
    max_workers_io=4,      # IO线程数
    max_workers_cpu=None,  # CPU进程数，None表示使用CPU核心数
    compression_level=6
):
    """
    优化的pickle文件处理器
    
    Args:
        source_path: 源文件路径
        target_path: 目标路径
        done_files: 已完成的文件列表
        chunk_size: 每个分片的大小
        max_workers_io: IO操作的最大线程数
        max_workers_cpu: CPU密集型操作的最大进程数
        compression_level: 压缩级别 (1-9)
    """
    if done_files is None:
        done_files = []
    
    if max_workers_cpu is None:
        max_workers_cpu = min(mp.cpu_count(), 8)  # 限制最大进程数
    
    # 确保目标目录存在
    os.makedirs(target_path, exist_ok=True)
    
    # 获取所有pickle文件
    pkl_lists = glob.glob(os.path.join(source_path, "packed_tokens", "*.pkl"))
    
    print(f"Found {len(pkl_lists)} pickle files to process")
    print(f"Using {max_workers_io} IO threads and {max_workers_cpu} CPU processes")
    
    # 过滤已完成的文件
    remaining_files = [pkl for pkl in pkl_lists 
                      if os.path.basename(pkl) not in done_files]
    
    print(f"Remaining files to process: {len(remaining_files)}")
    
    if not remaining_files:
        print("All files already processed!")
        return
    
    # 阶段1: 并行读取和分割文件
    print("\nPhase 1: Reading and splitting files...")
    file_tasks = []
    
    with ThreadPoolExecutor(max_workers=max_workers_io) as executor:
        # 提交读取任务
        future_to_file = {
            executor.submit(process_single_pickle_file, pkl_file, target_path, chunk_size): pkl_file
            for pkl_file in remaining_files
        }
        
        # 收集结果
        with tqdm(total=len(remaining_files), desc="Reading files") as pbar:
            for future in as_completed(future_to_file):
                pkl_file = future_to_file[future]
                try:
                    basename, save_tasks, info = future.result()
                    if save_tasks is not None:
                        file_tasks.extend(save_tasks)
                        pbar.set_postfix_str(f"Loaded {basename}: {info} tokens")
                    else:
                        print(f"Failed to process {basename}: {info}")
                except Exception as e:
                    print(f"Error processing {pkl_file}: {str(e)}")
                pbar.update(1)
    
    if not file_tasks:
        print("No tasks to process!")
        return
    
    print(f"\nPhase 2: Compressing and saving {len(file_tasks)} splits...")
    
    # 阶段2: 并行压缩和保存
    # 使用ProcessPoolExecutor进行CPU密集型的压缩操作
    with ProcessPoolExecutor(max_workers=max_workers_cpu) as executor:
        # 提交压缩任务
        futures = [executor.submit(compress_and_save_split, task) for task in file_tasks]
        
        # 监控进度
        completed = 0
        with tqdm(total=len(file_tasks), desc="Compressing & saving") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if "Error" in result:
                        print(f"\n{result}")
                    completed += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"\nError in compression task: {str(e)}")
                    pbar.update(1)
    
    print(f"\nCompleted processing {completed} splits!")

def main():
    """主函数"""
    # 已完成的文件列表
    done_files = [
        # 在这里添加已完成的文件名
    ]
    
    # 运行优化版本
    optimized_pickle_processor(
        source_path="./CCI-DATA/",
        target_path="./CCI-DATA/packed_tokens_splits",
        done_files=done_files,
        chunk_size=100000,
        max_workers_io=4,      # 根据你的IO性能调整
        max_workers_cpu=None,  # 自动检测CPU核心数
        compression_level=6    # 平衡压缩率和速度
    )

if __name__ == "__main__":
    main()