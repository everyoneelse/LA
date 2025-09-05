import pickle
import os
import glob
from tqdm import tqdm
import numpy as np
import zlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

def read_all_pickle_objects(file_path):
    all_data = []
    with open(file_path, 'rb') as f:
        while True:
            try:
                data_chunk = pickle.load(f)
                all_data.extend(data_chunk)
            except EOFError:
                break
    return all_data

def process_and_save_split(args):
    """处理并保存单个分片"""
    split_data, output_path = args
    
    # 检查文件是否已存在
    if os.path.exists(output_path):
        return f"Skipped: {os.path.basename(output_path)}"
    
    try:
        # 压缩存储
        compressed_data = zlib.compress(
            pickle.dumps(split_data, protocol=pickle.HIGHEST_PROTOCOL)
        )
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        return f"Saved: {os.path.basename(output_path)}"
    except Exception as e:
        return f"Error: {output_path} - {str(e)}"

# 配置
path = r"./CCI-DATA/"
tar = "./CCI-DATA/packed_tokens_splits"
pkl_lists = glob.glob(os.path.join(path, "packed_tokens","*.pkl"))

done = [
    # 在这里添加已完成的文件
]

# 确保目标目录存在
os.makedirs(tar, exist_ok=True)

# 获取CPU核心数，用于进程池
num_cores = mp.cpu_count()
print(f"Using {num_cores} CPU cores for parallel processing")

for pkl in tqdm(pkl_lists, desc="Processing files"):
    basename = os.path.basename(pkl)
    if basename in done:
        continue
    
    print(f"\nProcessing: {basename}")
    
    # 读取数据
    l_packed_tokens = read_all_pickle_objects(pkl)
    
    chunk_size = 100000
    splits = [l_packed_tokens[i:i + chunk_size] 
              for i in range(0, len(l_packed_tokens), chunk_size)]
    
    # 准备任务列表
    tasks = []
    for idx, split in enumerate(splits):
        output_path = os.path.join(tar, basename.replace(".pkl", f"_{idx}.pkl"))
        tasks.append((split, output_path))
    
    # 使用进程池并行处理压缩和保存
    # 进程池适合CPU密集型的压缩任务
    with ProcessPoolExecutor(max_workers=min(num_cores, len(tasks))) as executor:
        futures = [executor.submit(process_and_save_split, task) for task in tasks]
        
        # 显示进度
        results = []
        for future in tqdm(as_completed(futures), 
                          total=len(futures), 
                          desc=f"Saving splits for {basename}"):
            result = future.result()
            results.append(result)
            
    # 打印结果摘要
    saved_count = sum(1 for r in results if r.startswith("Saved"))
    skipped_count = sum(1 for r in results if r.startswith("Skipped"))
    error_count = sum(1 for r in results if r.startswith("Error"))
    
    print(f"  Saved: {saved_count}, Skipped: {skipped_count}, Errors: {error_count}")
    
    if error_count > 0:
        print("  Errors:")
        for r in results:
            if r.startswith("Error"):
                print(f"    {r}")

print("\nAll files processed!")