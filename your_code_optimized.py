import pickle
import os
import glob
from tqdm import tqdm
import numpy as np
import zlib
from concurrent.futures import ProcessPoolExecutor
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

def save_compressed_split(split_data, output_path):
    """保存单个压缩分片"""
    if os.path.exists(output_path):
        return False  # 文件已存在，跳过
    
    # 压缩存储
    compressed_data = zlib.compress(pickle.dumps(split_data, protocol=pickle.HIGHEST_PROTOCOL))
    with open(output_path, 'wb') as f:
        f.write(compressed_data)
    return True  # 成功保存

path = r"./CCI-DATA/"
tar = "./CCI-DATA/packed_tokens_splits"
pkl_lists = glob.glob(os.path.join(path, "packed_tokens","*.pkl"))

done = [
# "part_6e0dfb93869a9a9ef1cc258e601fb23b.pkl",
# "part_be034f07a60fbdc2bffcbcbd79994727.pkl",
# "part_562e040f6df30cbfdf836fd41692c14e.pkl",
# "part_27372e9ac505e19dda30e31c0e6debb5.pkl",
# "part_1466e9241f2dbdcfbd0e8bec13983ab6.pkl",
# "part_012a3e95511ec5a43048722d199a67a8.pkl",
# "part_4e4f5192738136963fcf0a2e3f840d75.pkl",
# "part_828989e43970b4ecbf9bbba02d25a593.pkl",
# "part_c55d8327c9f81dd0670c52081a729d0a.pkl",
# "part_f761121a59ea418d0a4699b588bebe23.pkl",
# "part_83204ccaabcb70d318ba8032d65dae1f.pkl",
# "part_c2c111954ca753460b474c33dd3a2226.pkl",
# "part_6b81209c31bc2061b93b42b8d12f3a5e.pkl",
]

# 获取CPU核心数
max_workers = min(mp.cpu_count(), 8)  # 限制最大进程数为8
print(f"Using {max_workers} processes for parallel compression")

for pkl in tqdm(pkl_lists):
    basename = os.path.basename(pkl)
    if basename in done:
        continue
        
    l_packed_tokens = read_all_pickle_objects(pkl)
    
    chunk_size = 100000
    splits = [l_packed_tokens[i:i + chunk_size] 
              for i in range(0, len(l_packed_tokens), chunk_size)]
    
    # 使用进程池并行处理压缩和保存
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = []
        for idx, split in enumerate(splits):
            output_path = os.path.join(tar, basename.replace(".pkl", f"_{idx}.pkl"))
            future = executor.submit(save_compressed_split, split, output_path)
            futures.append(future)
        
        # 等待所有任务完成，显示进度
        saved_count = 0
        for future in tqdm(futures, desc=f"Processing {basename}"):
            if future.result():  # 如果返回True表示成功保存
                saved_count += 1
        
        print(f"Saved {saved_count} new splits for {basename}")