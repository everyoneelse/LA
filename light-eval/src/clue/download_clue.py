#!/usr/bin/env python3
"""
CLUE数据集下载脚本
支持下载CLUE benchmark的所有任务数据
"""

import os
import json
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse

# CLUE任务列表和对应的下载链接
CLUE_TASKS = {
    'afqmc': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip',
        'desc': '蚂蚁金融语义相似度'
    },
    'tnews': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip', 
        'desc': '今日头条中文新闻分类（短文本）'
    },
    'iflytek': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip',
        'desc': '科大讯飞长文本分类'
    },
    'cmnli': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/cmnli_public.zip',
        'desc': '中文自然语言推理'
    },
    'wsc': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/cluewsc2020_public.zip',
        'desc': '中文指代消解'
    },
    'csl': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/csl_public.zip',
        'desc': '中文科学文献关键词识别'
    },
    'chid': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/chid_public.zip',
        'desc': '成语阅读理解'
    },
    'c3': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/c3_public.zip',
        'desc': '中文多选阅读理解'
    },
    'ocnli': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/ocnli_public.zip',
        'desc': '原生中文自然语言推理'
    },
    'cmrc': {
        'url': 'https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip',
        'desc': '中文阅读理解'
    }
}

def download_file(url, dest_path):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))

def extract_archive(file_path, extract_to):
    """解压文件"""
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.endswith('.tar.gz'):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {file_path}")

def download_clue_task(task_name, data_dir):
    """下载单个CLUE任务数据"""
    if task_name not in CLUE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(CLUE_TASKS.keys())}")
    
    task_info = CLUE_TASKS[task_name]
    print(f"\n下载任务: {task_name} - {task_info['desc']}")
    
    # 创建任务目录
    task_dir = Path(data_dir) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载文件
    archive_name = task_info['url'].split('/')[-1]
    archive_path = task_dir / archive_name
    
    if archive_path.exists():
        print(f"文件已存在: {archive_path}")
    else:
        print(f"下载中: {task_info['url']}")
        download_file(task_info['url'], archive_path)
    
    # 解压文件
    print(f"解压中: {archive_path}")
    extract_archive(str(archive_path), str(task_dir))
    
    # 删除压缩包
    archive_path.unlink()
    print(f"完成: {task_name}")
    
    return task_dir

def download_all_clue_tasks(data_dir):
    """下载所有CLUE任务数据"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载CLUE数据集到: {data_dir}")
    
    for task_name in CLUE_TASKS:
        try:
            download_clue_task(task_name, data_dir)
        except Exception as e:
            print(f"下载 {task_name} 失败: {e}")
            continue
    
    print("\n所有任务下载完成!")

def main():
    parser = argparse.ArgumentParser(description='下载CLUE benchmark数据集')
    parser.add_argument('--data_dir', type=str, default='data/clue',
                        help='数据保存目录')
    parser.add_argument('--task', type=str, default='all',
                        help='要下载的任务名称，默认下载所有任务')
    
    args = parser.parse_args()
    
    if args.task == 'all':
        download_all_clue_tasks(args.data_dir)
    else:
        download_clue_task(args.task, args.data_dir)

if __name__ == '__main__':
    main()