#!/usr/bin/env python3
"""
多领域数据采样和打包工具
支持新闻、代码、数学数据的智能采样和高效打包
"""

import os
import json
import random
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SamplingConfig:
    """采样配置"""
    # 数据路径配置
    news_data_path: str = "./raw_data/news/"
    code_data_path: str = "./raw_data/code/"
    math_data_path: str = "./raw_data/math/"
    output_path: str = "./packed_data/"
    
    # 采样配置
    total_samples: int = 1_000_000  # 总样本数
    news_ratio: float = 0.60        # 新闻数据比例
    code_ratio: float = 0.25        # 代码数据比例
    math_ratio: float = 0.15        # 数学数据比例
    
    # 质量过滤配置
    min_length: int = 50            # 最小文本长度
    max_length: int = 8192          # 最大文本长度
    min_quality_score: float = 0.7  # 最小质量分数
    
    # 打包配置
    sequence_length: int = 2048     # 打包后的序列长度
    pack_samples: bool = True       # 是否进行序列打包
    samples_per_file: int = 10000   # 每个文件的样本数
    
    # 性能配置
    num_workers: int = 8            # 并行工作进程数
    chunk_size: int = 1000          # 每次处理的块大小
    
    # 随机种子
    random_seed: int = 42

@dataclass
class DataSample:
    """数据样本"""
    text: str
    domain: str
    source: str
    quality_score: float
    length: int
    hash_id: str

class QualityFilter:
    """数据质量过滤器"""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.bad_patterns = self._load_bad_patterns()
    
    def _load_bad_patterns(self) -> List[str]:
        """加载不良模式"""
        return [
            r'<[^>]+>',  # HTML标签
            r'http[s]?://\S+',  # URL
            r'\b\d{11,}\b',  # 长数字串（可能是ID）
            r'[^\w\s\u4e00-\u9fff]',  # 非中英文和常见标点
        ]
    
    def calculate_quality_score(self, text: str, domain: str) -> float:
        """计算质量分数"""
        score = 1.0
        
        # 长度检查
        if len(text) < self.config.min_length:
            return 0.0
        if len(text) > self.config.max_length:
            score -= 0.2
        
        # 字符质量检查
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        if domain == "news":
            # 新闻应该主要是中文或英文
            if ascii_ratio > 0.8:  # 主要是英文
                score -= 0.1
        elif domain == "code":
            # 代码应该主要是ASCII字符
            if ascii_ratio < 0.7:
                score -= 0.3
        
        # 重复字符检查
        max_repeat = max(len(list(g)) for k, g in 
                        __import__('itertools').groupby(text))
        if max_repeat > 10:
            score -= 0.2
        
        # 特殊模式检查
        import re
        for pattern in self.bad_patterns:
            matches = len(re.findall(pattern, text))
            if matches > 0:
                score -= min(0.1 * matches, 0.5)
        
        return max(0.0, min(1.0, score))
    
    def is_valid(self, sample: DataSample) -> bool:
        """判断样本是否有效"""
        return (sample.quality_score >= self.config.min_quality_score and
                self.config.min_length <= sample.length <= self.config.max_length)

class DataSampler:
    """数据采样器"""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.quality_filter = QualityFilter(config)
        self.seen_hashes = set()
        
        # 设置随机种子
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def _hash_text(self, text: str) -> str:
        """计算文本哈希值用于去重"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_domain_data(self, domain: str, data_path: str, 
                         target_samples: int) -> List[DataSample]:
        """加载特定领域的数据"""
        logger.info(f"加载{domain}数据从 {data_path}...")
        
        samples = []
        processed_count = 0
        
        # 遍历数据目录
        data_dir = Path(data_path)
        if not data_dir.exists():
            logger.warning(f"数据目录不存在: {data_path}")
            return []
        
        # 获取所有数据文件
        file_patterns = ['*.txt', '*.json', '*.jsonl']
        data_files = []
        for pattern in file_patterns:
            data_files.extend(data_dir.glob(pattern))
        
        if not data_files:
            logger.warning(f"在 {data_path} 中未找到数据文件")
            return []
        
        # 随机打乱文件顺序
        random.shuffle(data_files)
        
        for file_path in tqdm(data_files, desc=f"处理{domain}文件"):
            if len(samples) >= target_samples:
                break
                
            try:
                samples.extend(self._process_file(file_path, domain, 
                                                target_samples - len(samples)))
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        logger.info(f"{domain}数据加载完成，获得 {len(samples)} 个有效样本")
        return samples
    
    def _process_file(self, file_path: Path, domain: str, 
                     remaining_samples: int) -> List[DataSample]:
        """处理单个文件"""
        samples = []
        
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if len(samples) >= remaining_samples:
                                break
                            sample = self._create_sample(item, domain, str(file_path))
                            if sample and self._is_duplicate(sample):
                                samples.append(sample)
                    else:
                        sample = self._create_sample(data, domain, str(file_path))
                        if sample and not self._is_duplicate(sample):
                            samples.append(sample)
            
            elif file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(samples) >= remaining_samples:
                            break
                        try:
                            item = json.loads(line.strip())
                            sample = self._create_sample(item, domain, str(file_path))
                            if sample and not self._is_duplicate(sample):
                                samples.append(sample)
                        except json.JSONDecodeError:
                            continue
            
            elif file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        sample = self._create_sample({'text': content}, domain, str(file_path))
                        if sample and not self._is_duplicate(sample):
                            samples.append(sample)
        
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {e}")
        
        return samples
    
    def _create_sample(self, data: dict, domain: str, source: str) -> Optional[DataSample]:
        """创建数据样本"""
        # 提取文本内容
        text = None
        if isinstance(data, str):
            text = data
        elif 'text' in data:
            text = data['text']
        elif 'content' in data:
            text = data['content']
        elif 'code' in data:
            text = data['code']
        elif 'problem' in data and 'solution' in data:
            text = f"问题: {data['problem']}\n解答: {data['solution']}"
        
        if not text or not isinstance(text, str):
            return None
        
        # 基本清理
        text = text.strip()
        if not text:
            return None
        
        # 计算质量分数
        quality_score = self.quality_filter.calculate_quality_score(text, domain)
        
        # 创建样本
        sample = DataSample(
            text=text,
            domain=domain,
            source=source,
            quality_score=quality_score,
            length=len(text),
            hash_id=self._hash_text(text)
        )
        
        return sample if self.quality_filter.is_valid(sample) else None
    
    def _is_duplicate(self, sample: DataSample) -> bool:
        """检查是否重复"""
        if sample.hash_id in self.seen_hashes:
            return True
        self.seen_hashes.add(sample.hash_id)
        return False
    
    def sample_data(self) -> Dict[str, List[DataSample]]:
        """执行数据采样"""
        logger.info("开始数据采样...")
        
        # 计算各领域目标样本数
        news_samples = int(self.config.total_samples * self.config.news_ratio)
        code_samples = int(self.config.total_samples * self.config.code_ratio)
        math_samples = int(self.config.total_samples * self.config.math_ratio)
        
        logger.info(f"目标样本数 - 新闻: {news_samples}, 代码: {code_samples}, 数学: {math_samples}")
        
        # 并行加载各领域数据
        sampled_data = {}
        
        # 加载新闻数据
        sampled_data['news'] = self._load_domain_data(
            'news', self.config.news_data_path, news_samples
        )
        
        # 加载代码数据
        sampled_data['code'] = self._load_domain_data(
            'code', self.config.code_data_path, code_samples
        )
        
        # 加载数学数据
        sampled_data['math'] = self._load_domain_data(
            'math', self.config.math_data_path, math_samples
        )
        
        # 统计信息
        total_sampled = sum(len(samples) for samples in sampled_data.values())
        logger.info(f"采样完成，总计 {total_sampled} 个样本")
        
        return sampled_data

class DataPacker:
    """数据打包器"""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
    
    def pack_sequences(self, samples: List[DataSample]) -> List[Dict]:
        """将样本打包成固定长度序列"""
        logger.info("开始序列打包...")
        
        packed_data = []
        current_sequence = ""
        current_domains = []
        current_sources = []
        sequence_boundaries = []
        
        for sample in tqdm(samples, desc="打包序列"):
            # 添加领域标记
            domain_tag = f"<|{sample.domain}|>"
            sample_text = f"{domain_tag}{sample.text}<|end|>"
            
            # 检查是否需要开始新序列
            if len(current_sequence) + len(sample_text) > self.config.sequence_length:
                if current_sequence:  # 保存当前序列
                    packed_data.append({
                        'text': current_sequence,
                        'domains': list(set(current_domains)),
                        'sources': current_sources,
                        'boundaries': sequence_boundaries,
                        'length': len(current_sequence)
                    })
                
                # 开始新序列
                current_sequence = sample_text
                current_domains = [sample.domain]
                current_sources = [sample.source]
                sequence_boundaries = [len(sample_text)]
            else:
                # 添加到当前序列
                start_pos = len(current_sequence)
                current_sequence += sample_text
                current_domains.append(sample.domain)
                current_sources.append(sample.source)
                sequence_boundaries.append(len(current_sequence))
        
        # 保存最后一个序列
        if current_sequence:
            packed_data.append({
                'text': current_sequence,
                'domains': list(set(current_domains)),
                'sources': current_sources,
                'boundaries': sequence_boundaries,
                'length': len(current_sequence)
            })
        
        logger.info(f"打包完成，生成 {len(packed_data)} 个序列")
        return packed_data
    
    def save_packed_data(self, packed_data: List[Dict], output_dir: Path):
        """保存打包数据"""
        logger.info(f"保存打包数据到 {output_dir}...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 分文件保存
        num_files = (len(packed_data) + self.config.samples_per_file - 1) // self.config.samples_per_file
        
        for i in range(num_files):
            start_idx = i * self.config.samples_per_file
            end_idx = min((i + 1) * self.config.samples_per_file, len(packed_data))
            
            file_data = packed_data[start_idx:end_idx]
            file_path = output_dir / f"packed_data_{i:04d}.pkl"
            
            with open(file_path, 'wb') as f:
                pickle.dump(file_data, f)
            
            logger.info(f"保存文件 {file_path}, 包含 {len(file_data)} 个序列")
        
        # 保存元数据
        metadata = {
            'total_sequences': len(packed_data),
            'num_files': num_files,
            'samples_per_file': self.config.samples_per_file,
            'sequence_length': self.config.sequence_length,
            'config': asdict(self.config)
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已保存到 {metadata_path}")

class DataSamplingPipeline:
    """完整的数据采样和打包流水线"""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.sampler = DataSampler(config)
        self.packer = DataPacker(config)
    
    def run(self):
        """运行完整流水线"""
        logger.info("开始数据采样和打包流水线...")
        
        # 1. 采样数据
        sampled_data = self.sampler.sample_data()
        
        # 2. 合并所有样本
        all_samples = []
        for domain, samples in sampled_data.items():
            all_samples.extend(samples)
        
        # 3. 随机打乱
        random.shuffle(all_samples)
        
        # 4. 打包数据（如果启用）
        if self.config.pack_samples:
            packed_data = self.packer.pack_sequences(all_samples)
        else:
            # 不打包，直接转换格式
            packed_data = []
            for sample in all_samples:
                packed_data.append({
                    'text': f"<|{sample.domain}|>{sample.text}<|end|>",
                    'domains': [sample.domain],
                    'sources': [sample.source],
                    'boundaries': [len(sample.text)],
                    'length': sample.length
                })
        
        # 5. 保存数据
        output_dir = Path(self.config.output_path)
        self.packer.save_packed_data(packed_data, output_dir)
        
        # 6. 生成统计报告
        self._generate_report(sampled_data, packed_data, output_dir)
        
        logger.info("数据采样和打包完成！")
    
    def _generate_report(self, sampled_data: Dict, packed_data: List[Dict], 
                        output_dir: Path):
        """生成统计报告"""
        
        # 统计信息
        domain_stats = {}
        for domain, samples in sampled_data.items():
            domain_stats[domain] = {
                'count': len(samples),
                'avg_length': np.mean([s.length for s in samples]),
                'avg_quality': np.mean([s.quality_score for s in samples]),
                'total_chars': sum(s.length for s in samples)
            }
        
        report = {
            'sampling_config': asdict(self.config),
            'domain_statistics': domain_stats,
            'packed_statistics': {
                'total_sequences': len(packed_data),
                'avg_sequence_length': np.mean([d['length'] for d in packed_data]),
                'total_characters': sum(d['length'] for d in packed_data)
            },
            'quality_distribution': self._calculate_quality_distribution(sampled_data)
        }
        
        # 保存报告
        report_path = output_dir / "sampling_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 数据采样和打包报告")
        print("="*60)
        print(f"总样本数: {sum(stats['count'] for stats in domain_stats.values()):,}")
        print(f"总序列数: {len(packed_data):,}")
        print(f"总字符数: {report['packed_statistics']['total_characters']:,}")
        print("\n各领域统计:")
        for domain, stats in domain_stats.items():
            print(f"  {domain}: {stats['count']:,} 样本, "
                  f"平均长度: {stats['avg_length']:.1f}, "
                  f"平均质量: {stats['avg_quality']:.3f}")
        print("="*60)
        
        logger.info(f"详细报告已保存到 {report_path}")
    
    def _calculate_quality_distribution(self, sampled_data: Dict) -> Dict:
        """计算质量分布"""
        all_scores = []
        for samples in sampled_data.values():
            all_scores.extend([s.quality_score for s in samples])
        
        return {
            'mean': float(np.mean(all_scores)),
            'std': float(np.std(all_scores)),
            'min': float(np.min(all_scores)),
            'max': float(np.max(all_scores)),
            'percentiles': {
                '25': float(np.percentile(all_scores, 25)),
                '50': float(np.percentile(all_scores, 50)),
                '75': float(np.percentile(all_scores, 75)),
                '90': float(np.percentile(all_scores, 90)),
                '95': float(np.percentile(all_scores, 95)),
            }
        }

def main():
    """主函数"""
    
    # 配置采样参数
    config = SamplingConfig(
        # 数据路径 - 请根据实际情况修改
        news_data_path="./raw_data/news/",
        code_data_path="./raw_data/code/",
        math_data_path="./raw_data/math/",
        output_path="./packed_data/",
        
        # 采样配置
        total_samples=100_000,  # 先用小规模测试
        news_ratio=0.60,
        code_ratio=0.25,
        math_ratio=0.15,
        
        # 序列配置
        sequence_length=2048,
        pack_samples=True,
        samples_per_file=5000,
        
        # 性能配置
        num_workers=mp.cpu_count(),
        chunk_size=1000,
    )
    
    # 运行流水线
    pipeline = DataSamplingPipeline(config)
    
    try:
        pipeline.run()
        print("✅ 数据采样和打包成功完成！")
    except Exception as e:
        logger.error(f"流水线执行失败: {e}")
        raise

if __name__ == "__main__":
    main()