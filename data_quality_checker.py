#!/usr/bin/env python3
"""
数据质量检查工具
用于分析和验证采样后数据的质量
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, packed_data_dir: str):
        self.data_dir = Path(packed_data_dir)
        self.metadata = self._load_metadata()
        self.sample_data = self._load_sample_data()
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_sample_data(self, max_samples: int = 1000) -> List[Dict]:
        """加载样本数据用于分析"""
        sample_data = []
        data_files = list(self.data_dir.glob("packed_data_*.pkl"))
        
        if not data_files:
            logger.warning("未找到数据文件")
            return []
        
        # 随机选择文件
        selected_files = random.sample(data_files, min(3, len(data_files)))
        
        for file_path in selected_files:
            try:
                with open(file_path, 'rb') as f:
                    sequences = pickle.load(f)
                    sample_data.extend(random.sample(sequences, 
                                                   min(max_samples // len(selected_files), 
                                                       len(sequences))))
            except Exception as e:
                logger.error(f"加载文件 {file_path} 失败: {e}")
        
        logger.info(f"加载了 {len(sample_data)} 个样本用于质量检查")
        return sample_data
    
    def check_domain_distribution(self) -> Dict:
        """检查领域分布"""
        domain_counts = Counter()
        domain_lengths = defaultdict(list)
        
        for sample in self.sample_data:
            domains = sample.get('domains', [])
            length = sample.get('length', 0)
            
            for domain in domains:
                domain_counts[domain] += 1
                domain_lengths[domain].append(length)
        
        # 计算统计信息
        domain_stats = {}
        for domain, count in domain_counts.items():
            lengths = domain_lengths[domain]
            domain_stats[domain] = {
                'count': count,
                'percentage': count / len(self.sample_data) * 100,
                'avg_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'min_length': np.min(lengths),
                'max_length': np.max(lengths)
            }
        
        return domain_stats
    
    def check_text_quality(self) -> Dict:
        """检查文本质量"""
        quality_metrics = {
            'avg_length': [],
            'char_diversity': [],
            'word_diversity': [],
            'special_char_ratio': [],
            'digit_ratio': [],
            'uppercase_ratio': [],
            'line_count': [],
            'empty_lines': []
        }
        
        for sample in self.sample_data:
            text = sample.get('text', '')
            
            # 基本统计
            quality_metrics['avg_length'].append(len(text))
            quality_metrics['char_diversity'].append(len(set(text)))
            
            # 词汇多样性
            words = re.findall(r'\b\w+\b', text.lower())
            quality_metrics['word_diversity'].append(len(set(words)) / max(len(words), 1))
            
            # 字符类型比例
            special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))
            digits = len(re.findall(r'\d', text))
            uppercase = len(re.findall(r'[A-Z]', text))
            
            quality_metrics['special_char_ratio'].append(special_chars / max(len(text), 1))
            quality_metrics['digit_ratio'].append(digits / max(len(text), 1))
            quality_metrics['uppercase_ratio'].append(uppercase / max(len(text), 1))
            
            # 行统计
            lines = text.split('\n')
            quality_metrics['line_count'].append(len(lines))
            quality_metrics['empty_lines'].append(sum(1 for line in lines if not line.strip()))
        
        # 计算汇总统计
        summary = {}
        for metric, values in quality_metrics.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return summary
    
    def check_domain_tags(self) -> Dict:
        """检查领域标签使用情况"""
        tag_patterns = {
            'news': r'<\|news\|>',
            'code': r'<\|code\|>',
            'math': r'<\|math\|>',
            'end': r'<\|end\|>'
        }
        
        tag_counts = Counter()
        tag_positions = defaultdict(list)
        
        for sample in self.sample_data:
            text = sample.get('text', '')
            
            for tag_name, pattern in tag_patterns.items():
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                tag_counts[tag_name] += len(matches)
                
                for match in matches:
                    tag_positions[tag_name].append(match.start() / len(text))
        
        return {
            'tag_counts': dict(tag_counts),
            'tag_positions': {k: {
                'mean_position': np.mean(v) if v else 0,
                'std_position': np.std(v) if v else 0
            } for k, v in tag_positions.items()}
        }
    
    def detect_anomalies(self) -> List[Dict]:
        """检测异常样本"""
        anomalies = []
        
        for idx, sample in enumerate(self.sample_data):
            text = sample.get('text', '')
            length = len(text)
            
            # 检查异常长度
            if length < 10:
                anomalies.append({
                    'index': idx,
                    'type': 'too_short',
                    'value': length,
                    'text_preview': text[:100]
                })
            
            # 检查重复字符
            max_repeat = max(len(list(g)) for k, g in 
                           __import__('itertools').groupby(text)) if text else 0
            if max_repeat > 50:
                anomalies.append({
                    'index': idx,
                    'type': 'excessive_repetition',
                    'value': max_repeat,
                    'text_preview': text[:100]
                })
            
            # 检查编码问题
            try:
                text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                anomalies.append({
                    'index': idx,
                    'type': 'encoding_error',
                    'text_preview': text[:100]
                })
            
            # 检查空白内容
            if not text.strip():
                anomalies.append({
                    'index': idx,
                    'type': 'empty_content',
                    'text_preview': repr(text[:100])
                })
        
        return anomalies
    
    def generate_quality_report(self) -> Dict:
        """生成完整的质量报告"""
        logger.info("生成数据质量报告...")
        
        report = {
            'metadata': self.metadata,
            'sample_size': len(self.sample_data),
            'domain_distribution': self.check_domain_distribution(),
            'text_quality': self.check_text_quality(),
            'domain_tags': self.check_domain_tags(),
            'anomalies': self.detect_anomalies()
        }
        
        # 保存报告
        report_path = self.data_dir / "quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"质量报告已保存至: {report_path}")
        return report
    
    def visualize_quality_metrics(self, report: Dict):
        """可视化质量指标"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('数据质量分析报告', fontsize=16, fontweight='bold')
        
        # 1. 领域分布饼图
        ax = axes[0, 0]
        domain_dist = report['domain_distribution']
        if domain_dist:
            domains = list(domain_dist.keys())
            counts = [stats['count'] for stats in domain_dist.values()]
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(domains)]
            
            ax.pie(counts, labels=domains, autopct='%1.1f%%', colors=colors)
            ax.set_title('领域分布')
        
        # 2. 文本长度分布直方图
        ax = axes[0, 1]
        lengths = [sample.get('length', 0) for sample in self.sample_data]
        ax.hist(lengths, bins=50, alpha=0.7, color='skyblue')
        ax.set_xlabel('文本长度')
        ax.set_ylabel('频次')
        ax.set_title('文本长度分布')
        ax.axvline(np.mean(lengths), color='red', linestyle='--', 
                  label=f'平均值: {np.mean(lengths):.0f}')
        ax.legend()
        
        # 3. 各领域平均长度对比
        ax = axes[0, 2]
        if domain_dist:
            domains = list(domain_dist.keys())
            avg_lengths = [stats['avg_length'] for stats in domain_dist.values()]
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'][:len(domains)]
            
            bars = ax.bar(domains, avg_lengths, color=colors, alpha=0.7)
            ax.set_ylabel('平均长度')
            ax.set_title('各领域平均文本长度')
            
            # 添加数值标签
            for bar, length in zip(bars, avg_lengths):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                       f'{length:.0f}', ha='center', va='bottom')
        
        # 4. 质量指标箱线图
        ax = axes[1, 0]
        quality_data = []
        quality_labels = []
        
        text_quality = report['text_quality']
        for metric in ['word_diversity', 'special_char_ratio', 'digit_ratio']:
            if metric in text_quality:
                # 重新计算原始数据用于箱线图
                values = []
                for sample in self.sample_data:
                    text = sample.get('text', '')
                    if metric == 'word_diversity':
                        words = re.findall(r'\b\w+\b', text.lower())
                        values.append(len(set(words)) / max(len(words), 1))
                    elif metric == 'special_char_ratio':
                        special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))
                        values.append(special_chars / max(len(text), 1))
                    elif metric == 'digit_ratio':
                        digits = len(re.findall(r'\d', text))
                        values.append(digits / max(len(text), 1))
                
                quality_data.append(values)
                quality_labels.append(metric.replace('_', ' ').title())
        
        if quality_data:
            ax.boxplot(quality_data, labels=quality_labels)
            ax.set_ylabel('比例/分数')
            ax.set_title('文本质量指标分布')
            ax.tick_params(axis='x', rotation=45)
        
        # 5. 标签使用统计
        ax = axes[1, 1]
        tag_counts = report['domain_tags']['tag_counts']
        if tag_counts:
            tags = list(tag_counts.keys())
            counts = list(tag_counts.values())
            colors = ['gold', 'lightcoral', 'lightblue', 'lightgreen'][:len(tags)]
            
            bars = ax.bar(tags, counts, color=colors, alpha=0.7)
            ax.set_ylabel('使用次数')
            ax.set_title('领域标签使用统计')
            
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                       f'{count}', ha='center', va='bottom')
        
        # 6. 异常检测结果
        ax = axes[1, 2]
        anomalies = report['anomalies']
        if anomalies:
            anomaly_types = Counter(a['type'] for a in anomalies)
            types = list(anomaly_types.keys())
            counts = list(anomaly_types.values())
            colors = ['red', 'orange', 'yellow', 'pink'][:len(types)]
            
            bars = ax.bar(types, counts, color=colors, alpha=0.7)
            ax.set_ylabel('异常数量')
            ax.set_title('异常类型统计')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                       f'{count}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, '未发现异常', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('异常检测结果')
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = self.data_dir / "quality_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"质量分析图已保存至: {plot_path}")
        
        plt.show()
    
    def print_summary(self, report: Dict):
        """打印质量摘要"""
        print("\n" + "="*60)
        print("📊 数据质量检查摘要")
        print("="*60)
        
        # 基本信息
        print(f"样本数量: {report['sample_size']:,}")
        if 'total_sequences' in report['metadata']:
            print(f"总序列数: {report['metadata']['total_sequences']:,}")
        
        # 领域分布
        print("\n🏷️  领域分布:")
        domain_dist = report['domain_distribution']
        for domain, stats in domain_dist.items():
            print(f"  {domain}: {stats['count']} 样本 ({stats['percentage']:.1f}%)")
            print(f"    平均长度: {stats['avg_length']:.0f} ± {stats['std_length']:.0f}")
        
        # 文本质量
        print("\n📝 文本质量:")
        text_quality = report['text_quality']
        print(f"  平均长度: {text_quality['avg_length']['mean']:.0f}")
        print(f"  字符多样性: {text_quality['char_diversity']['mean']:.0f}")
        print(f"  词汇多样性: {text_quality['word_diversity']['mean']:.3f}")
        
        # 异常检测
        anomalies = report['anomalies']
        print(f"\n⚠️  异常检测: 发现 {len(anomalies)} 个异常")
        if anomalies:
            anomaly_types = Counter(a['type'] for a in anomalies)
            for atype, count in anomaly_types.items():
                print(f"  {atype}: {count} 个")
        
        # 标签使用
        tag_counts = report['domain_tags']['tag_counts']
        print(f"\n🏷️  标签使用:")
        for tag, count in tag_counts.items():
            print(f"  {tag}: {count} 次")
        
        print("="*60)

def main():
    """主函数"""
    
    # 检查数据目录
    data_dir = "./packed_data/"
    if not Path(data_dir).exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请先运行数据采样脚本生成打包数据")
        return
    
    # 创建质量检查器
    checker = DataQualityChecker(data_dir)
    
    try:
        # 生成质量报告
        report = checker.generate_quality_report()
        
        # 打印摘要
        checker.print_summary(report)
        
        # 生成可视化图表
        checker.visualize_quality_metrics(report)
        
        print("✅ 数据质量检查完成!")
        print(f"详细报告: {data_dir}/quality_report.json")
        print(f"可视化图表: {data_dir}/quality_analysis.png")
        
    except Exception as e:
        logger.error(f"质量检查失败: {e}")
        raise

if __name__ == "__main__":
    main()