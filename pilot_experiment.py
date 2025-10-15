#!/usr/bin/env python3
"""
小规模Pipeline验证实验
用于验证多领域预训练数据处理流程
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据配置
    total_tokens: int = 100_000_000  # 100M tokens
    news_ratio: float = 0.60
    code_ratio: float = 0.25
    math_ratio: float = 0.15
    
    # 模型配置
    model_name: str = "microsoft/DialoGPT-small"  # 125M参数基线模型
    max_length: int = 512
    
    # 训练配置
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_train_steps: int = 50_000
    eval_steps: int = 1000
    save_steps: int = 5000
    
    # 输出路径
    output_dir: str = "./pilot_results"
    data_dir: str = "./pilot_data"

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_sample_data(self) -> Dict[str, List[str]]:
        """创建样本数据 - 模拟真实数据"""
        
        # 新闻样本
        news_samples = [
            "今日股市收盘，上证指数上涨2.3%，创下近期新高。分析师认为，这主要得益于科技股的强劲表现。",
            "气象部门发布暴雨预警，提醒市民注意出行安全。预计降雨将持续到明天上午。",
            "新能源汽车销量再创新高，同比增长45%。业内专家表示，这反映了消费者对环保出行的重视。",
            "教育部发布新政策，将进一步推进义务教育均衡发展，缩小城乡教育差距。",
            "人工智能技术在医疗领域的应用取得重大突破，诊断准确率提升至95%以上。"
        ] * 1000  # 扩展样本
        
        # 代码样本
        code_samples = [
            """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 计算前10个斐波那契数
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")""",
            
            """class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)""",
                
            """import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('正弦函数图像')
plt.legend()
plt.grid(True)
plt.show()""",
            
            """async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"HTTP {response.status}")

# 并发请求示例
async def main():
    urls = ["http://api1.com", "http://api2.com"]
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results""",
            
            """SELECT u.name, u.email, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2024-01-01'
GROUP BY u.id, u.name, u.email
HAVING COUNT(o.id) > 5
ORDER BY order_count DESC;"""
        ] * 200  # 扩展样本
        
        # 数学样本
        math_samples = [
            """问题：求解方程 2x + 3 = 11
解：
2x + 3 = 11
2x = 11 - 3
2x = 8
x = 4

验证：2(4) + 3 = 8 + 3 = 11 ✓""",

            """定理：勾股定理
对于直角三角形，如果两条直角边的长度分别为a和b，斜边长度为c，则：
a² + b² = c²

证明：
设直角三角形ABC，∠C = 90°，BC = a，AC = b，AB = c
在边长为(a+b)的正方形中构造四个全等的直角三角形...
通过面积关系可得：a² + b² = c²""",

            """积分计算：
∫(2x + 1)dx = x² + x + C

验证：
d/dx(x² + x + C) = 2x + 1 ✓

定积分：
∫₀² (2x + 1)dx = [x² + x]₀² = (4 + 2) - (0 + 0) = 6""",

            """概率问题：
掷两个公平骰子，求点数和为7的概率。

解：
总的可能结果：6 × 6 = 36种
和为7的情况：(1,6), (2,5), (3,4), (4,3), (5,2), (6,1) = 6种
概率 P(和=7) = 6/36 = 1/6 ≈ 0.167""",

            """矩阵运算：
设 A = [1 2; 3 4], B = [5 6; 7 8]

A + B = [1+5 2+6; 3+7 4+8] = [6 8; 10 12]

A × B = [1×5+2×7 1×6+2×8; 3×5+4×7 3×6+4×8] = [19 22; 43 50]"""
        ] * 300  # 扩展样本
        
        return {
            "news": news_samples,
            "code": code_samples,
            "math": math_samples
        }
    
    def prepare_datasets(self) -> DatasetDict:
        """准备训练数据集"""
        
        # 创建样本数据
        sample_data = self.create_sample_data()
        
        # 计算各领域的样本数量
        total_samples = sum(len(samples) for samples in sample_data.values())
        
        news_count = int(len(sample_data["news"]) * self.config.news_ratio)
        code_count = int(len(sample_data["code"]) * self.config.code_ratio)
        math_count = int(len(sample_data["math"]) * self.config.math_ratio)
        
        # 随机采样
        random.shuffle(sample_data["news"])
        random.shuffle(sample_data["code"])
        random.shuffle(sample_data["math"])
        
        selected_news = sample_data["news"][:news_count]
        selected_code = sample_data["code"][:code_count]
        selected_math = sample_data["math"][:math_count]
        
        # 创建数据集
        datasets = {}
        
        for domain, texts in [("news", selected_news), ("code", selected_code), ("math", selected_math)]:
            # 标记领域信息
            formatted_texts = [f"[{domain.upper()}] {text}" for text in texts]
            
            # 分割训练集和验证集
            split_idx = int(len(formatted_texts) * 0.9)
            train_texts = formatted_texts[:split_idx]
            val_texts = formatted_texts[split_idx:]
            
            datasets[f"{domain}_train"] = Dataset.from_dict({"text": train_texts})
            datasets[f"{domain}_val"] = Dataset.from_dict({"text": val_texts})
        
        # 合并训练集和验证集
        train_dataset = concatenate_datasets([
            datasets["news_train"],
            datasets["code_train"], 
            datasets["math_train"]
        ])
        
        val_dataset = concatenate_datasets([
            datasets["news_val"],
            datasets["code_val"],
            datasets["math_val"]
        ])
        
        # 随机打乱
        train_dataset = train_dataset.shuffle(seed=42)
        val_dataset = val_dataset.shuffle(seed=42)
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    
    def tokenize_function(self, examples):
        """分词函数"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.config.max_length,
            return_special_tokens_mask=True,
        )

class PilotTrainer:
    """试点训练器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.processor = DataProcessor(config)
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    
    def run_experiment(self):
        """运行实验"""
        logger.info("开始小规模Pipeline验证实验...")
        
        # 1. 准备数据
        logger.info("准备数据集...")
        datasets = self.processor.prepare_datasets()
        
        # 分词
        tokenized_datasets = datasets.map(
            self.processor.tokenize_function,
            batched=True,
            remove_columns=datasets["train"].column_names,
        )
        
        # 2. 加载模型
        logger.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        # 3. 设置训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=500,
            logging_steps=100,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to=None,  # 不使用wandb等
        )
        
        # 4. 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.processor.tokenizer,
            mlm=False,
        )
        
        # 5. 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
        )
        
        # 6. 开始训练
        logger.info("开始训练...")
        train_result = trainer.train()
        
        # 7. 保存结果
        logger.info("保存模型和结果...")
        trainer.save_model()
        
        # 保存tokenizer
        logger.info("保存tokenizer...")
        self.processor.tokenizer.save_pretrained(self.config.output_dir)
        
        # 8. 评估
        logger.info("评估模型...")
        eval_results = trainer.evaluate()
        
        # 9. 生成报告
        self.generate_report(train_result, eval_results, datasets)
        
        logger.info("实验完成!")
        return train_result, eval_results
    
    def generate_report(self, train_result, eval_results, datasets):
        """生成实验报告"""
        
        report = {
            "experiment_config": {
                "total_tokens": self.config.total_tokens,
                "domain_ratios": {
                    "news": self.config.news_ratio,
                    "code": self.config.code_ratio,
                    "math": self.config.math_ratio,
                },
                "model_name": self.config.model_name,
                "max_length": self.config.max_length,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
            "data_statistics": {
                "train_samples": len(datasets["train"]),
                "val_samples": len(datasets["validation"]),
                "total_samples": len(datasets["train"]) + len(datasets["validation"]),
            },
            "training_results": {
                "final_train_loss": train_result.training_loss,
                "train_runtime": train_result.training_loss,
                "train_steps": train_result.global_step,
            },
            "evaluation_results": eval_results,
        }
        
        # 保存报告
        report_path = Path(self.config.output_dir) / "experiment_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"实验报告已保存至: {report_path}")
        
        # 打印关键结果
        print("\n" + "="*50)
        print("实验结果摘要")
        print("="*50)
        print(f"训练样本数: {report['data_statistics']['train_samples']:,}")
        print(f"验证样本数: {report['data_statistics']['val_samples']:,}")
        print(f"最终训练损失: {train_result.training_loss:.4f}")
        print(f"最终验证损失: {eval_results['eval_loss']:.4f}")
        print(f"训练步数: {train_result.global_step:,}")
        print("="*50)

def main():
    """主函数"""
    config = ExperimentConfig()
    trainer = PilotTrainer(config)
    
    try:
        train_result, eval_results = trainer.run_experiment()
        print("✅ 实验成功完成!")
    except Exception as e:
        logger.error(f"实验失败: {e}")
        raise

if __name__ == "__main__":
    main()