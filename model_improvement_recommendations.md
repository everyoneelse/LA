# 预训练模型改进建议

## 当前问题分析

您遇到的现象**完全正常**。这是预训练模型的典型行为：

1. **数据模式学习**: 模型在中文新闻数据上训练，学会了新闻文本的模式
2. **上下文延续**: 在新闻中，联系方式后通常跟着更多相关信息
3. **缺乏停止信号**: 预训练模型没有学会在特定内容后主动停止

## 立即可用的解决方案

### 方案1: 后处理提取（推荐）✅

```python
import re

def clean_phone_generation(generated_text):
    """清理手机号生成结果"""
    phone_pattern = r'1[3-9]\d{9}'  # 中国手机号模式
    match = re.search(phone_pattern, generated_text)
    if match:
        return generated_text[:match.end()]
    return generated_text

# 使用示例
results = model.generate(["卢经理 联系方式:"], None, max_gen_len=15, temperature=1, top_p=0.6)
clean_result = clean_phone_generation(results[0])
print(clean_result)  # 输出: "13974628606"
```

### 方案2: 调整生成参数

```python
# 更保守的参数设置
results = model.generate(
    ["卢经理 联系方式:"], 
    None,
    max_gen_len=12,        # 减少最大生成长度
    temperature=0.3,       # 降低随机性
    top_p=0.8,            # 调整采样范围
    additional_stop_symbols=['\n', '  ', '。', '，']  # 添加停止符号
)
```

### 方案3: 多次生成选择

```python
def generate_best_phone(prompt, model, attempts=3):
    """多次生成，选择最佳结果"""
    results = []
    for _ in range(attempts):
        gen = model.generate([prompt], None, max_gen_len=15, temperature=1, top_p=0.6)
        clean = clean_phone_generation(gen[0])
        if re.match(r'1[3-9]\d{9}$', clean):  # 只包含手机号
            return clean
        results.append(clean)
    
    # 返回最短的结果
    return min(results, key=len)
```

## 长期改进方案

### 1. 指令微调 (Instruction Tuning)

准备专门的训练数据：

```json
[
  {
    "instruction": "根据提示生成联系方式，只输出手机号",
    "input": "卢经理 联系方式:",
    "output": "13974628606"
  },
  {
    "instruction": "提供联系电话",
    "input": "张总监 手机号码:",
    "output": "13812345678"
  }
]
```

使用这些数据进行微调，教会模型在合适位置停止。

### 2. 强化学习人类反馈 (RLHF)

1. **收集人类反馈**: 对生成结果进行评分
2. **训练奖励模型**: 学习什么是好的输出
3. **PPO训练**: 使用强化学习优化生成策略

### 3. 提示工程优化

```python
# 更明确的提示格式
prompts = [
    "请只输出手机号码：卢经理联系方式",
    "联系电话（仅数字）：张经理",
    "手机号：李总"
]
```

### 4. 数据增强

在现有训练数据中添加：
- 明确的停止标记
- 结构化的联系方式数据
- 各种格式的手机号样本

## 模型训练建议

### 1. 数据质量改进

```python
# 添加特殊标记的训练数据
training_samples = [
    "联系人：张经理 手机：13812345678<|endoftext|>",
    "客服电话：13987654321<|stop|>",
    "销售热线：13765432109\n"
]
```

### 2. 损失函数调整

```python
# 在特定位置增加停止损失
def custom_loss_with_stop_penalty(logits, labels, stop_positions):
    base_loss = F.cross_entropy(logits, labels)
    
    # 在应该停止的位置增加停止token的概率
    stop_loss = 0
    for pos in stop_positions:
        stop_logits = logits[:, pos, stop_token_id]
        stop_loss += -torch.log_softmax(stop_logits, dim=-1).mean()
    
    return base_loss + 0.1 * stop_loss
```

### 3. 评估指标

```python
def evaluate_phone_extraction(model, test_prompts):
    """评估手机号提取质量"""
    metrics = {
        'exact_match': 0,      # 完全匹配手机号
        'clean_extraction': 0,  # 成功提取且无多余内容
        'contains_phone': 0,    # 包含手机号
        'avg_length': 0         # 平均生成长度
    }
    
    for prompt in test_prompts:
        result = model.generate([prompt], ...)
        # 计算各项指标
    
    return metrics
```

## 推理优化建议

### 1. 动态停止条件

```python
class DynamicStopper:
    def __init__(self):
        self.phone_pattern = re.compile(r'1[3-9]\d{9}')
    
    def should_stop(self, generated_text, new_token):
        # 如果已经生成了完整手机号，检查下一个token
        if self.phone_pattern.search(generated_text):
            # 如果下一个token不是数字，停止生成
            if not new_token.isdigit():
                return True
        return False
```

### 2. 束搜索 (Beam Search) 优化

```python
# 使用束搜索生成多个候选，选择最佳结果
def beam_search_with_phone_scoring(model, prompt, beam_size=3):
    # 实现自定义束搜索，优先选择在手机号后停止的序列
    pass
```

### 3. 缓存和优化

```python
# 缓存常见提示的结果
phone_cache = {}

def cached_phone_generation(prompt, model):
    if prompt in phone_cache:
        return phone_cache[prompt]
    
    result = generate_and_clean_phone(prompt, model)
    phone_cache[prompt] = result
    return result
```

## 部署建议

### 1. 在线服务优化

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.post("/generate_phone")
async def generate_phone_endpoint(prompt: str):
    # 异步生成和后处理
    result = await asyncio.to_thread(generate_and_clean_phone, prompt)
    return {"phone": result}
```

### 2. 批量处理

```python
def batch_phone_generation(prompts, model, batch_size=8):
    """批量处理多个提示"""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_results = model.generate(batch, ...)
        cleaned = [clean_phone_generation(r) for r in batch_results]
        results.extend(cleaned)
    return results
```

## 监控和评估

### 1. 实时监控

```python
def monitor_generation_quality():
    """监控生成质量"""
    metrics = {
        'success_rate': 0,      # 成功提取率
        'avg_response_time': 0,  # 平均响应时间
        'error_rate': 0,        # 错误率
    }
    # 实现监控逻辑
    return metrics
```

### 2. A/B测试

```python
def ab_test_generation_methods():
    """对比不同生成方法的效果"""
    methods = ['post_processing', 'stop_symbols', 'low_temperature']
    # 实现A/B测试逻辑
    pass
```

## 总结

1. **立即解决**: 使用后处理提取手机号（推荐）
2. **短期优化**: 调整生成参数，添加停止符号
3. **长期改进**: 指令微调，RLHF训练
4. **持续监控**: 建立评估体系，持续优化

您遇到的现象是预训练模型的正常行为，通过合适的后处理可以完美解决。同时，这也为模型的进一步优化提供了明确的方向。