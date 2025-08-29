# InternLM2 实际训练 FLOPs 监控工具

## 🎯 目标

提供能够在实际训练过程中**实时测量真实 FLOPs** 的工具，而非理论估算。

## 🚀 快速开始

### 最小集成（仅需3行代码）

```python
# 在你现有的训练循环中添加：
from runtime_flops_profiler import RuntimeFLOPsMonitor
flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')  # 1️⃣ 添加这行
flops_monitor.start_monitoring()                           # 2️⃣ 添加这行

for batch_idx, (inputs, targets) in enumerate(train_loader):
    # 替换原来的 forward/backward/step：
    result = flops_monitor.measure_step(inputs, targets, optimizer)  # 3️⃣ 替换这行
    
    # 可选：每100步记录FLOPs
    if batch_idx % 100 == 0:
        step_flops = result['step_flops']
        print(f"Step {batch_idx}: {flops_monitor.format_flops(step_flops)}")
```

## 🛠️ 两种测量方法

### Method 1: Hooks (推荐)
- ✅ **低开销** (~1-2% 性能影响)
- ✅ **实时监控**
- ✅ **适用于生产训练**
- ❌ 对自定义操作可能不够精确

### Method 2: PyTorch Profiler (最精确)
- ✅ **官方工具，最准确**
- ✅ **详细操作分解**
- ❌ **较高开销** (~5-10% 性能影响)
- ❌ **训练速度较慢**

## 📊 实时获得的指标

### 基础指标
- **Step FLOPs**: 当前训练步的FLOPs
- **Total FLOPs**: 累计总FLOPs
- **Average FLOPs/step**: 平均每步FLOPs
- **FLOPs/second**: 计算吞吐量

### 详细分解
- **Linear operations**: 矩阵乘法、全连接层
- **Attention**: 多头注意力计算
- **Convolutions**: 卷积操作
- **Other**: 激活函数、归一化等

### 性能统计
- **Min/Max/Std**: FLOPs变化范围
- **Throughput**: 实际计算性能
- **Efficiency**: 硬件利用率洞察

## 🔧 集成示例

### InternLM2 训练集成

```python
# 在 accessory/main_pretrain.py 中修改：
from runtime_flops_profiler import RuntimeFLOPsMonitor

def train_one_epoch(model, train_loader, optimizer, criterion, args):
    # 初始化FLOPs监控
    flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')
    flops_monitor.start_monitoring()
    
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)
        
        # 测量这一步的FLOPs（替代手动forward/backward）
        flops_result = flops_monitor.measure_step(input_ids, labels, optimizer)
        
        # 记录FLOPs信息
        if step % args.log_interval == 0:
            step_flops = flops_result['step_flops']
            total_flops = flops_result['total_flops']
            print(f"Step {step:6d} | FLOPs: {flops_monitor.format_flops(step_flops)} | "
                  f"Total: {flops_monitor.format_flops(total_flops)}")
    
    flops_monitor.cleanup()
```

### 高级集成（带检查点）

```python
class FLOPsAwareTrainer:
    def __init__(self, model, optimizer, train_loader):
        self.flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')
        self.flops_monitor.start_monitoring()
    
    def save_checkpoint_with_flops(self, epoch, step):
        flops_stats = self.flops_monitor.get_statistics()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'flops_info': {
                'total_flops': flops_stats['total_flops'],
                'avg_flops_per_step': flops_stats['avg_flops_per_step'],
                'flops_per_second': flops_stats['flops_per_second']
            }
        }
        torch.save(checkpoint, f'checkpoint_with_flops_{epoch}_{step}.pt')
```

## 🔍 与理论计算对比

```python
# 比较运行时测量与理论计算
from internlm2_compute_calculator import InternLM2ComputeCalculator

def compare_runtime_vs_theoretical(model, config_path, batch_size, seq_len):
    # 理论计算
    calc = InternLM2ComputeCalculator(config_path=config_path)
    theoretical_flops = calc.calculate_total_flops_per_step(batch_size, seq_len)
    
    # 运行时测量
    monitor = RuntimeFLOPsMonitor(model, method='hooks')
    monitor.start_monitoring()
    result = monitor.measure_step(inputs, targets, optimizer)
    runtime_flops = result['step_flops']
    
    # 对比分析
    ratio = runtime_flops / theoretical_flops
    print(f"Runtime/Theoretical ratio: {ratio:.2f}")
    
    if 0.8 <= ratio <= 1.2:
        print("✅ 运行时和理论估算相近")
    else:
        print("⚠️ 存在显著差异，需要进一步分析")
```

## 📈 实用建议

### 1. 选择合适的方法
- **生产训练**: 使用 `method='hooks'`（低开销）
- **详细分析**: 使用 `method='profiler'`（高精度）

### 2. 记录策略
- 每N步记录FLOPs（不要每步都记录）
- 在检查点中保存FLOPs信息
- 跟踪训练过程中的FLOPs趋势

### 3. 性能考虑
- Hooks方法增加约1-2%开销
- Profiler方法增加约5-10%开销
- 如使用Profiler，注意监控内存使用

### 4. 验证准确性
- 与理论估算对比验证
- 检查相似步骤间的一致性
- 验证已知模型的FLOPs

## 🚨 常见问题解决

### 问题1：自定义操作未计数
**解决方案**：为自定义操作添加hooks
```python
def custom_op_flop_count(module, input, output):
    flops = calculate_custom_flops(input, output)
    flops_monitor.flop_counter.current_step_flops += flops

model.custom_layer.register_forward_hook(custom_op_flop_count)
```

### 问题2：内存问题
**解决方案**：
- 切换到hooks方法
- 只对部分步骤进行profiling
- 减少profiler记录选项

### 问题3：FLOPs计数不一致
**解决方案**：
- 检查模型动态行为
- 验证输入尺寸一致性
- 查找条件计算

## 📚 文件说明

- `runtime_flops_profiler.py` - 主要FLOPs监控工具
- `practical_flops_example.py` - 实用集成示例
- `training_with_flops_template.py` - 通用集成模板
- `internlm2_flops_integration.py` - InternLM2特定集成
- `flops_integration_guide.py` - 详细集成指南

## 🎯 使用流程

1. **复制工具**: 将 `runtime_flops_profiler.py` 复制到训练项目
2. **选择方法**: hooks（低开销）或profiler（高精度）
3. **小规模测试**: 先用小模型验证准确性
4. **扩展到完整训练**: 应用到实际训练流程
5. **分析结果**: 使用数据优化训练效率

## ✨ 主要优势

- **真实测量**: 获取实际运行FLOPs，非理论估算
- **实时监控**: 训练过程中持续跟踪计算量
- **低侵入性**: 最少3行代码即可集成
- **详细分析**: 提供操作级别的FLOPs分解
- **灵活配置**: 支持不同监控方法和记录策略
- **生产就绪**: 适合实际训练环境使用

---

**立即开始**: 复制 `runtime_flops_profiler.py` 到你的项目，用hooks方法开始监控！🚀