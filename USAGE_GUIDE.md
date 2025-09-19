# GPU调度器使用指南

## 概述

这个GPU调度器系统可以根据时间段自动切换GPU使用并管理预训练任务：

- **白天 (8:00-18:00)**: 仅使用GPU 0,3
- **夜晚 (18:00-8:00)**: 使用所有GPU 0,1,2,3

## 文件说明

### 主要脚本
- `gpu_scheduler.py` - 完整版调度器（需要psutil等依赖）
- `gpu_scheduler_simple.py` - 简化版调度器（无外部依赖）
- `setup_gpu_scheduler.py` - 交互式配置脚本
- `gpu_monitor.py` - 监控和控制工具（需要psutil）

### 工具脚本
- `demo_gpu_scheduler.py` - 演示脚本
- `check_system.py` - 系统环境检查
- `start_scheduler.sh` - 快速启动脚本

### 配置文件
- `gpu_scheduler_config.json` - 主配置文件
- `gpu_scheduler_config_demo.json` - 演示配置文件

## 快速开始

### 1. 选择版本

根据您的环境选择合适的版本：

**完整版** (推荐，如果可以安装依赖包)：
```bash
# 需要安装依赖
pip install psutil torch numpy
# 或者
apt install python3-psutil python3-torch python3-numpy

# 使用完整版
python3 gpu_scheduler.py
```

**简化版** (如果无法安装依赖包)：
```bash
# 无需额外依赖，直接使用
python3 gpu_scheduler_simple.py
```

### 2. 配置系统

#### 方法1: 使用配置向导（推荐）
```bash
python3 setup_gpu_scheduler.py
```

#### 方法2: 手动编辑配置文件
复制并编辑示例配置：
```bash
cp gpu_scheduler_config_demo.json gpu_scheduler_config.json
# 然后编辑 gpu_scheduler_config.json 中的路径
```

#### 方法3: 使用演示配置
```bash
# 运行演示来生成配置
python3 demo_gpu_scheduler.py
```

### 3. 启动调度器

#### 使用快速启动脚本（推荐）
```bash
./start_scheduler.sh
```

#### 手动启动
```bash
# 前台运行（用于测试）
python3 gpu_scheduler_simple.py

# 后台运行
nohup python3 gpu_scheduler_simple.py --daemon > scheduler.log 2>&1 &
```

## 详细配置

### 时间段配置

编辑配置文件中的 `time_schedules` 部分：

```json
{
  "time_schedules": {
    "day_shift": {
      "start_time": "08:00",
      "end_time": "18:00",
      "allowed_gpus": [0, 3],
      "description": "白天班次：上午8点到下午6点，只使用GPU 0,3"
    },
    "night_shift": {
      "start_time": "18:00", 
      "end_time": "08:00",
      "allowed_gpus": [0, 1, 2, 3],
      "description": "夜班：下午6点到上午8点，使用所有GPU"
    }
  }
}
```

### 预训练配置

设置您的预训练脚本和参数：

```json
{
  "pretraining": {
    "script_path": "/workspace/accessory/main_pretrain.py",
    "base_args": [
      "--batch_size", "4",
      "--accum_iter", "4", 
      "--lr", "0.001",
      "--max_words", "2048"
    ],
    "required_args": {
      "llama_config": "/path/to/your/config.json",
      "tokenizer_path": "/path/to/your/tokenizer.model",
      "data_meta_path": "/path/to/your/data_meta.json",
      "data_root": "/path/to/your/data"
    }
  }
}
```

## 监控和控制

### 查看状态
```bash
# 使用简化版调度器
python3 gpu_scheduler_simple.py --status

# 如果安装了psutil，可以使用监控脚本
python3 gpu_monitor.py --status
```

### 持续监控
```bash
# 每30秒刷新状态
python3 gpu_monitor.py --watch
```

### 查看日志
```bash
# 查看调度器日志
tail -f gpu_scheduler_simple.log

# 或使用监控脚本
python3 gpu_monitor.py --logs 50
```

### 停止训练
```bash
# 停止所有训练进程
python3 gpu_scheduler_simple.py --stop

# 或使用监控脚本
python3 gpu_monitor.py --kill
```

### 停止调度器
```bash
# 找到调度器进程并停止
pkill -f gpu_scheduler
```

## 工作流程

1. **启动**: 调度器读取配置并开始监控
2. **时间检查**: 每分钟检查当前时间段
3. **GPU切换**: 当需要切换时：
   - 优雅停止当前训练
   - 等待进程完全停止
   - 使用新GPU配置重启训练
4. **恢复**: 自动从最新检查点恢复
5. **监控**: 持续监控进程状态

## 故障排除

### 常见问题

1. **权限错误**
   ```bash
   chmod +x *.py *.sh
   ```

2. **配置文件错误**
   ```bash
   python3 setup_gpu_scheduler.py  # 重新配置
   ```

3. **进程无法启动**
   ```bash
   # 检查配置
   python3 gpu_scheduler_simple.py --status
   
   # 查看日志
   tail -f gpu_scheduler_simple.log
   ```

4. **训练进程卡住**
   ```bash
   # 强制停止
   python3 gpu_scheduler_simple.py --stop
   
   # 或者
   pkill -f pretrain
   ```

### 调试技巧

1. **测试配置**
   ```bash
   python3 demo_gpu_scheduler.py
   ```

2. **检查系统环境**
   ```bash
   python3 check_system.py
   ```

3. **手动测试训练命令**
   ```bash
   # 从日志中复制训练命令，手动运行测试
   CUDA_VISIBLE_DEVICES=0,3 python3 /workspace/accessory/main_pretrain.py ...
   ```

## 高级功能

### 自定义时间段

您可以定义更多时间段：

```json
{
  "time_schedules": {
    "morning": {
      "start_time": "06:00",
      "end_time": "12:00",
      "allowed_gpus": [0, 1]
    },
    "afternoon": {
      "start_time": "12:00", 
      "end_time": "18:00",
      "allowed_gpus": [2, 3]
    },
    "night": {
      "start_time": "18:00",
      "end_time": "06:00", 
      "allowed_gpus": [0, 1, 2, 3]
    }
  }
}
```

### 调整监控参数

```json
{
  "monitoring": {
    "check_interval": 60,     // 检查间隔（秒）
    "grace_period": 30,       // 优雅停止等待时间
    "max_restart_attempts": 3 // 最大重启次数
  }
}
```

## 安全注意事项

- 调度器会强制终止现有训练进程
- 确保重要数据已保存到检查点
- 建议先在测试环境验证
- 定期备份配置和检查点
- 监控磁盘空间

## 扩展建议

- 添加邮件/消息通知
- 集成GPU使用率监控
- 实现Web界面
- 添加作业队列功能
- 集成更多调度策略

## 支持

如果遇到问题：

1. 查看日志文件
2. 运行系统检查
3. 使用演示模式测试
4. 检查配置文件格式
5. 确认文件路径正确