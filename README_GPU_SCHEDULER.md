# GPU Scheduler for Time-based Pretraining

这个脚本系统可以根据时间段自动切换GPU使用并管理预训练任务。

## 功能特点

- **时间段管理**: 根据预设时间自动切换可用GPU
  - 白天 (8:00-18:00): 仅使用GPU 0,3
  - 夜晚 (18:00-8:00): 使用所有GPU 0,1,2,3
- **智能任务管理**: 自动停止当前训练，切换GPU配置，重新启动训练
- **检查点恢复**: 支持从最新检查点自动恢复训练
- **进程监控**: 监控训练进程状态，异常时自动重启
- **日志记录**: 详细的操作日志和状态记录

## 文件说明

- `gpu_scheduler.py` - 主调度器脚本
- `setup_gpu_scheduler.py` - 配置设置脚本
- `gpu_monitor.py` - 监控和控制工具
- `gpu_scheduler_config.json` - 配置文件（运行setup后生成）

## 快速开始

### 1. 初始配置

运行配置脚本来设置你的环境：

```bash
python3 setup_gpu_scheduler.py
```

这个脚本会：
- 扫描工作区寻找相关文件
- 交互式配置时间段和GPU分配
- 设置训练脚本路径和参数
- 生成配置文件

### 2. 启动调度器

```bash
# 前台运行（用于测试）
python3 gpu_scheduler.py

# 后台运行（推荐用于生产）
python3 gpu_scheduler.py --daemon

# 使用自定义配置文件
python3 gpu_scheduler.py --config my_config.json
```

### 3. 监控状态

```bash
# 查看当前状态
python3 gpu_monitor.py --status

# 持续监控
python3 gpu_monitor.py --watch

# 查看日志
python3 gpu_monitor.py --logs 100
```

### 4. 控制操作

```bash
# 停止所有训练进程
python3 gpu_monitor.py --kill

# 启动调度器
python3 gpu_monitor.py --start

# 检查调度器状态
python3 gpu_scheduler.py --status

# 停止调度器
python3 gpu_scheduler.py --stop
```

## 配置说明

### 时间段配置

```json
{
  "time_schedules": {
    "day_shift": {
      "start_time": "08:00",
      "end_time": "18:00",
      "allowed_gpus": [0, 3],
      "description": "Day shift: 8 AM - 6 PM, GPUs 0,3 only"
    },
    "night_shift": {
      "start_time": "18:00", 
      "end_time": "08:00",
      "allowed_gpus": [0, 1, 2, 3],
      "description": "Night shift: 6 PM - 8 AM, All GPUs"
    }
  }
}
```

### 预训练配置

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
      "llama_config": "/path/to/config.json",
      "tokenizer_path": "/path/to/tokenizer.model", 
      "data_meta_path": "/path/to/data_meta.json",
      "data_root": "/path/to/data"
    },
    "checkpoint_dir": "./checkpoints_scheduled",
    "resume_from_checkpoint": true
  }
}
```

### 监控配置

```json
{
  "monitoring": {
    "check_interval": 60,        // 检查间隔（秒）
    "grace_period": 30,          // 优雅停止等待时间（秒）
    "max_restart_attempts": 3,   // 最大重启尝试次数
    "log_level": "INFO"          // 日志级别
  }
}
```

## 工作流程

1. **启动**: 调度器启动并读取配置
2. **时间检查**: 每分钟检查当前时间段
3. **GPU切换**: 如果需要切换GPU配置：
   - 优雅停止当前训练进程
   - 等待进程完全停止
   - 使用新的GPU配置启动训练
4. **监控**: 持续监控训练进程状态
5. **恢复**: 如果进程异常退出，自动重启

## 高级功能

### 自定义时间段

你可以定义多个时间段，调度器会自动选择当前时间匹配的时间段：

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

### 检查点管理

调度器会自动：
- 保存检查点到指定目录
- 在重启时从最新检查点恢复
- 管理检查点文件

### 日志分析

日志文件包含详细信息：
- 时间段切换记录
- GPU分配变化
- 进程启动/停止
- 错误和警告

## 故障排除

### 常见问题

1. **配置文件错误**
   ```bash
   # 重新运行配置
   python3 setup_gpu_scheduler.py
   ```

2. **进程无法启动**
   ```bash
   # 检查配置中的路径是否正确
   python3 gpu_monitor.py --status
   
   # 查看详细日志
   python3 gpu_monitor.py --logs 50
   ```

3. **GPU不可用**
   ```bash
   # 检查GPU状态（如果nvidia-smi可用）
   nvidia-smi
   
   # 查看调度器日志
   tail -f gpu_scheduler.log
   ```

4. **训练进程卡住**
   ```bash
   # 强制停止所有训练进程
   python3 gpu_monitor.py --kill
   
   # 重启调度器
   python3 gpu_monitor.py --start
   ```

### 调试模式

设置日志级别为DEBUG来获取更多信息：

```json
{
  "monitoring": {
    "log_level": "DEBUG"
  }
}
```

## 安全注意事项

- 调度器会强制终止现有训练进程，确保重要数据已保存
- 建议先在测试环境中验证配置
- 定期备份检查点和配置文件
- 监控磁盘空间，避免日志文件过大

## 扩展功能

系统设计为可扩展的，你可以：
- 添加更多时间段
- 集成GPU使用率监控
- 添加邮件/消息通知
- 实现Web界面监控
- 集成作业队列系统

## 许可证

本脚本基于现有的预训练框架，请遵循相应的许可证条款。