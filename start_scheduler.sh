#!/bin/bash
"""
GPU Scheduler Quick Start Script
快速启动GPU调度器的脚本
"""

echo "🚀 GPU调度器快速启动"
echo "=================================="

# 检查配置文件是否存在
if [ ! -f "gpu_scheduler_config.json" ]; then
    echo "⚠️  配置文件不存在，正在运行配置向导..."
    python3 setup_gpu_scheduler.py
    
    if [ $? -ne 0 ]; then
        echo "❌ 配置失败，退出"
        exit 1
    fi
fi

echo "✅ 配置文件已存在"

# 检查是否有运行中的调度器
if pgrep -f "gpu_scheduler.py" > /dev/null; then
    echo "⚠️  检测到运行中的调度器进程"
    echo "是否要停止现有进程并重启？(y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🛑 停止现有进程..."
        pkill -f "gpu_scheduler.py"
        sleep 2
    else
        echo "📊 显示当前状态："
        python3 gpu_monitor.py --status
        exit 0
    fi
fi

# 显示当前状态
echo "📊 当前状态："
python3 gpu_monitor.py --status

echo ""
echo "🚀 启动选项："
echo "1) 前台运行 (用于测试和调试)"
echo "2) 后台运行 (推荐用于生产环境)"
echo "3) 仅显示状态"
echo "4) 退出"

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "🔍 前台启动调度器 (Ctrl+C 停止)..."
        python3 gpu_scheduler.py
        ;;
    2)
        echo "🔧 后台启动调度器..."
        nohup python3 gpu_scheduler.py --daemon > scheduler_daemon.log 2>&1 &
        sleep 2
        echo "✅ 调度器已在后台启动"
        echo "📋 查看日志: tail -f scheduler_daemon.log"
        echo "📊 查看状态: python3 gpu_monitor.py --status"
        ;;
    3)
        echo "📊 当前详细状态："
        python3 gpu_monitor.py --status
        ;;
    4)
        echo "👋 退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "💡 有用的命令："
echo "  查看状态: python3 gpu_monitor.py --status"
echo "  持续监控: python3 gpu_monitor.py --watch"
echo "  查看日志: python3 gpu_monitor.py --logs 50"
echo "  停止训练: python3 gpu_monitor.py --kill"
echo "  停止调度器: pkill -f gpu_scheduler.py"