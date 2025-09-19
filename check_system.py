#!/usr/bin/env python3
"""
System Check Script for GPU Scheduler
检查系统环境和依赖项的脚本
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_packages():
    """检查Python依赖包"""
    print("🐍 Python包检查:")
    
    required_packages = [
        'psutil', 'torch', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 安装缺失的包:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_gpu_environment():
    """检查GPU环境"""
    print("\n🖥️  GPU环境检查:")
    
    # 检查CUDA环境变量
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"   ✅ CUDA_HOME: {cuda_home}")
    else:
        print("   ⚠️  CUDA_HOME未设置")
    
    # 检查nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ nvidia-smi可用")
            # 解析GPU数量
            lines = result.stdout.split('\n')
            gpu_count = 0
            for line in lines:
                if 'GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'RTX' in line:
                    gpu_count += 1
            print(f"   🎯 检测到 {gpu_count} 个GPU")
        else:
            print("   ❌ nvidia-smi不可用或出错")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ nvidia-smi命令不存在")
    
    # 检查PyTorch CUDA支持
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ PyTorch CUDA可用 ({gpu_count} 个GPU)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"      GPU {i}: {gpu_name}")
        else:
            print("   ❌ PyTorch CUDA不可用")
    except ImportError:
        print("   ❌ PyTorch未安装")


def check_workspace_files():
    """检查工作区文件"""
    print("\n📁 工作区文件检查:")
    
    # 检查关键文件
    key_files = [
        "/workspace/accessory/main_pretrain.py",
        "/workspace/accessory/engine_pretrain.py"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (缺失)")
    
    # 检查示例数据
    example_files = [
        "/workspace/data_example/PretrainMeta.json",
        "/workspace/data_example/ShareGPT.json"
    ]
    
    print("\n📊 示例数据检查:")
    for file_path in example_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (缺失)")


def check_scheduler_files():
    """检查调度器文件"""
    print("\n🤖 调度器文件检查:")
    
    scheduler_files = [
        "gpu_scheduler.py",
        "setup_gpu_scheduler.py", 
        "gpu_monitor.py",
        "demo_gpu_scheduler.py",
        "start_scheduler.sh"
    ]
    
    for file_name in scheduler_files:
        file_path = Path(file_name)
        if file_path.exists():
            # 检查是否可执行
            if os.access(file_path, os.X_OK):
                print(f"   ✅ {file_name} (可执行)")
            else:
                print(f"   ⚠️  {file_name} (存在但不可执行)")
        else:
            print(f"   ❌ {file_name} (缺失)")


def check_config_file():
    """检查配置文件"""
    print("\n⚙️  配置文件检查:")
    
    config_files = [
        "gpu_scheduler_config.json",
        "gpu_scheduler_config_demo.json"
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                
                print(f"   ✅ {config_file}")
                
                # 检查配置完整性
                required_sections = ['time_schedules', 'pretraining', 'monitoring']
                missing_sections = []
                
                for section in required_sections:
                    if section not in config:
                        missing_sections.append(section)
                
                if missing_sections:
                    print(f"      ⚠️  缺少配置节: {missing_sections}")
                else:
                    print(f"      ✅ 配置完整")
                
                # 检查必需参数
                required_args = config.get('pretraining', {}).get('required_args', {})
                missing_args = [k for k, v in required_args.items() if not v]
                
                if missing_args:
                    print(f"      ⚠️  未设置的必需参数: {missing_args}")
                else:
                    print(f"      ✅ 所有必需参数已设置")
                    
            except json.JSONDecodeError as e:
                print(f"   ❌ {config_file} (JSON格式错误: {e})")
            except Exception as e:
                print(f"   ❌ {config_file} (读取错误: {e})")
        else:
            print(f"   ❌ {config_file} (不存在)")


def check_running_processes():
    """检查运行中的进程"""
    print("\n🔄 进程检查:")
    
    try:
        import psutil
        
        # 检查调度器进程
        scheduler_procs = []
        training_procs = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline:
                    cmdline_str = ' '.join(cmdline)
                    if 'gpu_scheduler.py' in cmdline_str:
                        scheduler_procs.append(proc.info['pid'])
                    elif 'pretrain' in cmdline_str and 'python' in proc.info['name']:
                        training_procs.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if scheduler_procs:
            print(f"   🤖 调度器进程: {scheduler_procs}")
        else:
            print("   📭 没有运行中的调度器进程")
        
        if training_procs:
            print(f"   🚀 训练进程: {training_procs}")
        else:
            print("   📭 没有运行中的训练进程")
            
    except ImportError:
        print("   ❌ psutil未安装，无法检查进程")


def generate_system_report():
    """生成系统报告"""
    print("\n📋 生成系统报告...")
    
    report = {
        "timestamp": subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "environment_variables": {
            "CUDA_HOME": os.environ.get('CUDA_HOME', 'Not set'),
            "CUDA_VISIBLE_DEVICES": os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
            "PATH": os.environ.get('PATH', '')[:200] + "..." if len(os.environ.get('PATH', '')) > 200 else os.environ.get('PATH', '')
        }
    }
    
    try:
        with open("system_check_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("   ✅ 系统报告已保存到: system_check_report.json")
    except Exception as e:
        print(f"   ❌ 保存系统报告失败: {e}")


def main():
    print("🔍 GPU调度器系统检查")
    print("=" * 50)
    
    # 运行各项检查
    checks_passed = 0
    total_checks = 6
    
    if check_python_packages():
        checks_passed += 1
    
    check_gpu_environment()
    checks_passed += 1
    
    check_workspace_files()
    checks_passed += 1
    
    check_scheduler_files()
    checks_passed += 1
    
    check_config_file()
    checks_passed += 1
    
    check_running_processes()
    checks_passed += 1
    
    # 生成报告
    generate_system_report()
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 检查总结:")
    print(f"   完成检查: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        print("   ✅ 系统检查完成，环境看起来正常")
    else:
        print("   ⚠️  发现一些问题，请查看上面的详细信息")
    
    print("\n💡 下一步:")
    if not Path("gpu_scheduler_config.json").exists():
        print("   1. 运行配置向导: python3 setup_gpu_scheduler.py")
    else:
        print("   1. 配置文件已存在，可以开始使用")
    
    print("   2. 运行演示: python3 demo_gpu_scheduler.py")
    print("   3. 启动调度器: ./start_scheduler.sh")
    print("   4. 监控状态: python3 gpu_monitor.py --status")


if __name__ == "__main__":
    main()