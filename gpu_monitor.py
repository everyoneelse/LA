#!/usr/bin/env python3
"""
GPU Monitor and Control Script

Simple utility to monitor and control the GPU scheduler and training processes.
"""

import json
import psutil
import datetime
import argparse
import subprocess
import time
from pathlib import Path


def get_training_processes():
    """Find all running training processes"""
    training_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
        try:
            if proc.info['name'] in ['python3', 'python']:
                cmdline = proc.info['cmdline']
                if cmdline and any('pretrain' in str(arg) for arg in cmdline):
                    training_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(cmdline),
                        'memory_mb': proc.info['memory_info'].rss // 1024 // 1024,
                        'start_time': datetime.datetime.fromtimestamp(proc.info['create_time']).strftime('%Y-%m-%d %H:%M:%S')
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return training_processes


def get_scheduler_status():
    """Get current scheduler status"""
    try:
        result = subprocess.run(
            ['python3', 'gpu_scheduler.py', '--status'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {"error": f"Scheduler error: {result.stderr}"}
    except subprocess.TimeoutExpired:
        return {"error": "Scheduler status check timed out"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from scheduler"}
    except FileNotFoundError:
        return {"error": "gpu_scheduler.py not found"}
    except Exception as e:
        return {"error": f"Error getting scheduler status: {e}"}


def format_time_period(period_info):
    """Format time period information"""
    if 'period' not in period_info:
        return "Unknown period"
    
    period_name = period_info['period']
    gpus = period_info.get('allowed_gpus', [])
    description = period_info.get('description', f"{period_name} - GPUs: {gpus}")
    
    return description


def show_status():
    """Show comprehensive status"""
    print("=" * 70)
    print("🖥️  GPU SCHEDULER STATUS")
    print("=" * 70)
    
    # Current time
    now = datetime.datetime.now()
    print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Scheduler status
    print("📊 Scheduler Status:")
    scheduler_status = get_scheduler_status()
    
    if "error" in scheduler_status:
        print(f"   ❌ {scheduler_status['error']}")
    else:
        current_period = scheduler_status.get('current_period', {})
        print(f"   📅 Current period: {format_time_period(current_period)}")
        print(f"   🏃 Job running: {'✅ Yes' if scheduler_status.get('job_running', False) else '❌ No'}")
        
        if scheduler_status.get('current_job_pid'):
            print(f"   🆔 Current job PID: {scheduler_status['current_job_pid']}")
        
        current_gpus = scheduler_status.get('current_gpus', [])
        if current_gpus:
            print(f"   🎯 Current GPUs: {current_gpus}")
    
    print()
    
    # Training processes
    print("🚀 Training Processes:")
    training_procs = get_training_processes()
    
    if not training_procs:
        print("   📭 No training processes found")
    else:
        for proc in training_procs:
            print(f"   🔹 PID {proc['pid']} (Started: {proc['start_time']}, Memory: {proc['memory_mb']} MB)")
            print(f"      Command: {proc['cmdline'][:100]}{'...' if len(proc['cmdline']) > 100 else ''}")
    
    print()
    
    # Configuration file status
    print("⚙️  Configuration:")
    config_file = Path("gpu_scheduler_config.json")
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            schedules = config.get('time_schedules', {})
            print(f"   📋 {len(schedules)} time schedules configured")
            for name, schedule in schedules.items():
                start = schedule.get('start_time', 'Unknown')
                end = schedule.get('end_time', 'Unknown')
                gpus = schedule.get('allowed_gpus', [])
                print(f"      • {name}: {start}-{end}, GPUs {gpus}")
            
            pretraining = config.get('pretraining', {})
            script_path = pretraining.get('script_path', 'Not set')
            print(f"   📜 Training script: {script_path}")
            
            required_args = pretraining.get('required_args', {})
            missing_args = [k for k, v in required_args.items() if not v]
            if missing_args:
                print(f"   ⚠️  Missing required args: {missing_args}")
            else:
                print("   ✅ All required arguments configured")
                
        except Exception as e:
            print(f"   ❌ Error reading config: {e}")
    else:
        print("   ❌ Configuration file not found")
        print("   💡 Run: python3 setup_gpu_scheduler.py")
    
    print()
    
    # Log file status
    log_file = Path("gpu_scheduler.log")
    if log_file.exists():
        try:
            stat = log_file.stat()
            size_kb = stat.st_size // 1024
            modified = datetime.datetime.fromtimestamp(stat.st_mtime)
            print(f"📋 Log file: {log_file} ({size_kb} KB, modified: {modified.strftime('%Y-%m-%d %H:%M:%S')})")
        except Exception as e:
            print(f"📋 Log file: {log_file} (error reading: {e})")
    else:
        print("📋 Log file: Not found")


def show_logs(lines=50):
    """Show recent log entries"""
    log_file = Path("gpu_scheduler.log")
    
    if not log_file.exists():
        print("❌ Log file not found")
        return
    
    try:
        # Use tail to get last N lines
        result = subprocess.run(['tail', f'-{lines}', str(log_file)], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"📋 Last {lines} log entries:")
            print("-" * 70)
            print(result.stdout)
        else:
            # Fallback to Python implementation
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                print(f"📋 Last {len(recent_lines)} log entries:")
                print("-" * 70)
                for line in recent_lines:
                    print(line.rstrip())
                    
    except Exception as e:
        print(f"❌ Error reading log file: {e}")


def kill_training():
    """Kill all training processes"""
    training_procs = get_training_processes()
    
    if not training_procs:
        print("📭 No training processes found")
        return
    
    print(f"🛑 Found {len(training_procs)} training process(es), terminating...")
    
    for proc_info in training_procs:
        try:
            proc = psutil.Process(proc_info['pid'])
            print(f"   Terminating PID {proc_info['pid']}...")
            proc.terminate()
            
            # Wait for graceful shutdown
            try:
                proc.wait(timeout=10)
                print(f"   ✅ PID {proc_info['pid']} terminated gracefully")
            except psutil.TimeoutExpired:
                print(f"   🔨 Force killing PID {proc_info['pid']}...")
                proc.kill()
                proc.wait()
                print(f"   ✅ PID {proc_info['pid']} force killed")
                
        except psutil.NoSuchProcess:
            print(f"   ⚠️  PID {proc_info['pid']} already terminated")
        except Exception as e:
            print(f"   ❌ Error terminating PID {proc_info['pid']}: {e}")


def start_scheduler():
    """Start the GPU scheduler"""
    try:
        print("🚀 Starting GPU scheduler...")
        subprocess.Popen(['python3', 'gpu_scheduler.py', '--daemon'])
        time.sleep(2)  # Give it a moment to start
        
        # Check if it started successfully
        scheduler_status = get_scheduler_status()
        if "error" not in scheduler_status:
            print("✅ GPU scheduler started successfully")
        else:
            print(f"❌ Error starting scheduler: {scheduler_status['error']}")
            
    except Exception as e:
        print(f"❌ Error starting scheduler: {e}")


def main():
    parser = argparse.ArgumentParser(description="GPU Monitor and Control")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--logs", type=int, metavar="N", help="Show last N log entries (default: 50)")
    parser.add_argument("--kill", action="store_true", help="Kill all training processes")
    parser.add_argument("--start", action="store_true", help="Start GPU scheduler")
    parser.add_argument("--watch", action="store_true", help="Watch status continuously")
    
    args = parser.parse_args()
    
    if args.logs is not None:
        show_logs(args.logs if args.logs > 0 else 50)
    elif args.kill:
        kill_training()
    elif args.start:
        start_scheduler()
    elif args.watch:
        try:
            while True:
                print("\033[2J\033[H")  # Clear screen
                show_status()
                print("\n⏱️  Refreshing in 30 seconds... (Ctrl+C to stop)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n👋 Stopped watching")
    else:
        show_status()


if __name__ == "__main__":
    main()