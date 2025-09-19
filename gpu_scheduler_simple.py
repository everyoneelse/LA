#!/usr/bin/env python3
"""
Simplified GPU Scheduler (No external dependencies)

This version doesn't require psutil and uses basic system commands for process management.
这个简化版本不需要psutil，使用基本的系统命令进行进程管理。
"""

import os
import sys
import json
import time
import signal
import subprocess
import datetime
import threading
import argparse
from pathlib import Path


class SimpleGPUScheduler:
    def __init__(self, config_path: str = "gpu_scheduler_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.current_job_pid = None
        self.current_gpus = set()
        self.shutdown_flag = threading.Event()
        self.job_lock = threading.Lock()
        
        # Setup logging (simple file logging)
        self.log_file = open("gpu_scheduler_simple.log", "a")
        self.log(f"Scheduler initialized at {datetime.datetime.now()}")
        
        # Track job state
        self.job_state = {
            "is_running": False,
            "start_time": None,
            "gpus_used": [],
            "process_pid": None,
            "restart_count": 0
        }
    
    def log(self, message):
        """Simple logging function"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
        self.log_file.write(log_entry)
        self.log_file.flush()
    
    def load_config(self) -> dict:
        """Load configuration from JSON file or create default config"""
        default_config = {
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
            },
            "pretraining": {
                "script_path": "/workspace/accessory/main_pretrain.py",
                "base_args": [
                    "--batch_size", "4",
                    "--accum_iter", "4",
                    "--lr", "0.001",
                    "--max_words", "2048",
                    "--output_dir", "./output_scheduled",
                    "--save_freq", "5000",
                    "--val_freq", "10000"
                ],
                "required_args": {
                    "llama_config": None,
                    "tokenizer_path": None,
                    "data_meta_path": None,
                    "data_root": None
                },
                "checkpoint_dir": "./checkpoints_scheduled",
                "resume_from_checkpoint": True
            },
            "monitoring": {
                "check_interval": 60,
                "grace_period": 30,
                "max_restart_attempts": 3
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for missing keys
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
            except Exception as e:
                self.log(f"Error loading config: {e}. Using default config.")
                
        # Create default config file
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        self.log(f"Created default config at {self.config_path}")
        return default_config
    
    def get_current_time_period(self) -> dict:
        """Determine current time period and allowed GPUs"""
        now = datetime.datetime.now().time()
        
        for period_name, schedule in self.config["time_schedules"].items():
            start_time = datetime.time.fromisoformat(schedule["start_time"])
            end_time = datetime.time.fromisoformat(schedule["end_time"])
            
            # Handle overnight periods (e.g., 18:00 to 08:00)
            if start_time > end_time:
                if now >= start_time or now < end_time:
                    return {
                        "period": period_name,
                        "allowed_gpus": schedule["allowed_gpus"],
                        "description": schedule["description"]
                    }
            else:
                if start_time <= now < end_time:
                    return {
                        "period": period_name,
                        "allowed_gpus": schedule["allowed_gpus"],
                        "description": schedule["description"]
                    }
        
        # Default fallback
        return {
            "period": "unknown",
            "allowed_gpus": [0, 3],
            "description": "Default period"
        }
    
    def kill_existing_training(self):
        """Kill existing training processes using system commands"""
        if self.current_job_pid:
            try:
                # Try to terminate gracefully first
                os.kill(self.current_job_pid, signal.SIGTERM)
                self.log(f"Sent SIGTERM to PID {self.current_job_pid}")
                
                # Wait for graceful shutdown
                time.sleep(self.config["monitoring"]["grace_period"])
                
                # Check if still running and force kill if needed
                try:
                    os.kill(self.current_job_pid, 0)  # Check if process exists
                    self.log(f"Process {self.current_job_pid} still running, force killing")
                    os.kill(self.current_job_pid, signal.SIGKILL)
                except OSError:
                    self.log(f"Process {self.current_job_pid} terminated gracefully")
                    
            except OSError as e:
                self.log(f"Error terminating process {self.current_job_pid}: {e}")
            
            self.current_job_pid = None
        
        # Also kill any other training processes
        self.kill_training_processes()
    
    def kill_training_processes(self):
        """Kill any existing training processes using pkill"""
        try:
            # Kill Python processes running pretraining scripts
            result = subprocess.run(['pkill', '-f', 'pretrain'], capture_output=True)
            if result.returncode == 0:
                self.log("Killed existing training processes")
            else:
                self.log("No existing training processes found")
        except Exception as e:
            self.log(f"Error killing training processes: {e}")
    
    def find_latest_checkpoint(self, checkpoint_dir: Path) -> str:
        """Find the latest checkpoint file"""
        try:
            checkpoints = list(checkpoint_dir.glob("checkpoint-*.pth"))
            if checkpoints:
                # Sort by modification time and return the latest
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                return str(latest)
        except Exception as e:
            self.log(f"Error finding latest checkpoint: {e}")
        return None
    
    def build_training_command(self, gpu_ids: list) -> tuple:
        """Build the training command with appropriate GPU settings"""
        config = self.config["pretraining"]
        
        # Base command
        cmd = ["python3", config["script_path"]]
        
        # Add base arguments
        cmd.extend(config["base_args"])
        
        # Add required arguments
        for arg_name, arg_value in config["required_args"].items():
            if arg_value:
                cmd.extend([f"--{arg_name}", str(arg_value)])
        
        # Set GPU environment
        gpu_str = ",".join(map(str, gpu_ids))
        
        # Add distributed training args if multiple GPUs
        if len(gpu_ids) > 1:
            cmd.extend([
                "--model_parallel_size", "1",
                "--data_parallel", "fsdp"
            ])
        
        # Add resume checkpoint if available and enabled
        if config["resume_from_checkpoint"]:
            checkpoint_dir = Path(config["checkpoint_dir"])
            if checkpoint_dir.exists():
                latest_checkpoint = self.find_latest_checkpoint(checkpoint_dir)
                if latest_checkpoint:
                    cmd.extend(["--resume", latest_checkpoint])
        
        return cmd, gpu_str
    
    def start_training(self, gpu_ids: list) -> bool:
        """Start training with specified GPUs"""
        with self.job_lock:
            try:
                # Kill any existing training
                self.kill_existing_training()
                
                # Build command
                cmd, gpu_str = self.build_training_command(gpu_ids)
                
                self.log(f"Starting training with GPUs {gpu_ids}")
                self.log(f"Command: {' '.join(cmd)}")
                
                # Set environment variables
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_str
                
                # Start the process
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    preexec_fn=os.setsid  # Create new process group
                )
                
                self.current_job_pid = process.pid
                self.current_gpus = set(gpu_ids)
                
                # Update job state
                self.job_state.update({
                    "is_running": True,
                    "start_time": datetime.datetime.now().isoformat(),
                    "gpus_used": gpu_ids,
                    "process_pid": process.pid,
                    "restart_count": self.job_state.get("restart_count", 0) + 1
                })
                
                self.log(f"Training started with PID {process.pid}")
                
                # Start output monitoring thread
                threading.Thread(
                    target=self.monitor_training_output,
                    args=(process,),
                    daemon=True
                ).start()
                
                return True
                
            except Exception as e:
                self.log(f"Failed to start training: {e}")
                return False
    
    def monitor_training_output(self, process):
        """Monitor training process output"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Log important training output
                    if any(keyword in line.lower() for keyword in ['loss', 'epoch', 'iter', 'error', 'warning']):
                        self.log(f"TRAINING: {line.strip()}")
                
                # Check if process is still running
                if process.poll() is not None:
                    break
        except Exception as e:
            self.log(f"Error monitoring training output: {e}")
    
    def is_training_running(self) -> bool:
        """Check if training process is still running"""
        if not self.current_job_pid:
            return False
        
        try:
            # Send signal 0 to check if process exists
            os.kill(self.current_job_pid, 0)
            return True
        except OSError:
            return False
    
    def should_switch_gpus(self, current_period: dict) -> bool:
        """Determine if GPU configuration should be switched"""
        required_gpus = set(current_period["allowed_gpus"])
        return self.current_gpus != required_gpus
    
    def run_scheduler(self):
        """Main scheduler loop"""
        self.log("GPU Scheduler started")
        
        while not self.shutdown_flag.is_set():
            try:
                current_period = self.get_current_time_period()
                required_gpus = current_period["allowed_gpus"]
                
                self.log(f"Current period: {current_period['description']}")
                
                # Check if we need to switch GPU configuration
                if self.should_switch_gpus(current_period) or not self.is_training_running():
                    self.log(f"Switching to {current_period['description']}")
                    
                    success = self.start_training(required_gpus)
                    if not success:
                        self.log("Failed to start training, will retry")
                
                # Wait before next check
                time.sleep(self.config["monitoring"]["check_interval"])
                
            except KeyboardInterrupt:
                self.log("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                self.log(f"Error in scheduler loop: {e}")
                time.sleep(self.config["monitoring"]["check_interval"])
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources and terminate training"""
        self.log("Cleaning up...")
        self.kill_existing_training()
        
        # Save final job state
        try:
            with open("gpu_scheduler_simple_state.json", "w") as f:
                json.dump(self.job_state, f, indent=2)
        except Exception as e:
            self.log(f"Error saving job state: {e}")
        
        self.log_file.close()
    
    def get_status(self) -> dict:
        """Get current scheduler status"""
        current_period = self.get_current_time_period()
        return {
            "current_time": datetime.datetime.now().isoformat(),
            "current_period": current_period,
            "job_running": self.is_training_running(),
            "current_job_pid": self.current_job_pid,
            "current_gpus": list(self.current_gpus),
            "job_state": self.job_state
        }


def main():
    parser = argparse.ArgumentParser(description="Simple GPU Scheduler")
    parser.add_argument("--config", default="gpu_scheduler_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--status", action="store_true",
                       help="Show current status and exit")
    parser.add_argument("--stop", action="store_true",
                       help="Stop all training processes and exit")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon process")
    
    args = parser.parse_args()
    
    scheduler = SimpleGPUScheduler(args.config)
    
    if args.status:
        status = scheduler.get_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.stop:
        print("Stopping all training processes...")
        scheduler.kill_existing_training()
        return
    
    try:
        scheduler.run_scheduler()
    except KeyboardInterrupt:
        print("\nScheduler stopped by user")


if __name__ == "__main__":
    main()