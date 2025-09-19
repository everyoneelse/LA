#!/usr/bin/env python3
"""
GPU Scheduler for Time-based Pretraining Management

This script automatically manages GPU usage based on time periods:
- 8:00 AM - 6:00 PM: Only use GPUs 0,3
- 6:00 PM - 8:00 AM: Use all GPUs 0,1,2,3

Features:
- Automatic pretraining job switching based on time
- Graceful job termination and restart
- Configurable time periods and GPU allocations
- Process monitoring and recovery
- Logging and status reporting
"""

import os
import sys
import json
import time
import signal
import subprocess
import psutil
import datetime
import logging
import threading
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Set
import fcntl


class GPUScheduler:
    def __init__(self, config_path: str = "gpu_scheduler_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.current_job_pid = None
        self.current_gpus = set()
        self.shutdown_flag = threading.Event()
        self.job_lock = threading.Lock()
        
        # Setup logging
        self.setup_logging()
        
        # Track job state
        self.job_state = {
            "is_running": False,
            "start_time": None,
            "gpus_used": [],
            "process_pid": None,
            "restart_count": 0
        }
    
    def load_config(self) -> Dict:
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
                "max_restart_attempts": 3,
                "log_level": "INFO"
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
                print(f"Error loading config: {e}. Using default config.")
                
        # Create default config file
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default config at {self.config_path}")
        return default_config
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config["monitoring"]["log_level"])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gpu_scheduler.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('GPUScheduler')
    
    def get_current_time_period(self) -> Dict:
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
    
    def check_gpu_availability(self, gpu_ids: List[int]) -> List[int]:
        """Check which GPUs are actually available"""
        available_gpus = []
        try:
            # Try to get GPU info (simulate nvidia-smi)
            for gpu_id in gpu_ids:
                # Since nvidia-smi is not available, we'll assume GPUs are available
                # In real deployment, you would check actual GPU status
                available_gpus.append(gpu_id)
            return available_gpus
        except Exception as e:
            self.logger.warning(f"Could not check GPU availability: {e}")
            return gpu_ids  # Assume all requested GPUs are available
    
    def kill_existing_training(self):
        """Gracefully terminate existing training processes"""
        if self.current_job_pid:
            try:
                process = psutil.Process(self.current_job_pid)
                if process.is_running():
                    self.logger.info(f"Terminating training process {self.current_job_pid}")
                    
                    # Send SIGTERM first for graceful shutdown
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=self.config["monitoring"]["grace_period"])
                        self.logger.info("Training process terminated gracefully")
                    except psutil.TimeoutExpired:
                        # Force kill if graceful shutdown failed
                        self.logger.warning("Graceful shutdown timeout, force killing process")
                        process.kill()
                        process.wait()
                        
            except psutil.NoSuchProcess:
                self.logger.info("Training process already terminated")
            except Exception as e:
                self.logger.error(f"Error terminating training process: {e}")
            
            self.current_job_pid = None
        
        # Also kill any other training processes
        self.kill_training_processes()
    
    def kill_training_processes(self):
        """Kill any existing training processes"""
        try:
            # Look for Python processes running pretraining scripts
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] == 'python3' or proc.info['name'] == 'python':
                        cmdline = proc.info['cmdline']
                        if cmdline and any('pretrain' in arg for arg in cmdline):
                            self.logger.info(f"Killing existing training process {proc.info['pid']}")
                            proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            self.logger.error(f"Error killing training processes: {e}")
    
    def build_training_command(self, gpu_ids: List[int]) -> List[str]:
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
                    cmd.extend(["--resume", str(latest_checkpoint)])
        
        return cmd, gpu_str
    
    def find_latest_checkpoint(self, checkpoint_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint file"""
        try:
            checkpoints = list(checkpoint_dir.glob("checkpoint-*.pth"))
            if checkpoints:
                # Sort by modification time and return the latest
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                return latest
        except Exception as e:
            self.logger.error(f"Error finding latest checkpoint: {e}")
        return None
    
    def start_training(self, gpu_ids: List[int]) -> bool:
        """Start training with specified GPUs"""
        with self.job_lock:
            try:
                # Kill any existing training
                self.kill_existing_training()
                
                # Build command
                cmd, gpu_str = self.build_training_command(gpu_ids)
                
                self.logger.info(f"Starting training with GPUs {gpu_ids}")
                self.logger.info(f"Command: {' '.join(cmd)}")
                
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
                
                self.logger.info(f"Training started with PID {process.pid}")
                
                # Start output monitoring thread
                threading.Thread(
                    target=self.monitor_training_output,
                    args=(process,),
                    daemon=True
                ).start()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to start training: {e}")
                return False
    
    def monitor_training_output(self, process):
        """Monitor training process output"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Log training output (you might want to filter this)
                    if any(keyword in line.lower() for keyword in ['loss', 'epoch', 'iter', 'error', 'warning']):
                        self.logger.info(f"TRAINING: {line.strip()}")
                
                # Check if process is still running
                if process.poll() is not None:
                    break
        except Exception as e:
            self.logger.error(f"Error monitoring training output: {e}")
    
    def is_training_running(self) -> bool:
        """Check if training process is still running"""
        if not self.current_job_pid:
            return False
        
        try:
            process = psutil.Process(self.current_job_pid)
            return process.is_running()
        except psutil.NoSuchProcess:
            return False
    
    def should_switch_gpus(self, current_period: Dict) -> bool:
        """Determine if GPU configuration should be switched"""
        required_gpus = set(current_period["allowed_gpus"])
        return self.current_gpus != required_gpus
    
    def run_scheduler(self):
        """Main scheduler loop"""
        self.logger.info("GPU Scheduler started")
        
        while not self.shutdown_flag.is_set():
            try:
                current_period = self.get_current_time_period()
                required_gpus = current_period["allowed_gpus"]
                
                self.logger.debug(f"Current period: {current_period['description']}")
                self.logger.debug(f"Required GPUs: {required_gpus}")
                
                # Check if we need to switch GPU configuration
                if self.should_switch_gpus(current_period) or not self.is_training_running():
                    self.logger.info(f"Switching to {current_period['description']}")
                    
                    # Check GPU availability
                    available_gpus = self.check_gpu_availability(required_gpus)
                    
                    if available_gpus:
                        success = self.start_training(available_gpus)
                        if not success:
                            self.logger.error("Failed to start training, will retry")
                    else:
                        self.logger.warning(f"No GPUs available from required set: {required_gpus}")
                
                # Wait before next check
                time.sleep(self.config["monitoring"]["check_interval"])
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(self.config["monitoring"]["check_interval"])
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources and terminate training"""
        self.logger.info("Cleaning up...")
        self.kill_existing_training()
        
        # Save final job state
        try:
            with open("gpu_scheduler_state.json", "w") as f:
                json.dump(self.job_state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving job state: {e}")
    
    def get_status(self) -> Dict:
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
    parser = argparse.ArgumentParser(description="GPU Scheduler for Time-based Pretraining")
    parser.add_argument("--config", default="gpu_scheduler_config.json", 
                       help="Path to configuration file")
    parser.add_argument("--status", action="store_true",
                       help="Show current status and exit")
    parser.add_argument("--stop", action="store_true",
                       help="Stop all training processes and exit")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon process")
    
    args = parser.parse_args()
    
    scheduler = GPUScheduler(args.config)
    
    if args.status:
        status = scheduler.get_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.stop:
        print("Stopping all training processes...")
        scheduler.kill_existing_training()
        return
    
    if args.daemon:
        # Simple daemon mode (in production, use proper daemonization)
        try:
            scheduler.run_scheduler()
        except KeyboardInterrupt:
            print("Scheduler stopped by user")
    else:
        # Interactive mode
        try:
            scheduler.run_scheduler()
        except KeyboardInterrupt:
            print("\nScheduler stopped by user")


if __name__ == "__main__":
    main()