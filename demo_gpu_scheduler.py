#!/usr/bin/env python3
"""
GPU Scheduler Demo Script

This script demonstrates the GPU scheduler functionality with mock data.
"""

import json
import time
import datetime
from pathlib import Path


def create_demo_config():
    """Create a demo configuration for testing"""
    config = {
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
                "llama_config": "/workspace/accessory/configs/model/finetune/sg/llamaPeft_normBias.py",
                "tokenizer_path": "/workspace/tokenizer.model",
                "data_meta_path": "/workspace/data_example/PretrainMeta.json",
                "data_root": "/workspace/data_example"
            },
            "checkpoint_dir": "./checkpoints_scheduled",
            "resume_from_checkpoint": True
        },
        "monitoring": {
            "check_interval": 30,  # Shorter interval for demo
            "grace_period": 10,    # Shorter grace period for demo
            "max_restart_attempts": 3,
            "log_level": "INFO"
        }
    }
    
    with open("gpu_scheduler_config_demo.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Demo configuration created: gpu_scheduler_config_demo.json")
    return config


def show_current_time_period(config):
    """Show what the current time period would be"""
    now = datetime.datetime.now().time()
    print(f"🕐 Current time: {now.strftime('%H:%M:%S')}")
    
    for period_name, schedule in config["time_schedules"].items():
        start_time = datetime.time.fromisoformat(schedule["start_time"])
        end_time = datetime.time.fromisoformat(schedule["end_time"])
        
        # Handle overnight periods
        if start_time > end_time:
            if now >= start_time or now < end_time:
                print(f"📅 Current period: {period_name}")
                print(f"🎯 Allowed GPUs: {schedule['allowed_gpus']}")
                print(f"📝 Description: {schedule['description']}")
                return period_name, schedule
        else:
            if start_time <= now < end_time:
                print(f"📅 Current period: {period_name}")
                print(f"🎯 Allowed GPUs: {schedule['allowed_gpus']}")
                print(f"📝 Description: {schedule['description']}")
                return period_name, schedule
    
    print("❓ No matching time period found")
    return None, None


def simulate_schedule_changes():
    """Simulate how the schedule would change throughout the day"""
    print("\n🔄 Schedule Simulation (24-hour cycle):")
    print("-" * 50)
    
    config = json.load(open("gpu_scheduler_config_demo.json"))
    
    # Test different times throughout the day
    test_times = [
        "06:00", "08:00", "12:00", "16:00", "18:00", "20:00", "00:00", "04:00"
    ]
    
    for test_time in test_times:
        test_datetime = datetime.time.fromisoformat(test_time)
        
        # Determine which period this time falls into
        for period_name, schedule in config["time_schedules"].items():
            start_time = datetime.time.fromisoformat(schedule["start_time"])
            end_time = datetime.time.fromisoformat(schedule["end_time"])
            
            # Handle overnight periods
            if start_time > end_time:
                if test_datetime >= start_time or test_datetime < end_time:
                    print(f"⏰ {test_time} → {period_name} (GPUs: {schedule['allowed_gpus']})")
                    break
            else:
                if start_time <= test_datetime < end_time:
                    print(f"⏰ {test_time} → {period_name} (GPUs: {schedule['allowed_gpus']})")
                    break
        else:
            print(f"⏰ {test_time} → Unknown period")


def main():
    print("🚀 GPU Scheduler Demo")
    print("=" * 50)
    
    # Create demo configuration
    config = create_demo_config()
    
    print("\n📋 Configuration Overview:")
    print(f"   • Script path: {config['pretraining']['script_path']}")
    print(f"   • Check interval: {config['monitoring']['check_interval']} seconds")
    print(f"   • Grace period: {config['monitoring']['grace_period']} seconds")
    
    print("\n📅 Time Schedules:")
    for name, schedule in config["time_schedules"].items():
        print(f"   • {name}: {schedule['start_time']}-{schedule['end_time']} → GPUs {schedule['allowed_gpus']}")
    
    print("\n" + "=" * 50)
    
    # Show current time period
    show_current_time_period(config)
    
    # Simulate schedule changes
    simulate_schedule_changes()
    
    print("\n💡 Next Steps:")
    print("1. Review the demo configuration file")
    print("2. Update paths in the configuration to match your setup")
    print("3. Test the scheduler:")
    print("   python3 gpu_scheduler.py --config gpu_scheduler_config_demo.json --status")
    print("4. Start the scheduler:")
    print("   python3 gpu_scheduler.py --config gpu_scheduler_config_demo.json")
    print("5. Monitor with:")
    print("   python3 gpu_monitor.py --status")
    
    print("\n⚠️  Note: This demo uses shorter intervals for testing.")
    print("   For production use, run: python3 setup_gpu_scheduler.py")


if __name__ == "__main__":
    main()