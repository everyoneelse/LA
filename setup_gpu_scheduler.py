#!/usr/bin/env python3
"""
Setup script for GPU Scheduler

This script helps you configure the GPU scheduler for your specific environment.
It will create the configuration file with your specific paths and settings.
"""

import json
import os
import sys
from pathlib import Path


def find_files_in_workspace():
    """Find relevant files in the workspace"""
    workspace = Path("/workspace")
    
    results = {
        "llama_configs": [],
        "tokenizer_paths": [],
        "data_meta_paths": [],
        "data_roots": []
    }
    
    # Look for config files
    for config_file in workspace.rglob("*.json"):
        if any(keyword in config_file.name.lower() for keyword in ["config", "llama", "model"]):
            results["llama_configs"].append(str(config_file))
    
    # Look for tokenizer files
    for tokenizer_file in workspace.rglob("*tokenizer*"):
        if tokenizer_file.is_file():
            results["tokenizer_paths"].append(str(tokenizer_file))
    
    # Look for data meta files
    for meta_file in workspace.rglob("*.json"):
        if any(keyword in meta_file.name.lower() for keyword in ["meta", "data"]):
            results["data_meta_paths"].append(str(meta_file))
    
    # Look for potential data directories
    for data_dir in workspace.rglob("data*"):
        if data_dir.is_dir() and any(data_dir.iterdir()):
            results["data_roots"].append(str(data_dir))
    
    return results


def interactive_setup():
    """Interactive setup process"""
    print("=" * 60)
    print("GPU Scheduler Configuration Setup")
    print("=" * 60)
    print()
    
    # Find existing files
    print("🔍 Scanning workspace for relevant files...")
    found_files = find_files_in_workspace()
    
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
    
    # Configure time schedules
    print("\n📅 Time Schedule Configuration")
    print("Current schedule:")
    print(f"  Day shift (8AM-6PM): GPUs {config['time_schedules']['day_shift']['allowed_gpus']}")
    print(f"  Night shift (6PM-8AM): GPUs {config['time_schedules']['night_shift']['allowed_gpus']}")
    
    modify_schedule = input("\nDo you want to modify the time schedule? (y/N): ").lower().startswith('y')
    
    if modify_schedule:
        # Day shift configuration
        print("\n⏰ Day Shift Configuration (currently 8AM-6PM, GPUs 0,3)")
        day_start = input("Day shift start time (HH:MM format, default 08:00): ").strip() or "08:00"
        day_end = input("Day shift end time (HH:MM format, default 18:00): ").strip() or "18:00"
        day_gpus = input("Day shift GPUs (comma-separated, default 0,3): ").strip() or "0,3"
        
        try:
            day_gpu_list = [int(x.strip()) for x in day_gpus.split(',')]
            config['time_schedules']['day_shift'].update({
                "start_time": day_start,
                "end_time": day_end,
                "allowed_gpus": day_gpu_list,
                "description": f"Day shift: {day_start} - {day_end}, GPUs {day_gpu_list}"
            })
        except ValueError:
            print("Invalid GPU list, using default [0, 3]")
        
        # Night shift configuration  
        print("\n🌙 Night Shift Configuration (currently 6PM-8AM, all GPUs)")
        night_start = input("Night shift start time (HH:MM format, default 18:00): ").strip() or "18:00"
        night_end = input("Night shift end time (HH:MM format, default 08:00): ").strip() or "08:00"
        night_gpus = input("Night shift GPUs (comma-separated, default 0,1,2,3): ").strip() or "0,1,2,3"
        
        try:
            night_gpu_list = [int(x.strip()) for x in night_gpus.split(',')]
            config['time_schedules']['night_shift'].update({
                "start_time": night_start,
                "end_time": night_end,
                "allowed_gpus": night_gpu_list,
                "description": f"Night shift: {night_start} - {night_end}, GPUs {night_gpu_list}"
            })
        except ValueError:
            print("Invalid GPU list, using default [0, 1, 2, 3]")
    
    # Configure pretraining paths
    print("\n🔧 Pretraining Configuration")
    
    # LLaMA config
    if found_files["llama_configs"]:
        print(f"\nFound {len(found_files['llama_configs'])} potential config files:")
        for i, config_path in enumerate(found_files["llama_configs"][:5]):
            print(f"  {i+1}. {config_path}")
        
        choice = input(f"Select config file (1-{min(5, len(found_files['llama_configs']))}) or enter custom path: ").strip()
        
        try:
            if choice.isdigit() and 1 <= int(choice) <= len(found_files["llama_configs"]):
                config["pretraining"]["required_args"]["llama_config"] = found_files["llama_configs"][int(choice)-1]
            else:
                config["pretraining"]["required_args"]["llama_config"] = choice
        except (ValueError, IndexError):
            print("Invalid selection, you'll need to set this manually later")
    else:
        llama_config = input("Enter path to LLaMA config file: ").strip()
        if llama_config:
            config["pretraining"]["required_args"]["llama_config"] = llama_config
    
    # Tokenizer path
    if found_files["tokenizer_paths"]:
        print(f"\nFound {len(found_files['tokenizer_paths'])} potential tokenizer files:")
        for i, tokenizer_path in enumerate(found_files["tokenizer_paths"][:5]):
            print(f"  {i+1}. {tokenizer_path}")
        
        choice = input(f"Select tokenizer file (1-{min(5, len(found_files['tokenizer_paths']))}) or enter custom path: ").strip()
        
        try:
            if choice.isdigit() and 1 <= int(choice) <= len(found_files["tokenizer_paths"]):
                config["pretraining"]["required_args"]["tokenizer_path"] = found_files["tokenizer_paths"][int(choice)-1]
            else:
                config["pretraining"]["required_args"]["tokenizer_path"] = choice
        except (ValueError, IndexError):
            print("Invalid selection, you'll need to set this manually later")
    else:
        tokenizer_path = input("Enter path to tokenizer file: ").strip()
        if tokenizer_path:
            config["pretraining"]["required_args"]["tokenizer_path"] = tokenizer_path
    
    # Data meta path
    if found_files["data_meta_paths"]:
        print(f"\nFound {len(found_files['data_meta_paths'])} potential data meta files:")
        for i, meta_path in enumerate(found_files["data_meta_paths"][:5]):
            print(f"  {i+1}. {meta_path}")
        
        choice = input(f"Select data meta file (1-{min(5, len(found_files['data_meta_paths']))}) or enter custom path: ").strip()
        
        try:
            if choice.isdigit() and 1 <= int(choice) <= len(found_files["data_meta_paths"]):
                config["pretraining"]["required_args"]["data_meta_path"] = found_files["data_meta_paths"][int(choice)-1]
            else:
                config["pretraining"]["required_args"]["data_meta_path"] = choice
        except (ValueError, IndexError):
            print("Invalid selection, you'll need to set this manually later")
    else:
        data_meta_path = input("Enter path to data meta file: ").strip()
        if data_meta_path:
            config["pretraining"]["required_args"]["data_meta_path"] = data_meta_path
    
    # Data root
    if found_files["data_roots"]:
        print(f"\nFound {len(found_files['data_roots'])} potential data directories:")
        for i, data_root in enumerate(found_files["data_roots"][:5]):
            print(f"  {i+1}. {data_root}")
        
        choice = input(f"Select data root directory (1-{min(5, len(found_files['data_roots']))}) or enter custom path: ").strip()
        
        try:
            if choice.isdigit() and 1 <= int(choice) <= len(found_files["data_roots"]):
                config["pretraining"]["required_args"]["data_root"] = found_files["data_roots"][int(choice)-1]
            else:
                config["pretraining"]["required_args"]["data_root"] = choice
        except (ValueError, IndexError):
            print("Invalid selection, you'll need to set this manually later")
    else:
        data_root = input("Enter path to data root directory: ").strip()
        if data_root:
            config["pretraining"]["required_args"]["data_root"] = data_root
    
    # Training parameters
    print("\n⚙️  Training Parameters")
    modify_params = input("Do you want to modify training parameters? (y/N): ").lower().startswith('y')
    
    if modify_params:
        batch_size = input("Batch size per GPU (default 4): ").strip()
        if batch_size.isdigit():
            # Update batch_size in base_args
            for i, arg in enumerate(config["pretraining"]["base_args"]):
                if arg == "--batch_size" and i + 1 < len(config["pretraining"]["base_args"]):
                    config["pretraining"]["base_args"][i + 1] = batch_size
                    break
        
        accum_iter = input("Accumulation iterations (default 4): ").strip()
        if accum_iter.isdigit():
            # Update accum_iter in base_args
            for i, arg in enumerate(config["pretraining"]["base_args"]):
                if arg == "--accum_iter" and i + 1 < len(config["pretraining"]["base_args"]):
                    config["pretraining"]["base_args"][i + 1] = accum_iter
                    break
        
        learning_rate = input("Learning rate (default 0.001): ").strip()
        if learning_rate:
            try:
                float(learning_rate)
                # Update lr in base_args
                for i, arg in enumerate(config["pretraining"]["base_args"]):
                    if arg == "--lr" and i + 1 < len(config["pretraining"]["base_args"]):
                        config["pretraining"]["base_args"][i + 1] = learning_rate
                        break
            except ValueError:
                print("Invalid learning rate, using default")
    
    return config


def main():
    print("Setting up GPU Scheduler...")
    
    # Check if config already exists
    config_path = "gpu_scheduler_config.json"
    if os.path.exists(config_path):
        overwrite = input(f"Configuration file {config_path} already exists. Overwrite? (y/N): ")
        if not overwrite.lower().startswith('y'):
            print("Setup cancelled.")
            return
    
    # Run interactive setup
    config = interactive_setup()
    
    # Save configuration
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuration saved to {config_path}")
        print("\n📋 Next steps:")
        print("1. Review and edit the configuration file if needed")
        print("2. Make sure all file paths in the configuration are correct")
        print("3. Run the scheduler:")
        print(f"   python3 gpu_scheduler.py --config {config_path}")
        print("\n📊 To check status:")
        print(f"   python3 gpu_scheduler.py --config {config_path} --status")
        print("\n🛑 To stop all training:")
        print(f"   python3 gpu_scheduler.py --config {config_path} --stop")
        
        # Validate configuration
        missing_required = []
        for key, value in config["pretraining"]["required_args"].items():
            if not value:
                missing_required.append(key)
        
        if missing_required:
            print(f"\n⚠️  Warning: The following required arguments are not set:")
            for arg in missing_required:
                print(f"   - {arg}")
            print("   Please edit the configuration file to set these values.")
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())