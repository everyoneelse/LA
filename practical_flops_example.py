#!/usr/bin/env python3
"""
Practical FLOPs Monitoring Example

This file shows exactly how to integrate runtime FLOPs monitoring into your
existing InternLM2 training code. Copy the relevant parts to your training script.
"""

# ============================================================================
# STEP 1: MINIMAL INTEGRATION (3 lines of code)
# ============================================================================

def minimal_integration_example():
    """Show the absolute minimum changes needed to add FLOPs monitoring."""
    
    print("📋 MINIMAL INTEGRATION EXAMPLE")
    print("="*50)
    
    example_code = '''
# Your existing training loop:
for batch_idx, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Modified with FLOPs monitoring (ADD THESE 3 LINES):
from runtime_flops_profiler import RuntimeFLOPsMonitor
flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')  # ADD THIS LINE
flops_monitor.start_monitoring()                           # ADD THIS LINE

for batch_idx, (inputs, targets) in enumerate(train_loader):
    # REPLACE the forward/backward/step with this single call:
    result = flops_monitor.measure_step(inputs, targets, optimizer)  # REPLACE
    
    # Optional: Log FLOPs every 100 steps
    if batch_idx % 100 == 0:
        step_flops = result['step_flops']
        total_flops = result['total_flops']
        print(f"Step {batch_idx}: {flops_monitor.format_flops(step_flops)} "
              f"(Total: {flops_monitor.format_flops(total_flops)})")
'''
    
    print(example_code)
    print("✅ That's it! Just 3 lines of changes to get real FLOPs monitoring.")


# ============================================================================
# STEP 2: INTERNLM2 SPECIFIC INTEGRATION
# ============================================================================

def internlm2_integration_example():
    """Show how to integrate with InternLM2 training specifically."""
    
    print("\n📋 INTERNLM2 SPECIFIC INTEGRATION")
    print("="*50)
    
    # Show the exact modification for accessory/main_pretrain.py
    integration_code = '''
# In accessory/main_pretrain.py, find the training loop and modify:

# Original code (around line 200-250):
def train_one_epoch(model, train_loader, optimizer, criterion, args):
    model.train()
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)
        
        outputs = model(input_ids)
        loss = criterion(outputs.logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % args.log_interval == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

# Modified with FLOPs monitoring:
from runtime_flops_profiler import RuntimeFLOPsMonitor

def train_one_epoch(model, train_loader, optimizer, criterion, args):
    model.train()
    
    # Initialize FLOPs monitor
    flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')
    flops_monitor.start_monitoring()
    
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(args.device)
        labels = batch['labels'].to(args.device)
        
        # Measure FLOPs for this step (replaces manual forward/backward)
        flops_result = flops_monitor.measure_step(input_ids, labels, optimizer)
        
        # Extract loss from the result (if needed for logging)
        # Note: The measure_step already did forward/backward/step
        
        if step % args.log_interval == 0:
            step_flops = flops_result['step_flops']
            total_flops = flops_result['total_flops']
            avg_flops = flops_result['avg_flops_per_step']
            
            print(f"Step {step:6d} | "
                  f"FLOPs: {flops_monitor.format_flops(step_flops):>10} | "
                  f"Avg: {flops_monitor.format_flops(avg_flops):>10} | "
                  f"Total: {flops_monitor.format_flops(total_flops):>12}")
        
        # Detailed breakdown every 1000 steps
        if step % 1000 == 0 and step > 0:
            breakdown = flops_result.get('breakdown', {})
            print("\\n🔍 FLOPs Breakdown:")
            for op_type, flops in breakdown.items():
                pct = (flops / step_flops) * 100 if step_flops > 0 else 0
                print(f"  {op_type:15}: {flops_monitor.format_flops(flops):>10} ({pct:.1f}%)")
            print()
    
    # Final statistics
    final_stats = flops_monitor.get_statistics()
    print(f"\\n📊 Training Epoch Summary:")
    print(f"  Total Steps: {final_stats['total_steps']:,}")
    print(f"  Total FLOPs: {flops_monitor.format_flops(final_stats['total_flops'])}")
    print(f"  Avg FLOPs/step: {flops_monitor.format_flops(final_stats['avg_flops_per_step'])}")
    print(f"  Compute Rate: {flops_monitor.format_flops(final_stats['flops_per_second'])}/s")
    
    flops_monitor.cleanup()
    return final_stats
'''
    
    print(integration_code)


# ============================================================================
# STEP 3: ADVANCED MONITORING WITH CHECKPOINTS
# ============================================================================

def advanced_monitoring_example():
    """Show advanced monitoring with checkpoint integration."""
    
    print("\n📋 ADVANCED MONITORING WITH CHECKPOINTS")
    print("="*50)
    
    advanced_code = '''
# Advanced integration that saves FLOPs info in checkpoints

import torch
import json
from datetime import datetime

class FLOPsAwareTrainer:
    def __init__(self, model, optimizer, train_loader, save_dir):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.save_dir = save_dir
        
        # Initialize FLOPs monitoring
        self.flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')
        self.flops_monitor.start_monitoring()
        
        # Training statistics
        self.epoch_flops_history = []
        self.training_start_time = datetime.now()
    
    def train_epoch(self, epoch):
        """Train one epoch with comprehensive FLOPs tracking."""
        self.model.train()
        epoch_start_flops = self.flops_monitor.total_flops
        
        print(f"\\n🚀 Starting Epoch {epoch}")
        print("-" * 50)
        
        for step, batch in enumerate(self.train_loader):
            inputs, targets = batch
            
            # Measure step FLOPs
            result = self.flops_monitor.measure_step(inputs, targets, self.optimizer)
            
            # Periodic logging
            if step % 100 == 0:
                self.log_training_progress(step, result)
            
            # Save checkpoint with FLOPs info
            if step % 5000 == 0 and step > 0:
                self.save_checkpoint_with_flops(epoch, step)
        
        # Epoch summary
        epoch_end_flops = self.flops_monitor.total_flops
        epoch_flops = epoch_end_flops - epoch_start_flops
        self.epoch_flops_history.append(epoch_flops)
        
        print(f"\\n📊 Epoch {epoch} Complete:")
        print(f"  Epoch FLOPs: {self.flops_monitor.format_flops(epoch_flops)}")
        print(f"  Total FLOPs: {self.flops_monitor.format_flops(epoch_end_flops)}")
        
        return epoch_flops
    
    def log_training_progress(self, step, flops_result):
        """Log training progress with FLOPs information."""
        step_flops = flops_result['step_flops']
        total_flops = flops_result['total_flops']
        avg_flops = flops_result['avg_flops_per_step']
        
        # Calculate efficiency metrics
        stats = self.flops_monitor.get_statistics()
        throughput = stats.get('flops_per_second', 0)
        
        print(f"Step {step:6d} | "
              f"Step: {self.flops_monitor.format_flops(step_flops):>8} | "
              f"Avg: {self.flops_monitor.format_flops(avg_flops):>8} | "
              f"Rate: {self.flops_monitor.format_flops(throughput):>8}/s")
    
    def save_checkpoint_with_flops(self, epoch, step):
        """Save checkpoint including FLOPs statistics."""
        flops_stats = self.flops_monitor.get_statistics()
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'flops_info': {
                'total_flops': flops_stats['total_flops'],
                'total_steps': flops_stats['total_steps'],
                'avg_flops_per_step': flops_stats['avg_flops_per_step'],
                'flops_per_second': flops_stats.get('flops_per_second', 0),
                'epoch_flops_history': self.epoch_flops_history,
                'training_duration_seconds': flops_stats.get('elapsed_time_seconds', 0)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = f"{self.save_dir}/checkpoint_epoch_{epoch}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save FLOPs info as JSON for easy analysis
        flops_json_path = f"{self.save_dir}/flops_stats_epoch_{epoch}_step_{step}.json"
        with open(flops_json_path, 'w') as f:
            json.dump(checkpoint['flops_info'], f, indent=2)
        
        print(f"💾 Checkpoint saved with FLOPs info: {checkpoint_path}")
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        stats = self.flops_monitor.get_statistics()
        
        report = {
            'training_summary': {
                'total_training_steps': stats['total_steps'],
                'total_flops': stats['total_flops'],
                'average_flops_per_step': stats['avg_flops_per_step'],
                'compute_throughput_flops_per_second': stats.get('flops_per_second', 0),
                'training_duration_seconds': stats.get('elapsed_time_seconds', 0),
                'epochs_completed': len(self.epoch_flops_history)
            },
            'epoch_flops_breakdown': [
                {
                    'epoch': i + 1,
                    'flops': flops,
                    'flops_formatted': self.flops_monitor.format_flops(flops)
                }
                for i, flops in enumerate(self.epoch_flops_history)
            ],
            'performance_metrics': {
                'min_step_flops': min(self.flops_monitor.step_flops_history) if self.flops_monitor.step_flops_history else 0,
                'max_step_flops': max(self.flops_monitor.step_flops_history) if self.flops_monitor.step_flops_history else 0,
                'flops_std_deviation': np.std(self.flops_monitor.step_flops_history) if len(self.flops_monitor.step_flops_history) > 1 else 0
            }
        }
        
        return report
    
    def cleanup(self):
        """Clean up monitoring resources."""
        self.flops_monitor.cleanup()

# Usage example:
def main():
    # Your model, optimizer, data loader setup
    model = YourInternLM2Model()
    optimizer = torch.optim.AdamW(model.parameters())
    train_loader = YourDataLoader()
    
    # Create trainer with FLOPs monitoring
    trainer = FLOPsAwareTrainer(model, optimizer, train_loader, save_dir='./checkpoints')
    
    # Training loop
    for epoch in range(10):
        trainer.train_epoch(epoch)
    
    # Generate final report
    report = trainer.generate_training_report()
    
    # Save final report
    with open('./final_training_flops_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    trainer.cleanup()
    print("🎉 Training completed with full FLOPs monitoring!")

if __name__ == "__main__":
    main()
'''
    
    print(advanced_code)


# ============================================================================
# STEP 4: COMPARISON WITH THEORETICAL ESTIMATES
# ============================================================================

def comparison_example():
    """Show how to compare runtime FLOPs with theoretical estimates."""
    
    print("\n📋 RUNTIME vs THEORETICAL COMPARISON")
    print("="*50)
    
    comparison_code = '''
# Compare runtime measurements with theoretical calculations

from internlm2_compute_calculator import InternLM2ComputeCalculator
from runtime_flops_profiler import RuntimeFLOPsMonitor

def compare_runtime_vs_theoretical(model, config_path, batch_size, seq_len):
    """Compare runtime FLOPs measurement with theoretical calculation."""
    
    print("🔍 Comparing Runtime vs Theoretical FLOPs")
    print("="*60)
    
    # Theoretical calculation
    theoretical_calc = InternLM2ComputeCalculator(config_path=config_path)
    theoretical_forward = theoretical_calc.calculate_forward_flops(batch_size, seq_len)
    theoretical_total = theoretical_calc.calculate_total_flops_per_step(batch_size, seq_len)
    
    print(f"📊 Theoretical Estimates:")
    print(f"  Forward pass:  {theoretical_calc.format_number(theoretical_forward, 'FLOPs')}")
    print(f"  Total step:    {theoretical_calc.format_number(theoretical_total, 'FLOPs')}")
    
    # Runtime measurement
    flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')
    flops_monitor.start_monitoring()
    
    # Create sample inputs
    import torch
    inputs = torch.randint(0, 32000, (batch_size, seq_len))
    targets = torch.randint(0, 32000, (batch_size, seq_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Measure actual runtime FLOPs
    runtime_result = flops_monitor.measure_step(inputs, targets, optimizer)
    runtime_flops = runtime_result['step_flops']
    
    print(f"\\n📈 Runtime Measurement:")
    print(f"  Actual FLOPs:  {flops_monitor.format_flops(runtime_flops)}")
    
    # Comparison
    print(f"\\n🔄 Comparison:")
    ratio = runtime_flops / theoretical_total if theoretical_total > 0 else 0
    difference = abs(runtime_flops - theoretical_total)
    
    print(f"  Runtime/Theoretical ratio: {ratio:.2f}")
    print(f"  Absolute difference:       {flops_monitor.format_flops(difference)}")
    print(f"  Relative difference:       {abs(ratio - 1) * 100:.1f}%")
    
    if ratio < 0.8:
        print("  ⚠️  Runtime significantly lower - possible missing operations")
    elif ratio > 1.2:
        print("  ⚠️  Runtime significantly higher - possible overcounting")
    else:
        print("  ✅ Runtime and theoretical estimates are reasonably close")
    
    # Breakdown analysis
    breakdown = runtime_result.get('breakdown', {})
    if breakdown:
        print(f"\\n🔍 Runtime FLOPs Breakdown:")
        for op_type, flops in breakdown.items():
            pct = (flops / runtime_flops) * 100 if runtime_flops > 0 else 0
            print(f"  {op_type:15}: {flops_monitor.format_flops(flops):>10} ({pct:.1f}%)")
    
    flops_monitor.cleanup()
    
    return {
        'theoretical_flops': theoretical_total,
        'runtime_flops': runtime_flops,
        'ratio': ratio,
        'difference_percent': abs(ratio - 1) * 100
    }

# Example usage:
config_path = "internlm2_scaling/configs/internlm2-chat-1386M-h16-L16.json"
model = load_your_internlm2_model(config_path)  # Your model loading function
comparison_result = compare_runtime_vs_theoretical(model, config_path, batch_size=4, seq_len=2048)
'''
    
    print(comparison_code)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all examples."""
    print("🚀 PRACTICAL FLOPs MONITORING EXAMPLES")
    print("="*60)
    print("This file shows exactly how to add runtime FLOPs monitoring")
    print("to your InternLM2 training code with real examples.")
    print()
    
    minimal_integration_example()
    internlm2_integration_example()
    advanced_monitoring_example()
    comparison_example()
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS:")
    print("1. Copy runtime_flops_profiler.py to your training project")
    print("2. Choose integration method (minimal, advanced, or custom)")
    print("3. Test with small model first to verify accuracy")
    print("4. Scale up to full training runs")
    print("5. Use data to optimize training efficiency")
    print()
    print("📚 Files to use:")
    print("- runtime_flops_profiler.py (main profiler)")
    print("- training_with_flops_template.py (templates)")
    print("- internlm2_flops_integration.py (InternLM2 specific)")
    print()
    print("Ready to measure real training compute! 🚀")


if __name__ == "__main__":
    main()