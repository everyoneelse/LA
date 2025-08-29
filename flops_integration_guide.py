#!/usr/bin/env python3
"""
FLOPs Integration Guide for InternLM2 Training

This module provides integration examples and templates for adding runtime FLOPs
monitoring to your actual training code. Since we can't install PyTorch in this
environment, this serves as a comprehensive guide with working examples.
"""

# Template code for integration (requires PyTorch in actual usage)

INTEGRATION_TEMPLATE = '''
# ============================================================================
# INTEGRATION TEMPLATE: Add this to your training script
# ============================================================================

from runtime_flops_profiler import RuntimeFLOPsMonitor
import torch

class TrainingWithFLOPsMonitoring:
    """Enhanced training loop with FLOPs monitoring."""
    
    def __init__(self, model, optimizer, train_loader, method='hooks'):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        
        # Initialize FLOPs monitor
        self.flops_monitor = RuntimeFLOPsMonitor(model, method=method)
        self.flops_monitor.start_monitoring()
        
        # Statistics tracking
        self.step_count = 0
        self.epoch_flops = 0
        self.total_training_flops = 0
        
    def train_epoch(self, epoch):
        """Train one epoch with FLOPs monitoring."""
        self.model.train()
        self.epoch_flops = 0
        epoch_loss = 0.0
        
        print(f"\\n📈 Epoch {epoch} - Starting FLOPs monitoring")
        print("-" * 60)
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Measure FLOPs for this training step
            flops_result = self.flops_monitor.measure_step(
                inputs, targets, self.optimizer
            )
            
            # Track statistics
            self.step_count += 1
            step_flops = flops_result['step_flops']
            self.epoch_flops += step_flops
            self.total_training_flops += step_flops
            
            # Periodic reporting
            if batch_idx % 100 == 0:
                avg_flops = flops_result['avg_flops_per_step']
                print(f"Step {self.step_count:6d} | "
                      f"Batch FLOPs: {self.flops_monitor.format_flops(step_flops)} | "
                      f"Avg: {self.flops_monitor.format_flops(avg_flops)}")
                
                # Detailed breakdown every 500 steps
                if batch_idx % 500 == 0 and batch_idx > 0:
                    self.print_flops_breakdown(flops_result)
        
        # Epoch summary
        print(f"\\n📊 Epoch {epoch} Summary:")
        print(f"  Epoch FLOPs: {self.flops_monitor.format_flops(self.epoch_flops)}")
        print(f"  Total FLOPs: {self.flops_monitor.format_flops(self.total_training_flops)}")
        
        return self.epoch_flops
    
    def print_flops_breakdown(self, flops_result):
        """Print detailed FLOPs breakdown."""
        print("\\n🔍 FLOPs Breakdown:")
        breakdown = flops_result.get('breakdown', {})
        for op_type, flops in breakdown.items():
            percentage = (flops / flops_result['total_flops']) * 100 if flops_result['total_flops'] > 0 else 0
            print(f"  {op_type:15}: {self.flops_monitor.format_flops(flops):>12} ({percentage:.1f}%)")
    
    def get_training_summary(self):
        """Get comprehensive training summary."""
        stats = self.flops_monitor.get_statistics()
        
        summary = {
            'total_training_steps': self.step_count,
            'total_training_flops': self.total_training_flops,
            'average_flops_per_step': stats.get('avg_flops_per_step', 0),
            'flops_per_second': stats.get('flops_per_second', 0),
            'training_time_seconds': stats.get('elapsed_time_seconds', 0),
        }
        
        return summary
    
    def cleanup(self):
        """Clean up monitoring resources."""
        self.flops_monitor.cleanup()


# ============================================================================
# EXAMPLE USAGE IN YOUR TRAINING SCRIPT
# ============================================================================

def main_training_with_flops():
    """Example of how to integrate FLOPs monitoring into training."""
    
    # Your existing model, optimizer, data loader setup
    model = YourModel()  # Replace with your actual model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_loader = YourDataLoader()  # Replace with your data loader
    
    # Create training instance with FLOPs monitoring
    trainer = TrainingWithFLOPsMonitoring(
        model=model,
        optimizer=optimizer, 
        train_loader=train_loader,
        method='hooks'  # or 'profiler'
    )
    
    # Training loop
    num_epochs = 10
    total_flops_per_epoch = []
    
    for epoch in range(num_epochs):
        epoch_flops = trainer.train_epoch(epoch)
        total_flops_per_epoch.append(epoch_flops)
        
        # Optional: Save checkpoint with FLOPs info
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'flops_info': trainer.get_training_summary()
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
    
    # Final summary
    final_summary = trainer.get_training_summary()
    print("\\n" + "="*80)
    print("🏁 FINAL TRAINING SUMMARY")
    print("="*80)
    print(f"Total training steps: {final_summary['total_training_steps']:,}")
    print(f"Total FLOPs: {trainer.flops_monitor.format_flops(final_summary['total_training_flops'])}")
    print(f"Average FLOPs/step: {trainer.flops_monitor.format_flops(final_summary['average_flops_per_step'])}")
    print(f"Training time: {final_summary['training_time_seconds']:.1f} seconds")
    print(f"Compute throughput: {trainer.flops_monitor.format_flops(final_summary['flops_per_second'])}/s")
    
    # Cleanup
    trainer.cleanup()
    
    return final_summary

if __name__ == "__main__":
    main_training_with_flops()
'''

SPECIFIC_INTERNLM2_INTEGRATION = '''
# ============================================================================
# SPECIFIC INTEGRATION FOR INTERNLM2 TRAINING
# ============================================================================

# Add this to your InternLM2 training script (e.g., accessory/main_pretrain.py)

import sys
sys.path.append('/path/to/flops/profiler')  # Adjust path as needed
from runtime_flops_profiler import RuntimeFLOPsMonitor

def train_with_flops_monitoring(model, train_loader, optimizer, args):
    """Modified training function with FLOPs monitoring."""
    
    # Initialize FLOPs monitor
    flops_monitor = RuntimeFLOPsMonitor(
        model=model,
        method='hooks',  # Use hooks for minimal overhead
        use_cuda=torch.cuda.is_available()
    )
    flops_monitor.start_monitoring()
    
    model.train()
    total_flops = 0
    
    for step, batch in enumerate(train_loader):
        # Prepare batch (your existing code)
        inputs = batch['input_ids'].to(args.device)
        targets = batch['labels'].to(args.device)
        
        # Measure FLOPs for this step
        flops_result = flops_monitor.measure_step(inputs, targets, optimizer)
        step_flops = flops_result['step_flops']
        total_flops += step_flops
        
        # Your existing logging
        if step % args.log_interval == 0:
            print(f"Step {step:6d} | "
                  f"FLOPs: {flops_monitor.format_flops(step_flops)} | "
                  f"Total: {flops_monitor.format_flops(total_flops)}")
        
        # Detailed FLOPs logging
        if step % (args.log_interval * 10) == 0 and step > 0:
            breakdown = flops_result.get('breakdown', {})
            print("FLOPs breakdown:")
            for op_type, flops in breakdown.items():
                print(f"  {op_type}: {flops_monitor.format_flops(flops)}")
    
    # Final statistics
    final_stats = flops_monitor.get_statistics()
    print(f"\\nTraining completed:")
    print(f"Total FLOPs: {flops_monitor.format_flops(final_stats['total_flops'])}")
    print(f"Average FLOPs/step: {flops_monitor.format_flops(final_stats['avg_flops_per_step'])}")
    
    flops_monitor.cleanup()
    return final_stats

# ============================================================================
# INTEGRATION WITH EXISTING INTERNLM2 CODE
# ============================================================================

# In your main_pretrain.py or similar file, modify the training loop:

def modified_training_loop():
    """Example of modifying existing InternLM2 training loop."""
    
    # ... existing setup code ...
    
    # Add FLOPs monitoring
    from runtime_flops_profiler import RuntimeFLOPsMonitor
    flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')
    flops_monitor.start_monitoring()
    
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            # Your existing forward/backward pass
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'])
            
            # Measure FLOPs (this includes the forward/backward pass)
            flops_result = flops_monitor.measure_step(
                batch['input_ids'], 
                batch['labels'], 
                optimizer
            )
            
            # Add FLOPs to your existing logging
            if step % log_interval == 0:
                current_flops = flops_result['step_flops']
                total_flops = flops_result['total_flops']
                
                print(f"Epoch {epoch}, Step {step}: "
                      f"Loss {loss:.4f}, "
                      f"FLOPs {flops_monitor.format_flops(current_flops)}, "
                      f"Total {flops_monitor.format_flops(total_flops)}")
    
    flops_monitor.cleanup()
'''

def create_integration_examples():
    """Create example integration files."""
    
    print("🔧 Creating FLOPs Integration Examples")
    print("=" * 50)
    
    # Write integration template
    with open('/workspace/training_with_flops_template.py', 'w') as f:
        f.write(INTEGRATION_TEMPLATE)
    
    # Write InternLM2 specific integration
    with open('/workspace/internlm2_flops_integration.py', 'w') as f:
        f.write(SPECIFIC_INTERNLM2_INTEGRATION)
    
    print("✅ Created integration files:")
    print("  - training_with_flops_template.py (General template)")
    print("  - internlm2_flops_integration.py (InternLM2 specific)")


def print_integration_guide():
    """Print comprehensive integration guide."""
    
    guide = '''
🚀 RUNTIME FLOPs MONITORING INTEGRATION GUIDE
============================================

## 📋 Overview
This guide shows how to integrate runtime FLOPs monitoring into your actual 
InternLM2 training code to measure real compute usage during training.

## 🛠️ Setup Steps

### 1. Copy the profiler to your project
```bash
cp runtime_flops_profiler.py /path/to/your/training/project/
```

### 2. Install dependencies (if not already available)
```bash
pip install torch numpy  # Usually already installed for training
```

### 3. Import in your training script
```python
from runtime_flops_profiler import RuntimeFLOPsMonitor
```

## 🔧 Integration Methods

### Method 1: Hooks-based Monitoring (Recommended)
- ✅ Minimal overhead
- ✅ Real-time monitoring
- ✅ Works with any PyTorch model
- ❌ Approximate for custom operations

```python
# Initialize monitor
flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')
flops_monitor.start_monitoring()

# In training loop
for batch in train_loader:
    result = flops_monitor.measure_step(inputs, targets, optimizer)
    print(f"Step FLOPs: {flops_monitor.format_flops(result['step_flops'])}")
```

### Method 2: PyTorch Profiler (Most Accurate)
- ✅ Official PyTorch tool
- ✅ Very accurate
- ✅ Detailed operation breakdown
- ❌ Higher overhead
- ❌ Slower training

```python
# Initialize monitor
flops_monitor = RuntimeFLOPsMonitor(model, method='profiler')
flops_monitor.start_monitoring()

# Use same API as hooks method
result = flops_monitor.measure_step(inputs, targets, optimizer)
```

## 📊 What You Get

### Real-time Metrics
- **Step FLOPs**: FLOPs for current training step
- **Total FLOPs**: Cumulative FLOPs since training start
- **Average FLOPs/step**: Running average
- **FLOPs/second**: Compute throughput

### Breakdown Analysis
- **Linear operations**: Matrix multiplications, fully connected layers
- **Convolutions**: Conv1D, Conv2D, Conv3D operations  
- **Attention**: Multi-head attention computations
- **Other**: Activations, normalizations, etc.

### Performance Statistics
- **Min/Max/Std**: FLOPs variation across steps
- **Throughput**: Actual compute performance
- **Efficiency**: Hardware utilization insights

## 🎯 Integration Examples

### Simple Integration (Minimal Changes)
```python
# Add these 3 lines to your existing training loop:
flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')
flops_monitor.start_monitoring()

# Inside your training loop, replace:
# loss.backward(); optimizer.step()
# with:
result = flops_monitor.measure_step(inputs, targets, optimizer)
```

### Advanced Integration (Full Monitoring)
```python
class FLOPsTrainer:
    def __init__(self, model, optimizer, train_loader):
        self.flops_monitor = RuntimeFLOPsMonitor(model, method='hooks')
        # ... other setup
    
    def train_epoch(self):
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            result = self.flops_monitor.measure_step(inputs, targets, optimizer)
            
            # Log every 100 steps
            if batch_idx % 100 == 0:
                self.log_flops_stats(result)
    
    def log_flops_stats(self, result):
        print(f"Step FLOPs: {self.flops_monitor.format_flops(result['step_flops'])}")
        print(f"Total FLOPs: {self.flops_monitor.format_flops(result['total_flops'])}")
```

## 🔍 Monitoring Best Practices

### 1. Choose the Right Method
- Use **hooks** for production training (low overhead)
- Use **profiler** for detailed analysis (higher overhead)

### 2. Logging Strategy
- Log FLOPs every N steps (not every step)
- Save FLOPs info in checkpoints
- Track FLOPs trends over training

### 3. Performance Considerations
- Hooks add ~1-2% overhead
- Profiler adds ~5-10% overhead
- Monitor memory usage if using profiler

### 4. Validation
- Compare with theoretical estimates
- Check consistency across similar steps
- Validate against known model FLOPs

## 🚨 Common Issues & Solutions

### Issue 1: Custom Operations Not Counted
**Problem**: Your model uses custom CUDA kernels or operations
**Solution**: 
```python
# Add custom hook for your operation
def custom_op_flop_count(module, input, output):
    # Calculate FLOPs for your custom operation
    flops = calculate_custom_flops(input, output)
    flops_monitor.flop_counter.current_step_flops += flops
    return flops

# Register the hook
model.your_custom_layer.register_forward_hook(custom_op_flop_count)
```

### Issue 2: Memory Issues with Profiler
**Problem**: PyTorch Profiler uses too much memory
**Solution**: 
- Switch to hooks method
- Profile only subset of steps
- Reduce profiler recording options

### Issue 3: Inconsistent FLOPs Counts
**Problem**: FLOPs vary significantly between steps
**Solution**:
- Check for dynamic model behavior
- Verify input sizes are consistent
- Look for conditional computations

## 📈 Analysis and Reporting

### Training Report Template
```python
def generate_flops_report(flops_monitor):
    stats = flops_monitor.get_statistics()
    
    report = f"""
    TRAINING FLOPs REPORT
    ====================
    Total Training Steps: {stats['total_steps']:,}
    Total FLOPs: {flops_monitor.format_flops(stats['total_flops'])}
    Average FLOPs/Step: {flops_monitor.format_flops(stats['avg_flops_per_step'])}
    Compute Throughput: {flops_monitor.format_flops(stats['flops_per_second'])}/s
    Training Duration: {stats['elapsed_time_seconds']:.1f} seconds
    
    Hardware Efficiency: {(stats['flops_per_second'] / theoretical_peak_flops) * 100:.1f}%
    """
    
    return report
```

## 🎯 Next Steps

1. **Test Integration**: Start with a small model/dataset
2. **Validate Accuracy**: Compare with theoretical calculations  
3. **Optimize Performance**: Minimize monitoring overhead
4. **Scale Up**: Apply to full training runs
5. **Analyze Results**: Use data for optimization decisions

## 📚 Files Created

- `runtime_flops_profiler.py`: Main profiling tool
- `training_with_flops_template.py`: General integration template
- `internlm2_flops_integration.py`: InternLM2 specific examples
- `flops_integration_guide.py`: This comprehensive guide

## 🔗 Related Tools

- `internlm2_compute_calculator.py`: Theoretical FLOPs estimation
- `test_flop_counter.py`: Testing and validation
- `accessory/util/flop_counter.py`: Existing FLOPs utilities

Ready to measure real training compute? Start with the hooks method! 🚀
'''
    
    print(guide)


if __name__ == "__main__":
    create_integration_examples()
    print()
    print_integration_guide()