
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
        
        print(f"\n📈 Epoch {epoch} - Starting FLOPs monitoring")
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
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  Epoch FLOPs: {self.flops_monitor.format_flops(self.epoch_flops)}")
        print(f"  Total FLOPs: {self.flops_monitor.format_flops(self.total_training_flops)}")
        
        return self.epoch_flops
    
    def print_flops_breakdown(self, flops_result):
        """Print detailed FLOPs breakdown."""
        print("\n🔍 FLOPs Breakdown:")
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
    print("\n" + "="*80)
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
