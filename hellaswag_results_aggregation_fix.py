"""
修正后的结果收集和聚合代码
"""
import torch
import torch.distributed as dist
from tqdm import tqdm

def evaluate_hellaswag_with_proper_aggregation(model, batched_data, device='cuda'):
    """
    正确的分布式结果收集和聚合
    """
    # 检查是否在分布式环境
    is_distributed = False
    rank = 0
    world_size = 1
    
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            is_distributed = True
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            show_progress = (rank == 0)
        else:
            show_progress = True
    except Exception:
        show_progress = True
    
    # 收集本地结果
    all_results = []
    
    if show_progress:
        for batch in tqdm(batched_data, desc="HellaSwag Eval", leave=False):
            batch_results = evaluate_hellaswag_batch(model, batch, device)
            all_results.extend(batch_results)
    else:
        for batch in batched_data:
            batch_results = evaluate_hellaswag_batch(model, batch, device)
            all_results.extend(batch_results)
    
    # 计算本地（每个rank）的指标
    correct_predictions = [r for r in all_results if r['correct'] is True]
    total_predictions = [r for r in all_results if r['correct'] is not None]
    
    local_correct = len(correct_predictions)
    local_total = len(total_predictions)
    
    # 如果在分布式环境，需要聚合所有rank的结果
    if is_distributed:
        # 将本地统计转换为tensor以便进行all_reduce
        local_correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=device)
        local_total_tensor = torch.tensor([local_total], dtype=torch.long, device=device)
        
        # 聚合所有rank的结果
        dist.all_reduce(local_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_tensor, op=dist.ReduceOp.SUM)
        
        # 获取全局统计
        global_correct = local_correct_tensor.item()
        global_total = local_total_tensor.item()
        
        # 计算全局准确率
        if global_total == 0:
            metrics = {
                'accuracy': 0.0, 
                'total_samples': 0, 
                'correct_samples': 0,
                'local_correct': local_correct,  # 保留本地统计用于调试
                'local_total': local_total,
                'rank': rank
            }
        else:
            accuracy = global_correct / global_total
            metrics = {
                'accuracy': accuracy,
                'total_samples': global_total,
                'correct_samples': global_correct,
                'local_correct': local_correct,  # 保留本地统计用于调试
                'local_total': local_total,
                'rank': rank
            }
        
        # 只在rank 0打印全局结果
        if rank == 0:
            print(f"\n=== Global HellaSwag Results ===")
            print(f"Total Accuracy: {metrics['accuracy']:.4f}")
            print(f"Correct/Total: {global_correct}/{global_total}")
            print(f"Distributed across {world_size} GPUs")
        
        # 可选：每个rank打印自己的本地贡献
        if rank == 0:  # 或者设置一个debug flag来控制
            # 收集所有rank的本地统计（用于调试）
            all_local_correct = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            all_local_total = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            
            # 这里可以使用gather来收集每个rank的详细信息
            # dist.all_gather(all_local_correct, local_correct_tensor)
            # dist.all_gather(all_local_total, local_total_tensor)
            # 然后打印每个rank的贡献
    
    else:
        # 单机单卡：本地结果就是全局结果
        if local_total == 0:
            metrics = {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
        else:
            accuracy = local_correct / local_total
            metrics = {
                'accuracy': accuracy,
                'total_samples': local_total,
                'correct_samples': local_correct
            }
        
        print(f"\n=== HellaSwag Results ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Correct/Total: {local_correct}/{local_total}")
    
    return metrics, all_results


def evaluate_hellaswag_with_barrier(model, batched_data, device='cuda'):
    """
    带同步屏障的版本，确保所有rank完成计算后再聚合
    """
    # 检查分布式设置
    is_distributed = False
    rank = 0
    
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            is_distributed = True
            rank = torch.distributed.get_rank()
            show_progress = (rank == 0)
        else:
            show_progress = True
    except Exception:
        show_progress = True
    
    # 收集结果
    all_results = []
    
    if show_progress:
        for batch in tqdm(batched_data, desc="HellaSwag Eval", leave=False):
            batch_results = evaluate_hellaswag_batch(model, batch, device)
            all_results.extend(batch_results)
    else:
        for batch in batched_data:
            batch_results = evaluate_hellaswag_batch(model, batch, device)
            all_results.extend(batch_results)
    
    # 添加同步屏障，确保所有rank都完成了评估
    if is_distributed:
        dist.barrier()
        if rank == 0:
            print("All ranks completed evaluation, aggregating results...")
    
    # 计算本地指标
    correct_predictions = [r for r in all_results if r['correct'] is True]
    total_predictions = [r for r in all_results if r['correct'] is not None]
    
    local_correct = len(correct_predictions)
    local_total = len(total_predictions)
    
    # 打印每个rank的本地统计（调试用）
    print(f"[Rank {rank}] Local results: {local_correct}/{local_total}")
    
    if is_distributed:
        # 聚合结果
        correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=device)
        total_tensor = torch.tensor([local_total], dtype=torch.long, device=device)
        
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        global_correct = correct_tensor.item()
        global_total = total_tensor.item()
        
        # 再次同步，确保所有rank都得到了聚合结果
        dist.barrier()
        
        if global_total == 0:
            metrics = {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
        else:
            accuracy = global_correct / global_total
            metrics = {
                'accuracy': accuracy,
                'total_samples': global_total,
                'correct_samples': global_correct
            }
        
        # 所有rank都可以访问全局metrics，但只rank 0打印
        if rank == 0:
            print(f"\n=== Final Results (Aggregated from {torch.distributed.get_world_size()} GPUs) ===")
            print(f"Global Accuracy: {accuracy:.4f} ({global_correct}/{global_total})")
    else:
        # 单机单卡
        if local_total == 0:
            metrics = {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
        else:
            accuracy = local_correct / local_total
            metrics = {
                'accuracy': accuracy,
                'total_samples': local_total,
                'correct_samples': local_correct
            }
        print(f"\nAccuracy: {accuracy:.4f} ({local_correct}/{local_total})")
    
    return metrics


# 用于验证的辅助函数
def verify_distributed_results(metrics, rank, world_size):
    """
    验证分布式结果的正确性
    """
    if 'local_correct' in metrics and 'local_total' in metrics:
        print(f"[Rank {rank}] Verification:")
        print(f"  Local contribution: {metrics['local_correct']}/{metrics['local_total']}")
        print(f"  Global result: {metrics['correct_samples']}/{metrics['total_samples']}")
        print(f"  Global accuracy: {metrics['accuracy']:.4f}")
        
        # 可以添加断言来验证
        # 例如：全局总数应该大于等于本地总数（在正确分割数据的情况下）
        assert metrics['total_samples'] >= metrics['local_total'], \
            f"Global total ({metrics['total_samples']}) should be >= local total ({metrics['local_total']})"


# 占位函数
def evaluate_hellaswag_batch(model, batch, device):
    """占位函数 - 替换为您的实际实现"""
    # 您的实际评估逻辑
    return [{'correct': True}]  # 示例返回值