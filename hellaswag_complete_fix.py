"""
完整的分布式HellaSwag评估代码修正
包含数据分割和结果聚合
"""
import torch
import torch.distributed as dist
from tqdm import tqdm
from typing import List, Dict, Any


def distribute_data_across_ranks(data: List, rank: int, world_size: int) -> List:
    """
    将数据分配给不同的rank
    确保每个rank处理不同的数据子集
    """
    total_samples = len(data)
    samples_per_rank = total_samples // world_size
    remainder = total_samples % world_size
    
    if rank < remainder:
        start_idx = rank * (samples_per_rank + 1)
        end_idx = start_idx + samples_per_rank + 1
    else:
        start_idx = rank * samples_per_rank + remainder
        end_idx = start_idx + samples_per_rank
    
    return data[start_idx:end_idx]


def batch_data(data_list: List, batch_size: int = 1) -> List[List]:
    """批处理数据"""
    batches = []
    for i in range(0, len(data_list), batch_size):
        batches.append(data_list[i:i + batch_size])
    return batches


def evaluate_hellaswag_batch(model, batch, device):
    """
    占位函数 - 替换为您的实际batch评估函数
    """
    # 您的实际实现
    results = []
    for item in batch:
        # 实际的评估逻辑
        result = {'correct': True, 'id': item.get('id', 0)}
        results.append(result)
    return results


def evaluate_hellaswag_distributed_complete(model, data, batch_size, device='cuda'):
    """
    完整的分布式HellaSwag评估
    包含：
    1. 数据在不同rank之间的分割
    2. 本地评估
    3. 结果的全局聚合
    """
    
    # ========== 第1步：检查分布式环境 ==========
    is_distributed = False
    rank = 0
    world_size = 1
    
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            is_distributed = True
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            is_distributed = False
            rank = 0
            world_size = 1
    except Exception:
        is_distributed = False
        rank = 0
        world_size = 1
    
    # ========== 第2步：分割数据（关键！） ==========
    if is_distributed:
        # 每个rank获取不同的数据子集
        rank_data = distribute_data_across_ranks(data, rank, world_size)
        total_samples_global = len(data)
        samples_this_rank = len(rank_data)
        
        print(f"[Rank {rank}] Processing {samples_this_rank} samples out of {total_samples_global} total")
        
        # 只在rank 0显示总体信息
        if rank == 0:
            print(f"[Rank 0] Total samples: {total_samples_global}")
            print(f"[Rank 0] Distributed across {world_size} GPUs")
            print(f"[Rank 0] Each GPU processes ~{total_samples_global // world_size} samples")
    else:
        # 单GPU：处理所有数据
        rank_data = data
        total_samples_global = len(data)
        samples_this_rank = len(data)
        print(f"Processing {total_samples_global} samples (single GPU)")
    
    # ========== 第3步：批处理rank特定的数据 ==========
    batched_data = batch_data(rank_data, batch_size)
    
    # ========== 第4步：本地评估 ==========
    all_results = []
    show_progress = (rank == 0) if is_distributed else True
    
    if show_progress:
        for batch in tqdm(batched_data, desc="HellaSwag Eval", leave=False):
            batch_results = evaluate_hellaswag_batch(model, batch, device)
            all_results.extend(batch_results)
    else:
        for batch in batched_data:
            batch_results = evaluate_hellaswag_batch(model, batch, device)
            all_results.extend(batch_results)
    
    # ========== 第5步：计算本地指标 ==========
    correct_predictions = [r for r in all_results if r.get('correct') is True]
    total_predictions = [r for r in all_results if r.get('correct') is not None]
    
    local_correct = len(correct_predictions)
    local_total = len(total_predictions)
    
    print(f"[Rank {rank}] Local results: {local_correct}/{local_total} correct")
    
    # ========== 第6步：聚合分布式结果 ==========
    if is_distributed:
        # 确保所有rank都完成了本地评估
        dist.barrier()
        
        # 将本地统计转换为tensor
        local_correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=device)
        local_total_tensor = torch.tensor([local_total], dtype=torch.long, device=device)
        
        # 聚合所有rank的结果
        dist.all_reduce(local_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total_tensor, op=dist.ReduceOp.SUM)
        
        # 获取全局统计
        global_correct = local_correct_tensor.item()
        global_total = local_total_tensor.item()
        
        # 再次同步
        dist.barrier()
        
        # 计算全局指标
        if global_total == 0:
            metrics = {
                'accuracy': 0.0,
                'total_samples': 0,
                'correct_samples': 0,
                'local_correct': local_correct,
                'local_total': local_total,
                'rank': rank,
                'world_size': world_size
            }
        else:
            accuracy = global_correct / global_total
            metrics = {
                'accuracy': accuracy,
                'total_samples': global_total,
                'correct_samples': global_correct,
                'local_correct': local_correct,
                'local_total': local_total,
                'rank': rank,
                'world_size': world_size
            }
        
        # 只在rank 0打印最终结果
        if rank == 0:
            print("\n" + "="*50)
            print(f"FINAL RESULTS (Aggregated from {world_size} GPUs)")
            print(f"Global Accuracy: {metrics['accuracy']:.4f}")
            print(f"Correct/Total: {global_correct}/{global_total}")
            print("="*50)
    
    else:
        # 单GPU：本地结果就是最终结果
        if local_total == 0:
            metrics = {
                'accuracy': 0.0,
                'total_samples': 0,
                'correct_samples': 0
            }
        else:
            accuracy = local_correct / local_total
            metrics = {
                'accuracy': accuracy,
                'total_samples': local_total,
                'correct_samples': local_correct
            }
        
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Correct/Total: {local_correct}/{local_total}")
        print("="*50)
    
    return metrics


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 模拟数据
    sample_data = [{'id': i, 'text': f'sample_{i}'} for i in range(1000)]
    
    # 模拟模型
    class DummyModel:
        def eval(self):
            pass
    
    model = DummyModel()
    
    # 运行评估
    metrics = evaluate_hellaswag_distributed_complete(
        model=model,
        data=sample_data,
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nFinal metrics: {metrics}")