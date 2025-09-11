"""
HellaSwag evaluation utilities for pretraining integration
"""
import os
import json
import jsonlines
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm


def load_hellaswag_data(data_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load HellaSwag validation data"""
    data_file = os.path.join(data_dir, 'hellaswag_val.jsonl')
    
    if not os.path.exists(data_file):
        # Try to download from HuggingFace datasets if file doesn't exist
        try:
            from datasets import load_dataset
            print(f"Downloading HellaSwag validation data to {data_file}")
            dataset = load_dataset("hellaswag", split="validation")
            
            # Create directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Save to local file
            with jsonlines.open(data_file, mode='w') as writer:
                for item in dataset:
                    writer.write(item)
            print(f"Downloaded HellaSwag validation data to {data_file}")
        except ImportError:
            print("Warning: datasets library not available, cannot download HellaSwag data")
            return []
        except Exception as e:
            print(f"Warning: Failed to download HellaSwag data: {e}")
            return []
    
    # Load data from file
    data = []
    try:
        with jsonlines.open(data_file) as reader:
            for item in reader:
                data.append(item)
                if max_samples is not None and len(data) >= max_samples:
                    break
    except Exception as e:
        print(f"Warning: Failed to load HellaSwag data from {data_file}: {e}")
        return []
    
    return data


def calculate_perplexity(model, text: str, device: str = 'cuda') -> float:
    """
    Calculate perplexity of a text using the model.
    
    Note: The model (MetaModel) expects input_ids and labels in the same format,
    and handles the label shifting internally:
    - output = output[:, :-1, :]  (remove last output token)
    - labels = labels[:, 1:]      (remove first label token, i.e., BOS)
    This ensures proper next-token prediction alignment.
    """
    try:
        # Tokenize the text
        tokens = model.tokenizer.encode(text, bos=True, eos=False)
        if len(tokens) == 0:
            return float('inf')
        
        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        # Create labels for loss calculation
        # The model expects labels in the same format as input_ids
        # It will handle the shifting internally (see model/meta.py line 248-249)
        labels = input_ids.clone()
        
        # Set BOS token to -100 to ignore it in loss calculation
        # This is because we typically don't want to predict the BOS token
        labels[:, 0] = -100
        
        # Debug: Print tensor info before forward pass (only on rank 0 in distributed mode)
        try:
            import torch.distributed as dist
            should_print = not dist.is_initialized() or dist.get_rank() == 0
        except:
            should_print = True
            
        if should_print:
            print(f"DEBUG: About to call model.forward with:")
            print(f"  input_ids: shape={input_ids.shape}, device={input_ids.device}")
            print(f"  labels: shape={labels.shape}, device={labels.device}")
            print(f"  labels sample: {labels.cpu().flatten()[:10].tolist()}")
        
        # Forward pass with timeout protection
        with torch.no_grad():
            try:
                # Add explicit CUDA synchronization before model call in distributed mode
                try:
                    import torch.distributed as dist
                    if dist.is_initialized() and input_ids.is_cuda:
                        torch.cuda.synchronize()
                except:
                    pass
                
                loss, _ = model(input_ids, labels)
                
                # Ensure loss is a scalar and on CPU for consistent behavior
                if hasattr(loss, 'item'):
                    loss_value = loss.item()
                else:
                    loss_value = float(loss)
                    
            except Exception as e:
                print(f"Error in model forward pass: {e}")
                return float('inf')
            perplexity = torch.exp(torch.tensor(loss_value)).item()
            return perplexity
    except Exception as e:
        print(f"Warning: Error calculating perplexity: {e}")
        return float('inf')


def evaluate_hellaswag_batch(model, batch_data: List[Dict[str, Any]], device: str = 'cuda') -> List[Dict[str, Any]]:
    """Evaluate a batch of HellaSwag examples"""
    results = []
    
    for item in batch_data:
        ctx = item['ctx']
        endings = item['endings']
        label = item['label'] if 'label' in item else None
        
        # Calculate perplexity for each ending
        perplexities = []
        for ending in endings:
            # Create full text by combining context and ending
            full_text = ctx + " " + ending
            ppl = calculate_perplexity(model, full_text, device)
            perplexities.append(ppl)
        
        # Predict the ending with lowest perplexity
        predicted_idx = np.argmin(perplexities)
        
        result = {
            'predicted': predicted_idx,
            'label': label,
            'correct': predicted_idx == label if label is not None else None,
            'perplexities': perplexities
        }
        results.append(result)
    
    return results


def batch_data(data_list: List, batch_size: int = 1) -> List[List]:
    """Batch data into smaller chunks"""
    batches = []
    for i in range(0, len(data_list), batch_size):
        batches.append(data_list[i:i + batch_size])
    return batches


def run_hellaswag_evaluation(model, data_dir: str, batch_size: int = 8, 
                           max_samples: Optional[int] = None, device: str = 'cuda') -> Dict[str, float]:
    """
    Run HellaSwag evaluation on the model
    
    Args:
        model: The model to evaluate
        data_dir: Directory containing HellaSwag data
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate (None for all)
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load data
    data = load_hellaswag_data(data_dir, max_samples)
    
    if len(data) == 0:
        print("Warning: No HellaSwag data found, skipping evaluation")
        return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    print(f"Evaluating on {len(data)} HellaSwag samples...")
    
    # Set model to eval mode
    original_training_mode = model.training
    model.eval()
    
    try:
        # Batch the data
        batched_data = batch_data(data, batch_size)
        
        all_results = []
        for batch in tqdm(batched_data, desc="HellaSwag Eval", leave=False):
            batch_results = evaluate_hellaswag_batch(model, batch, device)
            all_results.extend(batch_results)
        
        # Calculate metrics
        correct_predictions = [r for r in all_results if r['correct'] is True]
        total_predictions = [r for r in all_results if r['correct'] is not None]
        
        if len(total_predictions) == 0:
            metrics = {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
        else:
            accuracy = len(correct_predictions) / len(total_predictions)
            metrics = {
                'accuracy': accuracy,
                'total_samples': len(total_predictions),
                'correct_samples': len(correct_predictions)
            }
        
    finally:
        # Restore original training mode
        if original_training_mode:
            model.train()
    
    return metrics


def save_hellaswag_results(metrics: Dict[str, float], output_dir: str, iteration: int):
    """Save HellaSwag evaluation results"""
    if output_dir is None:
        return
        
    results_dir = os.path.join(output_dir, 'hellaswag_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save current results
    result_file = os.path.join(results_dir, f'iter_{iteration}.json')
    with open(result_file, 'w') as f:
        json.dump({
            'iteration': iteration,
            'metrics': metrics
        }, f, indent=2)
    
    # Update overall results log
    log_file = os.path.join(results_dir, 'hellaswag_log.jsonl')
    with jsonlines.open(log_file, mode='a') as writer:
        writer.write({
            'iteration': iteration,
            **metrics
        })


def print_hellaswag_results(metrics: Dict[str, float], iteration: int):
    """Print HellaSwag evaluation results"""
    print(f"[HellaSwag Eval - Iter {iteration}] "
          f"Accuracy: {metrics['accuracy']:.4f} "
          f"({metrics['correct_samples']}/{metrics['total_samples']})")