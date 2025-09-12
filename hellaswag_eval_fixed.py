"""
HellaSwag evaluation utilities for pretraining integration - Fixed version
"""
import os
import json
import jsonlines
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import traceback
import threading
import contextlib

def safe_tensor_debug(tensor, name="tensor", max_elements=10):
    """Safely print tensor info without causing CUDA sync issues in debugger"""
    try:
        if tensor is None:
            print(f"DEBUG {name}: None")
            return
        
        print(f"DEBUG {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        
        # Only print values if tensor is small or we're not in distributed mode
        if tensor.numel() <= max_elements or not (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1):
            if tensor.is_cuda:
                values = tensor.cpu().flatten()[:max_elements].tolist()
            else:
                values = tensor.flatten()[:max_elements].tolist()
            print(f"DEBUG {name} values: {values}")
        else:
            print(f"DEBUG {name}: skipping values (distributed mode or large tensor)")
    except Exception as e:
        print(f"DEBUG {name}: error accessing tensor - {e}")


def load_hellaswag_data(data_dir: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load HellaSwag validation data"""
    data_file = os.path.join(data_dir, 'hellaswag_val.jsonl')
    
    # Check if we're in distributed mode
    is_distributed = False
    is_rank_0 = True
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            is_distributed = True
            is_rank_0 = torch.distributed.get_rank() == 0
    except Exception:
        is_distributed = False
        is_rank_0 = True

    if not os.path.exists(data_file):
        # Try to download from HuggingFace datasets if file doesn't exist
        if not is_distributed or is_rank_0:
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
    
        # In distributed mode, other ranks wait for rank 0 to finish downloading
        if is_distributed and not is_rank_0:
            import time
            timeout = 300  # 5 minutes timeout
            start_time = time.time()
            while not os.path.exists(data_file) and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not os.path.exists(data_file):
                print(f"Warning: Timeout waiting for HellaSwag data file on rank {torch.distributed.get_rank()}")
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
        if is_rank_0:
            print(f"Warning: Failed to load HellaSwag data from {data_file}: {e}")
        return []
    
    return data


def calculate_perplexity_batch(model, 
                             texts: List[str], 
                             max_length: int = None, 
                             device: str = 'cuda', 
                             precision: str = "bf16") -> List[float]:
    """Calculate perplexity of a text using the model - Thread-safe version"""
    try:
        if not texts:
            return []
        
        batch_size = len(texts)
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = model.tokenizer.encode(text, bos=True, eos=False)
            if len(tokens) == 0:
                tokens = [model.tokenizer.bos_id]  # At least have BOS token
            all_tokens.append(tokens)
        
        # Find the maximum length in this batch
        if max_length is None:
            max_len = max(len(tokens) for tokens in all_tokens)
        else:
            max_len = max_length
            
        # Pad all sequences to the same length
        batch_input_ids = []
        batch_labels = []
        
        for tokens in all_tokens:
            # Truncate if too long
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            
            # Pad to max_len
            padded_tokens = tokens + [model.tokenizer.eos_id] * (max_len - len(tokens))
            
            # Create input_ids and labels
            input_ids = torch.tensor(padded_tokens, dtype=torch.long)
            labels = input_ids.clone()
            
            # Set padding tokens to -100 (ignore in loss)
            labels[len(tokens):] = 0
            # Set BOS token to -100 (ignore in loss)
            labels[0] = 0
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
        
        # Stack into batch tensors and move to device BEFORE any model operations
        batch_input_ids = torch.stack(batch_input_ids)
        batch_labels = torch.stack(batch_labels)
        
        # Move to device in a single operation
        if device != 'cpu':
            batch_input_ids = batch_input_ids.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
        
        # Ensure model is in eval mode and gradients are disabled
        model.eval()
        
        # Use multiple context managers for safety
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(precision in ["bf16", "fp16"]), 
                                       dtype=torch.bfloat16 if precision == "bf16" else torch.float16):
                
                # Critical: Synchronize before model forward pass in distributed mode
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    torch.distributed.barrier()
                
                # Forward pass with explicit error handling
                try:
                    model_output = model(batch_input_ids, batch_labels, None, reduction="none")
                except Exception as e:
                    print(f"Error in model forward pass: {e}")
                    print(f"Batch input shape: {batch_input_ids.shape}")
                    print(f"Batch labels shape: {batch_labels.shape}")
                    raise
                
                # Synchronize after model forward pass in distributed mode
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    torch.distributed.barrier()
                
                if isinstance(model_output, tuple):
                    loss, _ = model_output
                else:
                    loss = model_output

                loss = loss.reshape(batch_size, -1)
                loss = loss.mean(dim=-1)
                
                # Calculate per-sample average loss (ignoring -100 tokens)
                per_sample_losses = []
                for i in range(batch_input_ids.size(0)):
                    valid_tokens = (batch_labels[i] != 0)
                    if valid_tokens.sum() > 0:
                        # Move to CPU before converting to Python float
                        loss_value = loss[i].detach().cpu().item()
                        per_sample_losses.append(torch.exp(torch.tensor(loss_value)).item())
                    else:
                        per_sample_losses.append(float('inf'))
                
                return per_sample_losses

    except Exception as e:
        print(f"Warning: Error calculating perplexity: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return [float('inf')] * len(texts)


def calculate_perplexity(model, text: str, max_length: int, device: str = 'cuda') -> float:
    """
    Calculate perplexity of a single text (wrapper for batch function).
    """
    result = calculate_perplexity_batch(model, [text], max_length, device)
    return result[0] if result else float('inf')


def evaluate_hellaswag_batch(model, 
                           batch_data: List[Dict[str, Any]], 
                           device: str = 'cuda', 
                           max_length: int = 1024) -> List[Dict[str, Any]]:
    """Evaluate a batch of HellaSwag examples - Thread-safe version"""
    results = []
    
    # Store original model state
    original_training_mode = model.training
    
    try:
        # Ensure model is in eval mode
        model.eval()
        
        # Collect all texts that need evaluation
        all_texts = []
        text_to_item_ending = []  # (item_idx, ending_idx)

        for item_idx, item in enumerate(batch_data):
            ctx = item['ctx']
            endings = item['endings']

            for ending_idx, ending in enumerate(endings):
                full_text = ctx + " " + ending
                all_texts.append(full_text)
                text_to_item_ending.append((item_idx, ending_idx))
        
        # Batch calculate perplexities for all texts
        if all_texts:
            all_perplexities = calculate_perplexity_batch(model, all_texts, max_length, device)
        else:
            all_perplexities = []

        for item_idx, item in enumerate(batch_data):
            label = item['label'] if 'label' in item else None
            num_endings = len(item['endings'])

            # Calculate perplexity for each ending
            perplexities = []
            for ending_idx in range(num_endings):
                # Find the perplexity for this item's ending
                for text_idx, (t_item_idx, t_ending_idx) in enumerate(text_to_item_ending):
                    if t_item_idx == item_idx and t_ending_idx == ending_idx:
                        perplexities.append(all_perplexities[text_idx])
                        break
            
            # Predict the ending with lowest perplexity
            predicted_idx = np.argmin(perplexities) if perplexities else 0
            
            result = {
                'predicted': predicted_idx,
                'label': label,
                'correct': predicted_idx == label if label is not None else None,
                'perplexities': perplexities
            }
            results.append(result)
    
    finally:
        # Restore original training mode
        if original_training_mode:
            model.train()
    
    return results


def batch_data(data_list: List, batch_size: int = 1) -> List[List]:
    """Batch data into smaller chunks"""
    batches = []
    for i in range(0, len(data_list), batch_size):
        batches.append(data_list[i:i + batch_size])
    return batches


@contextlib.contextmanager
def evaluation_mode(model):
    """Context manager to safely handle model evaluation mode in distributed training"""
    original_training_mode = model.training
    try:
        model.eval()
        # Synchronize model state across all processes
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()
        yield model
    finally:
        if original_training_mode:
            model.train()
        # Synchronize model state restoration
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()


def run_hellaswag_evaluation(model, 
                           data_dir: str, 
                           batch_size: int = 8, 
                           max_samples: Optional[int] = None, 
                           device: str = 'cuda') -> Dict[str, float]:
    """
    Run HellaSwag evaluation on the model - Thread-safe version
    
    Args:
        model: The model to evaluate
        data_dir: Directory containing HellaSwag data
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate (None for all)
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    
    # Check distributed setup first
    is_distributed = False
    is_rank_0 = True
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            is_distributed = True
            is_rank_0 = torch.distributed.get_rank() == 0
    except Exception:
        is_distributed = False
        is_rank_0 = True

    # Load data
    data = load_hellaswag_data(data_dir, max_samples)
    
    if len(data) == 0:
        if is_rank_0:
            print("Warning: No HellaSwag data found, skipping evaluation")
        return {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    if is_rank_0:
        print(f"Evaluating on {len(data)} HellaSwag samples...")
    
    # Use the evaluation context manager
    with evaluation_mode(model):
        try:
            # Batch the data
            batched_data = batch_data(data, batch_size)
            
            all_results = []
            show_progress = is_rank_0

            if show_progress:
                for batch in tqdm(batched_data, desc="HellaSwag Eval", leave=False):
                    batch_results = evaluate_hellaswag_batch(model, batch, device)
                    all_results.extend(batch_results)
            else:
                for batch in batched_data:
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
                
        except Exception as e:
            if is_rank_0:
                print(f"Error during HellaSwag evaluation: {e}")
                print(f"Traceback: {traceback.format_exc()}")
            metrics = {'accuracy': 0.0, 'total_samples': 0, 'correct_samples': 0}
    
    return metrics


def save_hellaswag_results(metrics: Dict[str, float], output_dir: str, iteration: int):
    """Save HellaSwag evaluation results"""
    if output_dir is None:
        return
        
    # Only save results on rank 0 in distributed mode
    is_rank_0 = True
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            is_rank_0 = torch.distributed.get_rank() == 0
    except Exception:
        is_rank_0 = True

    if not is_rank_0:
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
    # Only print on rank 0 in distributed mode
    is_rank_0 = True
    try:
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            is_rank_0 = torch.distributed.get_rank() == 0
    except Exception:
        is_rank_0 = True

    if is_rank_0:
        print(f"[HellaSwag Eval - Iter {iteration}] "
              f"Accuracy: {metrics['accuracy']:.4f} "
              f"({metrics['correct_samples']}/{metrics['total_samples']})")