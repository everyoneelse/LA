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


def calculate_perplexity_batch(model, texts: List[str], device: str = 'cuda', max_length: int = 512) -> List[float]:
    """
    Calculate perplexity for a batch of texts with proper padding to ensure same length.
    This prevents distributed training deadlocks caused by different sequence lengths.
    Handles mixed precision (BFloat16/Float16) properly.
    """
    try:
        if not texts:
            return []
        
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
            padded_tokens = tokens + [model.tokenizer.pad_id] * (max_len - len(tokens))
            
            # Create input_ids and labels
            input_ids = torch.tensor(padded_tokens, dtype=torch.long)
            labels = input_ids.clone()
            
            # Set padding tokens to -100 (ignore in loss)
            labels[len(tokens):] = -100
            # Set BOS token to -100 (ignore in loss)
            labels[0] = -100
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
        
        # Stack into batch tensors
        batch_input_ids = torch.stack(batch_input_ids).to(device)
        batch_labels = torch.stack(batch_labels).to(device)
        
        # Forward pass with proper dtype handling
        with torch.no_grad():
            # First, detect the model's dtype from its parameters
            model_dtype = next(model.parameters()).dtype
            
            # Use the model's forward method which handles dtype properly
            loss, _ = model(batch_input_ids, batch_labels)
            
            # For per-sample loss calculation, we need to be more careful about dtypes
            # Get model output with proper autocast context
            if model_dtype == torch.bfloat16:
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
            elif model_dtype == torch.float16:
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            else:
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float32)
            
            with autocast_ctx:
                output = model.llma(batch_input_ids, None)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Shift for next token prediction
                output = output[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
                shifted_labels = batch_labels[:, 1:]  # [batch_size, seq_len-1]
                
                # Calculate loss for each sample with proper dtype
                criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                per_token_losses = criterion(output.reshape(-1, output.size(-1)), shifted_labels.reshape(-1))
                per_token_losses = per_token_losses.view(batch_input_ids.size(0), -1)  # [batch_size, seq_len-1]
            
            # Calculate per-sample average loss (ignoring -100 tokens)
            per_sample_losses = []
            for i in range(batch_input_ids.size(0)):
                valid_tokens = (shifted_labels[i] != -100)
                if valid_tokens.sum() > 0:
                    sample_loss = per_token_losses[i][valid_tokens].mean()
                    per_sample_losses.append(torch.exp(sample_loss).item())
                else:
                    per_sample_losses.append(float('inf'))
            
            return per_sample_losses
            
    except Exception as e:
        print(f"Warning: Error calculating batch perplexity: {e}")
        # Fallback: try individual calculation to isolate the problem
        try:
            print("Attempting fallback to individual perplexity calculation...")
            fallback_results = []
            for text in texts:
                try:
                    # Simple individual calculation
                    tokens = model.tokenizer.encode(text, bos=True, eos=False)
                    if len(tokens) == 0:
                        fallback_results.append(float('inf'))
                        continue
                    
                    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                    labels = input_ids.clone()
                    labels[:, 0] = -100
                    
                    with torch.no_grad():
                        loss, _ = model(input_ids, labels)
                        ppl = torch.exp(loss).item()
                        fallback_results.append(ppl)
                except Exception as inner_e:
                    print(f"Individual calculation failed for text: {inner_e}")
                    fallback_results.append(float('inf'))
            
            return fallback_results
        except Exception as fallback_e:
            print(f"Fallback also failed: {fallback_e}")
            return [float('inf')] * len(texts)


def calculate_perplexity(model, text: str, device: str = 'cuda') -> float:
    """
    Calculate perplexity of a single text (wrapper for batch function).
    """
    result = calculate_perplexity_batch(model, [text], device)
    return result[0] if result else float('inf')


def evaluate_hellaswag_batch(model, batch_data: List[Dict[str, Any]], device: str = 'cuda', max_length: int = 512) -> List[Dict[str, Any]]:
    """Evaluate a batch of HellaSwag examples with efficient batch processing"""
    results = []
    
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
        all_perplexities = calculate_perplexity_batch(model, all_texts, device, max_length)
    else:
        all_perplexities = []
    
    # Reconstruct results
    for item_idx, item in enumerate(batch_data):
        label = item['label'] if 'label' in item else None
        num_endings = len(item['endings'])
        
        # Extract perplexities for this item
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
    
    return results


def batch_data(data_list: List, batch_size: int = 1) -> List[List]:
    """Batch data into smaller chunks"""
    batches = []
    for i in range(0, len(data_list), batch_size):
        batches.append(data_list[i:i + batch_size])
    return batches


def run_hellaswag_evaluation(model, data_dir: str, batch_size: int = 4, 
                           max_samples: Optional[int] = None, device: str = 'cuda', max_length: int = 512) -> Dict[str, float]:
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