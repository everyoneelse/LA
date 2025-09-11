import argparse
import json
import jsonlines
from tqdm import tqdm
import torch
import numpy as np
from typing import List, Dict, Any
import os
import sys

sys.path.append(os.path.join(os.path.abspath(__file__).rsplit('/', 3)[0], 'accessory'))

from model.meta import MetaModel
from util import misc
from fairscale.nn.model_parallel import initialize as fs_init
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize

def get_args_parser():
    parser = argparse.ArgumentParser('HellaSwag evaluation', add_help=False)
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data/hellaswag/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--overwrite', action="store_true", default=False, 
                        help="Overwrite existed results")
    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='directory containing pretrained checkpoints')
    parser.add_argument('--pretrained_type', type=str, default="consolidated", choices=['consolidated', 'meta_ori'],
                        help='pretrained checkpoint save format')
    parser.add_argument('--max_seq_len', default=2048, type=int,
                        help='max input sequence length, which should be adjusted accordingly to the model')
    # Parallel parameters
    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument('--model_parallel_size', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser

def load_model(args):
    """Load model and tokenizer"""
    # define the model
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=False)
    print(f"load pretrained from {args.pretrained_path}")
    load_tensor_parallel_model_list(model, args.pretrained_path)

    if hasattr(args, 'quant') and args.quant:
        print("Quantizing model to 4bit!")
        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            },
            return_unused_kwargs=False,
        )
        quantize(model, quantization_config)
        
    model.bfloat16().cuda()
    return model

def load_hellaswag_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load HellaSwag validation data"""
    data_file = os.path.join(data_dir, 'hellaswag_val.jsonl')
    if not os.path.exists(data_file):
        # If the file doesn't exist, try to download from HuggingFace datasets
        try:
            from datasets import load_dataset
            dataset = load_dataset("hellaswag", split="validation")
            # Save to local file for future use
            os.makedirs(data_dir, exist_ok=True)
            with jsonlines.open(data_file, mode='w') as writer:
                for item in dataset:
                    writer.write(item)
            print(f"Downloaded HellaSwag validation data to {data_file}")
        except ImportError:
            raise ImportError("Please install datasets library: pip install datasets")
        except Exception as e:
            raise Exception(f"Failed to load HellaSwag data: {e}")
    
    # Load data from file
    data = []
    with jsonlines.open(data_file) as reader:
        for item in reader:
            data.append(item)
    return data

def format_hellaswag_prompt(ctx: str, endings: List[str]) -> List[str]:
    """Format HellaSwag prompt for each ending"""
    prompts = []
    for i, ending in enumerate(endings):
        # Create a prompt that asks the model to continue the context
        prompt = f"Context: {ctx}\nContinuation: {ending}"
        prompts.append(prompt)
    return prompts

def batch_data(data_list: List, batch_size: int = 1) -> List[List]:
    """Batch data into smaller chunks"""
    batches = []
    for i in range(0, len(data_list), batch_size):
        batches.append(data_list[i:i + batch_size])
    return batches

def calculate_perplexity(model, tokenizer, text: str) -> float:
    """Calculate perplexity of a text using the model"""
    model.eval()
    with torch.no_grad():
        # Tokenize the text
        tokens = tokenizer.encode(text, bos=True, eos=False)
        if len(tokens) == 0:
            return float('inf')
        
        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
        
        # Get logits from model
        try:
            # For MetaModel, we need to create labels for loss calculation
            labels = input_ids.clone()
            labels[:, 0] = -100  # Ignore the first token in loss calculation
            
            # Forward pass
            loss, _ = model(input_ids, labels)
            perplexity = torch.exp(loss).item()
            return perplexity
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')

def evaluate_hellaswag_batch(model, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            ppl = calculate_perplexity(model, model.tokenizer, full_text)
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

def run_hellaswag_inference(model, data: List[Dict[str, Any]], batch_size: int = 8) -> List[Dict[str, Any]]:
    """Run inference on HellaSwag data"""
    model.eval()  # CRITICAL: Set to eval mode to avoid sequence length optimization deadlock
    
    # Batch the data
    batched_data = batch_data(data, batch_size)
    
    all_results = []
    for batch in tqdm(batched_data, desc="Evaluating HellaSwag"):
        batch_results = evaluate_hellaswag_batch(model, batch)
        all_results.extend(batch_results)
    
    return all_results

def calculate_hellaswag_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate HellaSwag accuracy metrics"""
    correct_predictions = [r for r in results if r['correct'] is True]
    total_predictions = [r for r in results if r['correct'] is not None]
    
    if len(total_predictions) == 0:
        return {'accuracy': 0.0, 'total_samples': 0}
    
    accuracy = len(correct_predictions) / len(total_predictions)
    
    return {
        'accuracy': accuracy,
        'total_samples': len(total_predictions),
        'correct_samples': len(correct_predictions)
    }

def evaluate_hellaswag(model, data_dir: str, batch_size: int = 8, max_samples: int = None) -> Dict[str, float]:
    """Main function to evaluate model on HellaSwag"""
    # Load data
    data = load_hellaswag_data(data_dir)
    
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
        print(f"Evaluating on {max_samples} samples")
    
    print(f"Evaluating on {len(data)} HellaSwag samples...")
    
    # Run inference
    results = run_hellaswag_inference(model, data, batch_size)
    
    # Calculate metrics
    metrics = calculate_hellaswag_accuracy(results)
    
    return metrics

def main(args):
    # Setup paths
    path_split = args.pretrained_path.split('/')
    if path_split[-1] == '':
        path_split.pop(-1)
    model_name = path_split[-1] 
    
    infer_path = os.path.join('results', model_name, 'hellaswag/infer')
    os.makedirs(infer_path, exist_ok=True)
    eval_path = os.path.join('results', model_name, 'hellaswag/eval')
    os.makedirs(eval_path, exist_ok=True)

    # Load model
    model = load_model(args)
    
    # Load and evaluate on HellaSwag
    data = load_hellaswag_data(args.data_dir)
    
    infer_file = os.path.join(infer_path, 'hellaswag_infer.jsonl')
    if not args.overwrite and os.path.exists(infer_file):
        print(f"{infer_file} existed, skip inference!")
    else:
        print("Running HellaSwag inference...")
        results = run_hellaswag_inference(model, data, args.batch_size)
        
        # Save inference results
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                with jsonlines.open(infer_file, mode='w') as writer:
                    for result in results:
                        writer.write(result)
        else:
            with jsonlines.open(infer_file, mode='w') as writer:
                for result in results:
                    writer.write(result)
    
    # Calculate and save evaluation metrics
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # Load results from file
        results = []
        with jsonlines.open(infer_file) as reader:
            for item in reader:
                results.append(item)
        
        metrics = calculate_hellaswag_accuracy(results)
        
        print(f"HellaSwag Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Correct: {metrics['correct_samples']}/{metrics['total_samples']}")
        
        # Save results
        with open(os.path.join(eval_path, 'run_results.json'), 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)