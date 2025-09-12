"""
HellaSwag Dataset for distributed evaluation
"""
import os
import json
import jsonlines
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import numpy as np


class HellaSwagDataset(Dataset):
    """
    HellaSwag dataset compatible with distributed training
    """
    def __init__(self, 
                 data_file: str,
                 tokenizer,
                 max_length: int = 512,
                 max_samples: Optional[int] = None):
        """
        Args:
            data_file: Path to hellaswag data file
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to load
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_file, max_samples)
        
    def _load_data(self, data_file: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load HellaSwag data from file"""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        data = []
        with jsonlines.open(data_file) as reader:
            for item in reader:
                data.append(item)
                if max_samples is not None and len(data) >= max_samples:
                    break
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single HellaSwag example with all its endings
        Returns tokenized inputs for all endings of this example
        """
        item = self.data[idx]
        ctx = item['ctx']
        endings = item['endings']
        label = item.get('label', None)
        
        # Tokenize all endings for this context
        all_input_ids = []
        all_labels = []
        
        for ending in endings:
            full_text = ctx + " " + ending
            
            # Tokenize
            tokens = self.tokenizer.encode(full_text, bos=True, eos=False)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            # Pad to max_length
            input_ids = tokens + [self.tokenizer.pad_id] * (self.max_length - len(tokens))
            
            # Create labels (for perplexity calculation)
            labels = input_ids.copy()
            # Ignore padding tokens and BOS token in loss
            for i in range(len(tokens), self.max_length):
                labels[i] = -100
            labels[0] = -100  # Ignore BOS
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)  # [num_endings, seq_len]
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # [num_endings, seq_len]
        
        return {
            'input_ids': input_ids_tensor,
            'labels': labels_tensor,
            'label': label if label is not None else -1,
            'num_endings': len(endings)
        }


def collate_hellaswag(batch):
    """
    Custom collate function for HellaSwag dataset
    Flattens all endings into a single batch for efficient processing
    """
    all_input_ids = []
    all_labels = []
    metadata = []
    
    for item in batch:
        # Each item contains multiple endings
        all_input_ids.append(item['input_ids'])  # [num_endings, seq_len]
        all_labels.append(item['labels'])  # [num_endings, seq_len]
        metadata.append({
            'label': item['label'],
            'num_endings': item['num_endings']
        })
    
    # Concatenate all endings from all examples
    input_ids = torch.cat(all_input_ids, dim=0)  # [total_endings, seq_len]
    labels = torch.cat(all_labels, dim=0)  # [total_endings, seq_len]
    
    return input_ids, labels, metadata