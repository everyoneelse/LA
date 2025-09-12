"""
Improved HellaSwag Dataset with proper padding strategies for FSDP
"""
import os
import json
import jsonlines
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class HellaSwagDatasetImproved(Dataset):
    """
    HellaSwag dataset with improved padding for FSDP compatibility
    
    Key features:
    1. Dynamic padding to batch max length (more efficient)
    2. Proper attention mask handling
    3. FSDP-compatible data shapes
    """
    
    def __init__(self, 
                 data_file: str,
                 tokenizer,
                 max_length: int = 512,
                 max_samples: Optional[int] = None,
                 padding_side: str = 'right'):
        """
        Args:
            data_file: Path to hellaswag data file
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to load
            padding_side: 'left' or 'right' padding
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_side = padding_side
        self.data = self._load_data(data_file, max_samples)
        
        # Cache tokenized data for efficiency
        self.use_cache = True
        self.cache = {}
        
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
    
    def _tokenize_text(self, text: str) -> Tuple[List[int], int]:
        """
        Tokenize text and return tokens with actual length
        """
        tokens = self.tokenizer.encode(text, bos=True, eos=False)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        actual_length = len(tokens)
        return tokens, actual_length
    
    def __getitem__(self, idx):
        """
        Get a single HellaSwag example
        Returns raw tokenized data without padding (padding done in collate_fn)
        """
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
            
        item = self.data[idx]
        ctx = item['ctx']
        endings = item['endings']
        label = item.get('label', None)
        
        # Store all tokenized endings
        all_tokens = []
        all_lengths = []
        
        for ending in endings:
            full_text = ctx + " " + ending
            tokens, length = self._tokenize_text(full_text)
            all_tokens.append(tokens)
            all_lengths.append(length)
        
        result = {
            'tokens': all_tokens,  # List of token lists (variable length)
            'lengths': all_lengths,  # Actual lengths before padding
            'label': label if label is not None else -1,
            'num_endings': len(endings),
            'idx': idx
        }
        
        if self.use_cache:
            self.cache[idx] = result
            
        return result


def collate_hellaswag_dynamic(batch, tokenizer, max_length=512, padding_side='right'):
    """
    Dynamic padding collate function for HellaSwag
    Pads all sequences in the batch to the same length (batch max or max_length)
    
    This is crucial for FSDP compatibility!
    """
    all_tokens = []
    all_lengths = []
    metadata = []
    
    # Collect all token sequences
    for item in batch:
        for tokens in item['tokens']:
            all_tokens.append(tokens)
            all_lengths.append(len(tokens))
        
        metadata.append({
            'label': item['label'],
            'num_endings': item['num_endings'],
            'idx': item['idx']
        })
    
    # Find the maximum length in this batch
    batch_max_length = min(max(all_lengths), max_length)
    
    # Pad all sequences to the same length
    padded_input_ids = []
    attention_masks = []
    labels = []
    
    pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0
    
    for tokens in all_tokens:
        seq_len = len(tokens)
        
        if padding_side == 'right':
            # Right padding (most common)
            padded = tokens + [pad_id] * (batch_max_length - seq_len)
            mask = [1] * seq_len + [0] * (batch_max_length - seq_len)
        else:
            # Left padding (for some models like GPT)
            padded = [pad_id] * (batch_max_length - seq_len) + tokens
            mask = [0] * (batch_max_length - seq_len) + [1] * seq_len
        
        padded_input_ids.append(padded)
        attention_masks.append(mask)
        
        # Create labels for loss calculation
        label_seq = padded.copy()
        # Set padding tokens to -100 (ignored in loss)
        for i, m in enumerate(mask):
            if m == 0:  # Padding position
                label_seq[i] = -100
        # Also ignore BOS token
        if mask[0] == 1:  # Not padded at start
            label_seq[0] = -100
            
        labels.append(label_seq)
    
    # Convert to tensors
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, attention_mask, labels, metadata


class HellaSwagDatasetWithGrouping(Dataset):
    """
    Advanced HellaSwag dataset that groups similar-length sequences
    for more efficient padding and processing
    """
    
    def __init__(self, 
                 data_file: str,
                 tokenizer,
                 max_length: int = 512,
                 max_samples: Optional[int] = None,
                 group_by_length: bool = True):
        """
        Args:
            data_file: Path to hellaswag data file
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to load
            group_by_length: Whether to group similar-length sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_file, max_samples)
        
        # Pre-tokenize and sort by length for efficient batching
        if group_by_length:
            self._prepare_length_grouped_data()
        else:
            self.indices = list(range(len(self.data)))
    
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
    
    def _prepare_length_grouped_data(self):
        """
        Pre-compute lengths and create sorted indices
        This helps batch similar-length sequences together
        """
        lengths = []
        for item in self.data:
            # Use the maximum length among all endings as the item's length
            max_ending_length = 0
            ctx = item['ctx']
            for ending in item['endings']:
                full_text = ctx + " " + ending
                tokens = self.tokenizer.encode(full_text, bos=True, eos=False)
                max_ending_length = max(max_ending_length, len(tokens))
            lengths.append(min(max_ending_length, self.max_length))
        
        # Sort indices by length
        self.indices = sorted(range(len(self.data)), key=lambda i: lengths[i])
        self.lengths = [lengths[i] for i in self.indices]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item using potentially reordered index"""
        actual_idx = self.indices[idx]
        item = self.data[actual_idx]
        
        ctx = item['ctx']
        endings = item['endings']
        label = item.get('label', None)
        
        # Tokenize all endings
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        # Find max length for this item's endings
        item_max_length = 0
        tokenized_endings = []
        
        for ending in endings:
            full_text = ctx + " " + ending
            tokens = self.tokenizer.encode(full_text, bos=True, eos=False)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            tokenized_endings.append(tokens)
            item_max_length = max(item_max_length, len(tokens))
        
        # Pad all endings to the same length (item_max_length)
        pad_id = self.tokenizer.pad_id if hasattr(self.tokenizer, 'pad_id') else 0
        
        for tokens in tokenized_endings:
            seq_len = len(tokens)
            
            # Right padding
            padded = tokens + [pad_id] * (item_max_length - seq_len)
            mask = [1] * seq_len + [0] * (item_max_length - seq_len)
            
            # Create labels
            label_seq = padded.copy()
            for i in range(seq_len, item_max_length):
                label_seq[i] = -100  # Ignore padding in loss
            label_seq[0] = -100  # Ignore BOS token
            
            all_input_ids.append(padded)
            all_attention_masks.append(mask)
            all_labels.append(label_seq)
        
        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_masks, dtype=torch.long),
            'labels': torch.tensor(all_labels, dtype=torch.long),
            'label': label if label is not None else -1,
            'num_endings': len(endings)
        }


def create_hellaswag_dataloader(
    data_file: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    use_dynamic_padding: bool = True,
    group_by_length: bool = False
):
    """
    Create a DataLoader for HellaSwag evaluation with proper padding
    
    Args:
        data_file: Path to data file
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        max_samples: Max samples to load
        distributed: Whether to use distributed sampling
        world_size: Number of processes
        rank: Current process rank
        use_dynamic_padding: Use dynamic padding (more efficient)
        group_by_length: Group similar-length sequences (most efficient)
    
    Returns:
        DataLoader configured for FSDP-compatible evaluation
    """
    
    if use_dynamic_padding:
        # Use dynamic padding (pad to batch max)
        dataset = HellaSwagDatasetImproved(
            data_file=data_file,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples
        )
        
        collate_fn = lambda batch: collate_hellaswag_dynamic(
            batch, tokenizer, max_length
        )
    else:
        # Use pre-padded dataset (potentially with length grouping)
        dataset = HellaSwagDatasetWithGrouping(
            data_file=data_file,
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples,
            group_by_length=group_by_length
        )
        
        collate_fn = None  # Dataset already returns padded tensors
    
    # Create sampler for distributed training
    if distributed:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        sampler = None
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False if sampler else False,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues with tokenizer
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader