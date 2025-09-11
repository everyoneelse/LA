import warnings
import os
import torch
import torch.nn as nn
import json
from typing import List, Dict, Optional, Iterable
from pathlib import Path
import inspect
import importlib

from fairscale.nn.model_parallel import initialize as fs_init

from .tokenizer import Tokenizer, probe_tokenizer_path_from_pretrained
from accessory.util import misc, tensor_parallel
from accessory.util.tensor_type import default_tensor_type
import torch.distributed as dist


class MetaModel(nn.Module):
    def __init__(
        self, llama_type: str, llama_config: str|List[str], tokenizer_path: str,
        with_visual: bool = False, max_seq_len: int = 4096
    ) -> None:
        super().__init__()

        self.llama_type = llama_type
        self.with_visual = with_visual

        model_module = importlib.import_module(f"accessory.model.LLM.{llama_type}")
        ModelArgs = model_module.ModelArgs
        Transformer = model_module.Transformer

        llama_args = {}
        if isinstance(llama_config, str):
            llama_config = [llama_config]
        for _ in llama_config:
            with open(_, "r") as f:
                llama_args.update(json.loads(f.read()))
        llama_args['max_seq_len'] = max_seq_len
        llama_args['max_batch_size'] = 32

        tokenizer = Tokenizer(model_path=tokenizer_path)
        llama_args['vocab_size'] = tokenizer.n_words

        llama_args: ModelArgs = ModelArgs(**llama_args)

        if "tokenizer" in inspect.signature(Transformer.__init__).parameters:
            # generally it means the inner llm modify change the tokenizer
            model = Transformer(llama_args, tokenizer, with_visual=with_visual)
            assert hasattr(model, "tokenizer")
            self.tokenizer = model.tokenizer
        else:
            model = Transformer(llama_args, with_visual=with_visual)
            self.tokenizer = tokenizer

        print("Model Args:\n", model.args)

        self.llma = model

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self._set_default_trainability()

        self.is_peft = getattr(model, "is_peft", False)
        print(f"Model is Peft: {self.is_peft}")

        misc.mark_mp_params(self)

        param_count_local, param_count_all = 0, 0
        for name, param in self.named_parameters():
            is_model_parallel = getattr(param, "is_model_parallel", False)
            if param.requires_grad:
                if is_model_parallel:
                    param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    param_count_all += param.numel()
                param_count_local += param.numel()
        print(f"Trainable parameter count : {param_count_local} (local rank), {param_count_all} (all).")

    # ... [keeping all the other methods unchanged until forward] ...

    def forward(self, examples, labels, images=None):
        # Force CUDA synchronization to prevent hanging
        if labels.is_cuda:
            torch.cuda.synchronize(labels.device)
        
        # Create a detached copy to avoid gradient tracking issues
        labels_detached = labels.detach()
        
        with torch.no_grad():
            # Alternative approach: Avoid torch.count_nonzero which can hang
            # Instead, find the last non-zero position directly using boolean indexing
            try:
                # Create a mask for non-zero elements
                non_zero_mask = (labels_detached != 0).any(dim=0)  # Shape: [seq_len]
                
                # Find the last True position
                if non_zero_mask.any():
                    # Get indices where mask is True and take the last one
                    non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=False).squeeze(-1)
                    last_non_zero_pos = non_zero_indices[-1].item() if len(non_zero_indices) > 0 else -1
                else:
                    last_non_zero_pos = -1
                
                # Create a dummy non_zero_ tensor for compatibility with existing code
                seq_len = labels_detached.shape[1]
                non_zero_ = torch.zeros(seq_len, device=labels_detached.device, dtype=torch.long)
                # Mark positions up to last_non_zero_pos as having non-zero elements
                if last_non_zero_pos >= 0:
                    non_zero_[:last_non_zero_pos + 1] = 1
                
            except Exception:
                # Fallback to original method with CPU if alternative fails
                if labels_detached.is_cuda:
                    labels_cpu = labels_detached.cpu()
                    non_zero_ = torch.count_nonzero(labels_cpu, dim=0).to(labels_detached.device)
                else:
                    non_zero_ = torch.count_nonzero(labels_detached, dim=0)
            
            pos = non_zero_.shape[0] - 1
            while pos >= 0:
                if non_zero_[pos] == 0:
                    pos -= 1
                else:
                    break

            if pos == -1:  # nothing to predict in the whole batch
                print(f"[RANK {dist.get_rank()}] nothing to predict in the whole batch!", force=True)
                print(examples.cpu().tolist(), force=True)
                pos = 2
            examples = examples[:, :pos+1]
            labels_detached = labels_detached[:, :pos+1]

        output = self.llma(examples, images)
        if isinstance(output, tuple):
            output, additional_loss = output
        else:
            additional_loss = {}
        output = output[:, :-1, :]
        labels_detached = labels_detached[:, 1:]

        if labels_detached.sum() == 0:
           c_loss = output.mean() * 0
        else:
           c_loss = self.criterion(output.reshape(-1, self.tokenizer.n_words), labels_detached.flatten())
        return c_loss, additional_loss

    # ... [rest of the methods remain unchanged] ...