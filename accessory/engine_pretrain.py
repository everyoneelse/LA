import math
import sys
from typing import Iterable, Optional
import contextlib

import torch

import accessory.util.misc as misc
import accessory.util.lr_sched as lr_sched
from accessory.util.flop_counter import FLOPCounter, TokenCounter, get_model_config_from_model

from fairscale.nn.model_parallel import initialize as fs_init

import os
import re
import threading
import time as _time
from pathlib import Path
from accessory.model.meta import MetaModel
import torch.distributed as dist
import multiprocessing as mp


def _parse_iter_from_ckpt_dir(name: str):
    match = re.search(r"iter(\d+)", name)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _list_ckpt_dirs(output_dir: str):
    if output_dir is None or not os.path.isdir(output_dir):
        return []
    # Prefer eval snapshots if present to decouple from heavy training checkpoints
    snap_dir = os.path.join(output_dir, "eval_snapshots")
    search_root = snap_dir if os.path.isdir(snap_dir) else output_dir
    dirs = [d for d in os.listdir(search_root) if d.startswith("epoch")]
    dirs_full = [os.path.join(search_root, d) for d in dirs if os.path.isdir(os.path.join(search_root, d))]
    # sort by iteration if available, else by mtime
    def sort_key(path):
        it = _parse_iter_from_ckpt_dir(os.path.basename(path))
        return (0, it) if it is not None else (1, os.path.getmtime(path))
    return sorted(dirs_full, key=sort_key)


def _save_model_snapshot(model: torch.nn.Module, args, epoch: int, iteration: int):
    if args.output_dir is None:
        return
    save_dir_root = os.path.join(args.output_dir, "eval_snapshots")
    if misc.is_main_process():
        os.makedirs(save_dir_root, exist_ok=True)
    # All ranks should sync to ensure directory exists
    if dist.is_initialized():
        dist.barrier()
    save_name = f"epoch{epoch}-iter{iteration}"
    save_dir = os.path.join(save_dir_root, save_name)
    if misc.is_main_process():
        os.makedirs(save_dir, exist_ok=True)
    # Only DP rank 0 saves consolidated model to avoid duplication
    if fs_init.get_data_parallel_rank() != 0:
        return
    mp_rank = fs_init.get_model_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    with torch.no_grad():
        from torch.distributed.fsdp import FSDP, StateDictType, FullStateDictConfig
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            save_dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "tf32": torch.float,
            }[args.precision]
            consolidated_model_state_dict = {
                "model": {key: val.to(save_dtype) for key, val in model.state_dict().items()},
            }
            model_save_path = os.path.join(
                save_dir,
                f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.model.pth",
            )
            torch.save(consolidated_model_state_dict, model_save_path)
            # Best-effort save tokenizer and meta/config files for convenience
            try:
                model.tokenizer.save(save_dir)
            except Exception:
                pass
            try:
                import dataclasses, json
                model_args_save_path = os.path.join(save_dir, 'config.json')
                with open(model_args_save_path, 'w') as f:
                    json.dump(dataclasses.asdict(model.llma.args), f, indent=2)
                model_meta_save_path = os.path.join(save_dir, 'meta.json')
                with open(model_meta_save_path, 'w') as f:
                    json.dump({"llama_type": model.llama_type}, f, indent=2)
            except Exception:
                pass


def _async_validator_worker(args_namespace):
    try:
        args = args_namespace
        # Configure device for this process
        if args.val_device is not None:
            try:
                torch.cuda.set_device(args.val_device)
            except Exception:
                pass
        # Build validation dataset and loader locally
        from accessory.data import falcon, falcon_packed
        DatasetValCls = falcon_packed.FalconVal if args.packed_data else falcon.FalconVal
        dataset_val = DatasetValCls(args.data_meta_path, args.data_root, tokenizer_path=args.tokenizer_path,
                                    max_words=args.max_words)
        val_loader_local = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            shuffle=False,
            drop_last=True,
        )
        validated = set()
        while True:
            ckpt_dirs = _list_ckpt_dirs(args.output_dir)
            if len(ckpt_dirs) == 0:
                _time.sleep(5.0)
                continue
            latest = ckpt_dirs[-1]
            if latest in validated:
                _time.sleep(3.0)
                continue
            # Load a fresh eval model from checkpoint directory (merging TP shards if needed)
            device = args.val_device if args.val_device is not None else "cuda"
            try:
                dtype = {
                    "fp16": torch.float16,
                    "bf16": torch.bfloat16,
                    "tf32": torch.float32,
                }[args.precision]
                eval_model = MetaModel.from_pretrained(
                    pretrained_path=[latest],
                    llama_type=None,
                    llama_config=None,
                    tokenizer_path=None,
                    with_visual=False,
                    max_seq_len=args.max_words,
                    mp_group=None,
                    dtype=dtype,
                    device=device,
                    quant=False,
                )
            except Exception as e:
                print(f"[AsyncValidator/Child] Failed to load model from {latest}: {e}")
                _time.sleep(8.0)
                continue
            try:
                metrics = val_one_epoch_local(eval_model, val_loader_local, 0, args=args, max_batches=args.val_max_batches)
            except Exception as e:
                print(f"[AsyncValidator/Child] Validation failed on {latest}: {e}")
                metrics = None
            if metrics is not None:
                print(f"[AsyncValidator/Child] {os.path.basename(latest)} metrics: {metrics}")
            try:
                del eval_model
                torch.cuda.empty_cache()
            except Exception:
                pass
            validated.add(latest)
            _time.sleep(3.0)
    except KeyboardInterrupt:
        pass


def train_one_epoch(model: torch.nn.Module,
                    data_loader, val_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, start_iter, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    model.zero_grad(set_to_none=True)

    dataset_state = {}

    # Initialize FLOP counter and token counter
    model_config = get_model_config_from_model(model)
    flop_counter = FLOPCounter(model_config)
    token_counter = TokenCounter()
    
    # Track cumulative FLOPs
    total_flops = 0

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # Start async validator on main process if enabled
    async_validator = None
    if getattr(args, 'async_val', False) and misc.is_main_process():
        try:
            ctx = mp.get_context("spawn")
            # Use a slim namespace to avoid pickling large objects
            from types import SimpleNamespace
            args_ns = SimpleNamespace(**{k: getattr(args, k) for k in vars(args)})
            async_validator = ctx.Process(target=_async_validator_worker, args=(args_ns,), daemon=True)
            async_validator.start()
            print("[AsyncValidator] background process started (pid=%s)" % async_validator.pid)
        except Exception as e:
            print(f"[AsyncValidator] failed to start: {e}")
    for data_iter_step, (examples, labels, item_states) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, start_iter), start=start_iter
    ):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step, args)

        # Update token counter
        batch_size, seq_len = examples.shape
        # Create padding mask (non-zero tokens are valid)
        padding_mask = (examples != 0)
        token_counter.update(batch_size, seq_len, padding_mask)
        
        # Calculate FLOPs for this batch
        batch_flops = flop_counter.calculate_total_flops(batch_size, seq_len)
        total_flops += batch_flops

        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
             c_loss, additional_loss_dict = model(examples, labels)
        loss = c_loss
        for (add_loss, weight) in additional_loss_dict.values():
            loss = loss + add_loss * weight
        loss_value = loss.item()
        c_loss_value = c_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        update_grad = (data_iter_step + 1) % accum_iter == 0
        grad_norm = loss_scaler(
            loss, optimizer, model,
            parameters=model.parameters(),
            update_grad=update_grad,
            clip_grad=None if args.clip_grad <= 0 else args.clip_grad,
        )

        if update_grad:
            assert grad_norm is not None
            if torch.any(torch.isinf(grad_norm)):
                print("grad norm is inf")
            else:
                metric_logger.update(grad_norm=grad_norm)

            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)
        metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # process item states for resume
        for i in range(len(item_states['worker_id'])):
            worker_id, _curr_idx, _file_idx = item_states['worker_id'][i], item_states['_curr_idx'][i], item_states['_file_idx'][i]
            worker_id, _curr_idx, _file_idx = worker_id.item(), _curr_idx.item(), _file_idx.item()
            if worker_id not in dataset_state or \
            dataset_state[worker_id]['_file_idx'] < _file_idx or \
            (dataset_state[worker_id]['_file_idx'] == _file_idx and dataset_state[worker_id]['_curr_idx'] < _curr_idx):
                dataset_state[worker_id] = {"_curr_idx": _curr_idx, "_file_idx":  _file_idx}

        # save checkpoint
        if (data_iter_step + 1) % args.save_freq == 0:
            misc.save_checkpoint(
                output_dir=args.output_dir,
                args=args, epoch=epoch, iteration=data_iter_step, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, dataset_state=dataset_state)

        # validation
        if not getattr(args, 'async_val', False):
            if (data_iter_step + 1) % getattr(args, 'val_freq', 10000) == 0:
                val_metrics = val_one_epoch(model, val_loader, epoch, args=args)
                if log_writer is not None:
                    for metric_name, metric_value in val_metrics.items():
                        log_writer.add_scalar("val/" + metric_name, metric_value, data_iter_step)
                model.train(True)
        else:
            # Async mode: create lightweight snapshots for validator consumption
            if (data_iter_step + 1) % getattr(args, 'val_freq', 10000) == 0:
                try:
                    _save_model_snapshot(model, args, epoch, data_iter_step)
                except Exception as e:
                    print(f"[AsyncValidator] snapshot save failed at iter {data_iter_step}: {e}")

        for metric_name, metric in metric_logger.meters.items():
            metric_value = metric.value
            metric_value = misc.all_reduce_mean(metric_value, group=fs_init.get_data_parallel_group())
            if log_writer is not None:
                log_writer.add_scalar(metric_name, metric_value, data_iter_step)
        
        # Print token and FLOP statistics every print_freq iterations
        if (data_iter_step + 1) % print_freq == 0:
            total_tokens = token_counter.get_total_tokens()
            batch_tokens = token_counter.get_batch_tokens()
            
            # Aggregate across all processes
            if fs_init.get_data_parallel_world_size() > 1:
                total_tokens_tensor = torch.tensor(total_tokens, dtype=torch.long, device='cuda')
                batch_tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long, device='cuda')
                total_flops_tensor = torch.tensor(total_flops, dtype=torch.long, device='cuda')
                
                torch.distributed.all_reduce(total_tokens_tensor, group=fs_init.get_data_parallel_group())
                torch.distributed.all_reduce(batch_tokens_tensor, group=fs_init.get_data_parallel_group())
                torch.distributed.all_reduce(total_flops_tensor, group=fs_init.get_data_parallel_group())
                
                total_tokens = total_tokens_tensor.item()
                batch_tokens = batch_tokens_tensor.item()
                total_flops = total_flops_tensor.item()
            
            if misc.is_main_process():
                print(f"[Iter {data_iter_step + 1}] "
                      f"Tokens: {TokenCounter.format_tokens(batch_tokens)} (batch), "
                      f"{TokenCounter.format_tokens(total_tokens)} (total) | "
                      f"FLOPs: {FLOPCounter.format_flops(batch_flops)} (batch), "
                      f"{FLOPCounter.format_flops(total_flops)} (total)")
                
                # Log to tensorboard if available
                if log_writer is not None:
                    log_writer.add_scalar('tokens/batch', batch_tokens, data_iter_step)
                    log_writer.add_scalar('tokens/total', total_tokens, data_iter_step)
                    log_writer.add_scalar('flops/batch', batch_flops, data_iter_step)
                    log_writer.add_scalar('flops/total', total_flops, data_iter_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # The async validator is a daemon process; it will exit when main process exits
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@ torch.no_grad()
def val_one_epoch(model: torch.nn.Module,
                  data_loader: Iterable, epoch: int,
                  args=None):
    print("!!!start validation!!!")
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data_iter_step, (examples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
             c_loss, additional_loss_dict = model(examples, labels)
        c_loss_value = c_loss.item()

        metric_logger.update(closs=c_loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@ torch.no_grad()
def val_one_epoch_local(model: torch.nn.Module,
                        data_loader: Iterable, epoch: int,
                        args=None, max_batches: Optional[int] = None):
    print("[AsyncValidator] start local validation")
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for data_iter_step, (examples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if max_batches is not None and data_iter_step >= max_batches:
            break
        autocast_ctx = {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
            c_loss, additional_loss_dict = model(examples, labels)
        c_loss_value = c_loss.item()
        metric_logger.update(closs=c_loss_value)
    # DO NOT synchronize between processes here
    print("[AsyncValidator] Averaged stats (local):", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}