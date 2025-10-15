import math
import sys
import contextlib

import torch

import accessory.util.misc as misc
import accessory.util.lr_sched as lr_sched

from fairscale.nn.model_parallel import initialize as fs_init
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import json
import time


def simple_prompt_logging(prompts, args, step, epoch):
    """
    Simple fallback: just log the prompts and training progress
    This is useful when FSDP makes generation impossible during training
    """
    if not prompts or len(prompts) == 0:
        return
    
    if step % args.test_prompt_interval != 0:
        return
    
    # Only run on main process
    if not misc.is_main_process():
        return
    
    print(f"\n{'='*60}")
    print(f"PROMPT CHECKPOINT - Epoch: {epoch}, Step: {step}")
    print(f"{'='*60}")
    
    # Log training progress and prompts
    print(f"Training Progress: Epoch {epoch}, Step {step}")
    print(f"Scheduled prompts for testing:")
    
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")
    
    print(f"\nNote: Actual generation will be performed after training completes.")
    print(f"You can test these prompts manually using the inference demo.")
    
    # Optionally save to file for later testing
    if hasattr(args, 'output_dir') and args.output_dir:
        try:
            prompt_log_file = f"{args.output_dir}/prompt_test_log.jsonl"
            log_entry = {
                "epoch": epoch,
                "step": step,
                "timestamp": time.time(),
                "prompts": prompts,
                "test_params": {
                    "max_gen_len": args.test_prompt_max_gen_len,
                    "temperature": args.test_prompt_temperature,
                    "top_p": args.test_prompt_top_p
                }
            }
            with open(prompt_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"Prompts logged to: {prompt_log_file}")
        except Exception as e:
            print(f"Could not save prompt log: {e}")
    
    print(f"{'='*60}\n")


def test_prompts_during_training(model, prompts, args, step, epoch):
    """
    Test prompts during training - with multiple fallback strategies for FSDP models
    """
    if not prompts or len(prompts) == 0:
        return
    
    if step % args.test_prompt_interval != 0:
        return
    
    # Only run on main process to avoid duplicate outputs
    if not misc.is_main_process():
        return
    
    print(f"\n{'='*60}")
    print(f"PROMPT TEST - Epoch: {epoch}, Step: {step}")
    print(f"{'='*60}")
    
    # Save original training state
    was_training = model.training
    
    try:
        model.eval()
        
        with torch.no_grad():
            # Use the same autocast context as training
            autocast_ctx = {
                "bf16": torch.amp.autocast('cuda', dtype=torch.bfloat16),
                "fp16": torch.amp.autocast('cuda', dtype=torch.float16),
                "tf32": contextlib.nullcontext(),
            }[args.precision]
            
            success = False
            
            # Strategy 1: Try direct generation (works for non-FSDP models)
            if not success:
                try:
                    with autocast_ctx:
                        results = model.generate(
                            prompts, 
                            None,  # image
                            max_gen_len=args.test_prompt_max_gen_len, 
                            temperature=args.test_prompt_temperature, 
                            top_p=args.test_prompt_top_p
                        )
                    success = True
                    print("Using direct generation")
                except Exception as e:
                    print(f"Direct generation failed: {str(e)[:100]}...")
            
            # Strategy 2: Try FSDP summon_full_params
            if not success and isinstance(model, FSDP):
                try:
                    print("Trying FSDP summon_full_params...")
                    with FSDP.summon_full_params(model, writeback=False, recurse=True):
                        with autocast_ctx:
                            results = model.generate(
                                prompts, 
                                None,  # image
                                max_gen_len=args.test_prompt_max_gen_len, 
                                temperature=args.test_prompt_temperature, 
                                top_p=args.test_prompt_top_p
                            )
                    success = True
                    print("FSDP summon_full_params succeeded")
                except Exception as e:
                    print(f"FSDP summon_full_params failed: {str(e)[:100]}...")
            
            # Strategy 3: Try with state_dict_type context (alternative FSDP approach)
            if not success and isinstance(model, FSDP):
                try:
                    print("Trying FSDP state_dict_type context...")
                    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
                        with autocast_ctx:
                            results = model.generate(
                                prompts, 
                                None,  # image
                                max_gen_len=args.test_prompt_max_gen_len, 
                                temperature=args.test_prompt_temperature, 
                                top_p=args.test_prompt_top_p
                            )
                    success = True
                    print("FSDP state_dict_type context succeeded")
                except Exception as e:
                    print(f"FSDP state_dict_type context failed: {str(e)[:100]}...")
            
            # Strategy 4: Simple tokenization test (fallback)
            if not success:
                print("All generation strategies failed. Running simple tokenization test...")
                try:
                    # Just test tokenization to show the prompts
                    tokenizer = getattr(model, 'tokenizer', None)
                    if hasattr(model, 'module') and hasattr(model.module, 'tokenizer'):
                        tokenizer = model.module.tokenizer
                    elif hasattr(model, '_fsdp_wrapped_module') and hasattr(model._fsdp_wrapped_module, 'tokenizer'):
                        tokenizer = model._fsdp_wrapped_module.tokenizer
                    
                    if tokenizer:
                        for i, prompt in enumerate(prompts):
                            tokens = tokenizer.encode(prompt, bos=True, eos=False)
                            print(f"\nPrompt {i+1}: {prompt}")
                            print(f"Tokenized length: {len(tokens)} tokens")
                            print(f"Response: [Generation unavailable during FSDP training - will retry at next interval]")
                            print("-" * 40)
                        success = True
                    else:
                        print("Could not access tokenizer for fallback test")
                except Exception as e:
                    print(f"Tokenization fallback failed: {e}")
            
            # Display results if any strategy succeeded
            if success and 'results' in locals():
                for i, (prompt, result) in enumerate(zip(prompts, results)):
                    print(f"\nPrompt {i+1}: {prompt}")
                    print(f"Response: {result}")
                    print("-" * 40)
            elif not success:
                print("All strategies failed. This is common with FSDP during training.")
                print("Consider using a smaller model or testing prompts less frequently.")
                
    except Exception as e:
        print(f"Unexpected error in prompt testing: {e}")
        
    finally:
        # Always restore training state
        if was_training:
            model.train()
        print(f"{'='*60}\n")


def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, start_iter: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    model.zero_grad(set_to_none=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, batch_data in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, start_iter), start=start_iter):
        if len(batch_data) == 4:
            examples, labels, example_mask, imgs = batch_data
        else:
            examples, labels, example_mask = batch_data
            imgs = None
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_epoch(optimizer, data_iter_step / len(data_loader) + epoch, args)

        autocast_ctx = {
            "bf16": torch.amp.autocast('cuda', dtype=torch.bfloat16),
            "fp16": torch.amp.autocast('cuda', dtype=torch.float16),
            "tf32": contextlib.nullcontext(),
        }[args.precision]
        with autocast_ctx:
             c_loss, additional_loss_dict = model(examples, labels, images=imgs)
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

        for metric_name, metric in metric_logger.meters.items():
            metric_value = metric.value
            metric_value = misc.all_reduce_mean(metric_value, group=fs_init.get_data_parallel_group())
            if log_writer is not None:
                log_writer.add_scalar(metric_name, metric_value, data_iter_step + len(data_loader) * epoch)

        # test prompts during training
        if update_grad and hasattr(args, 'test_prompts') and args.test_prompts:
            test_mode = getattr(args, 'test_prompt_mode', 'auto')
            if test_mode == 'log_only':
                simple_prompt_logging(args.test_prompts, args, 
                                    (data_iter_step + 1) // accum_iter, epoch)
            else:
                test_prompts_during_training(model, args.test_prompts, args, 
                                           (data_iter_step + 1) // accum_iter, epoch)

        # save within epoch
        n_update_per_save = args.save_iteration_interval // accum_iter
        if update_grad and ((data_iter_step + 1) // accum_iter) % n_update_per_save == 0:
            misc.save_checkpoint(
                output_dir=args.output_dir,
                args=args, epoch=epoch, iteration=data_iter_step, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, dataset_state=None,
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
