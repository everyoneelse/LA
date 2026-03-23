import math

def adjust_learning_rate(optimizer, it, args):
    """Adjust learning rate by configured schedule type."""
    warmup_iters = max(int(getattr(args, "warmup_iters", 0)), 0)
    schedule = getattr(args, "lr_schedule", "cosine")

    if schedule == "constant":
        if warmup_iters > 0 and it < warmup_iters:
            lr = args.lr * it / warmup_iters
        else:
            lr = args.lr
    elif schedule == "cosine":
        if warmup_iters > 0 and it < warmup_iters:
            lr = args.lr * it / warmup_iters
        else:
            lr_decay_iters = max(int(getattr(args, "lr_decay_iters", warmup_iters)), warmup_iters + 1)
            if it > lr_decay_iters:
                lr = args.min_lr
            else:
                decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
                decay_ratio = min(max(decay_ratio, 0.0), 1.0)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                lr = args.min_lr + (args.lr - args.min_lr) * coeff
    else:
        raise ValueError(f"Unsupported lr_schedule: {schedule}")

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate_epoch(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
