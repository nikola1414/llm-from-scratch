import contextlib
import math

import torch


def get_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_context(device, dtype):
    """Mixed precision on CUDA; plain float32 elsewhere unless explicitly requested."""
    if dtype == "float32":
        return contextlib.nullcontext()
    pt_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    return torch.autocast(device_type=device.split(":")[0], dtype=pt_dtype)


def resolve_dtype(dtype, device):
    if dtype != "auto":
        return dtype
    if device.startswith("cuda"):
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


def cosine_lr(it, lr, min_lr, warmup_iters, decay_iters):
    """Linear warm-up, then cosine decay from lr to min_lr, then constant min_lr."""
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    if it >= decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / max(1, decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)


@torch.no_grad()
def evaluate_full(model, dataset, split, batch_size, block_size, device, ctx, max_tokens=None):
    """Mean next-token loss over (almost) every token of a split, deterministic."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for x, y in dataset.iter_eval_batches(split, batch_size, block_size, max_tokens):
        with ctx:
            _, loss = model(x.to(device), y.to(device))
        total += loss.item() * y.numel()
        count += y.numel()
    model.train(was_training)
    return total / max(count, 1)


def bits_per_char(loss, meta, split="val"):
    """Convert per-token cross-entropy (nats) into bits per character — comparable across
    tokenizers with different vocabularies."""
    tokens_per_char = meta[f"{split}_tokens"] / meta[f"{split}_chars"]
    return loss * tokens_per_char / math.log(2)
