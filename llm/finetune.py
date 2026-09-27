"""Supervised finetuning (instruction tuning) of a pretrained checkpoint.

Data: JSON lines, one {"prompt": ..., "response": ...} per line (see data/oz_sft.jsonl).
Each example becomes

    <|user|>{prompt}<|assistant|>{response}<|endoftext|>

and the loss is computed **only on the response tokens** (prompt targets are set to -1,
which cross_entropy ignores). The model therefore learns to answer rather than to imitate
questions, while the special tokens give it a clear signal for where a turn ends.

    python -m llm.finetune --ckpt runs/oz/best.pt --data data/oz_sft.jsonl --out_dir runs/oz_sft
    python -m llm.generate --ckpt runs/oz_sft/best.pt --chat
"""
import argparse
import json
import os
import random
import time

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .generate import ASSISTANT, EOT, USER
from .utils import autocast_context, cosine_lr, get_device, resolve_dtype


def build_example(tokenizer, prompt, response, block_size):
    p = tokenizer.encode(f"{USER}{prompt}{ASSISTANT}")
    r = tokenizer.encode(f"{response}{EOT}")
    ids = (p + r)[:block_size + 1]
    targets = ([-1] * len(p) + r)[:block_size + 1]
    return ids[:-1], targets[1:]


def collate(batch, pad_id, device):
    T = max(len(x) for x, _ in batch)
    X = torch.full((len(batch), T), pad_id, dtype=torch.long)
    Y = torch.full((len(batch), T), -1, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X.to(device), Y.to(device)


@torch.no_grad()
def eval_loss(model, examples, args, pad_id, device, ctx):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(examples), args.batch_size):
        x, y = collate(examples[i:i + args.batch_size], pad_id, device)
        with ctx:
            _, loss = model(x, y)
        count = (y != -1).sum().item()
        total, n = total + loss.item() * count, n + count
    model.train()
    return total / max(n, 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="pretrained checkpoint")
    p.add_argument("--data", required=True, help="JSONL with prompt/response pairs")
    p.add_argument("--out_dir", default="runs/sft")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=None, help="override the checkpoint's dropout")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    ctx = autocast_context(device, resolve_dtype("auto", device))
    model, tokenizer, ckpt = load_checkpoint(args.ckpt, device)
    if args.dropout is not None:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = args.dropout
            if hasattr(m, "dropout") and isinstance(m.dropout, float):
                m.dropout = args.dropout
    missing = [s for s in (USER, ASSISTANT, EOT) if s not in tokenizer.special_tokens]
    assert not missing, f"tokenizer lacks special tokens {missing}"

    with open(args.data, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    examples = [build_example(tokenizer, r["prompt"], r["response"], model.config.block_size) for r in rows]
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction)) if len(examples) > 10 else 0
    val, train = examples[:n_val], examples[n_val:]
    print(f"{len(train)} train / {len(val)} val examples, device {device}")

    os.makedirs(args.out_dir, exist_ok=True)
    optimizer = model.configure_optimizer(args.weight_decay, args.lr, (0.9, 0.99), device.split(":")[0])
    pad_id = tokenizer.special_tokens[EOT]
    steps_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    best, step, t0 = float("inf"), 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(step, args.lr, args.lr / 10, min(20, total_steps // 10), total_steps)
            x, y = collate(train[i:i + args.batch_size], pad_id, device)
            with ctx:
                _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        train_loss = eval_loss(model, train, args, pad_id, device, ctx)
        val_loss = eval_loss(model, val, args, pad_id, device, ctx) if val else train_loss
        print(f"epoch {epoch + 1:3d} | train {train_loss:.4f} | val {val_loss:.4f} | {time.time() - t0:.0f}s", flush=True)
        extra = dict(iter=step, best_val_loss=min(best, val_loss), args={**ckpt.get("args", {}), "sft": vars(args)})
        if val_loss < best:
            best = val_loss
            save_checkpoint(os.path.join(args.out_dir, "best.pt"), model, tokenizer, **extra)
        save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, tokenizer, **extra)
    print(f"best val loss {best:.4f}; saved to {args.out_dir}/best.pt")


if __name__ == "__main__":
    main()
