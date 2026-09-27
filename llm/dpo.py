"""Direct Preference Optimization (Rafailov et al., 2023).

Given pairs (prompt, chosen, rejected), DPO raises the likelihood of the chosen answer
relative to the rejected one, measured against a frozen reference model (the SFT model):

    loss = -log σ( β · [(log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r))] )

No reward model and no reinforcement learning loop are needed. β controls how far the
policy may move from the reference. An optional NLL term on the chosen answer
(`--sft_coef`, as in RPO) keeps the chosen likelihood from drifting down.

    python -m llm.dpo --ckpt runs/chat_sft/best.pt --data data/chat/dpo_train.jsonl \
        --val_data data/chat/dpo_val.jsonl --out_dir runs/chat_dpo
"""
import argparse
import copy
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, save_checkpoint
from .generate import ASSISTANT, EOT, USER
from .utils import cosine_lr, get_device


def encode_pair(tokenizer, prompt, answer, block_size):
    p = tokenizer.encode(f"{USER}{prompt}{ASSISTANT}")
    r = tokenizer.encode(f"{answer}{EOT}")
    ids = (p + r)[-(block_size + 1):]              # keep the answer if too long
    n_prompt = max(0, len(ids) - len(r))
    return ids, n_prompt


def batch_logps(model, seqs, pad_id, device):
    """Sum of log-probabilities of the answer tokens of each sequence."""
    T = max(len(ids) for ids, _ in seqs) - 1
    x = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    y = torch.full((len(seqs), T), -1, dtype=torch.long)
    for i, (ids, n_prompt) in enumerate(seqs):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1])
        tgt = torch.tensor(ids[1:])
        tgt[: max(0, n_prompt - 1)] = -1
        y[i, :n] = tgt
    x, y = x.to(device), y.to(device)
    logits, _ = model(x, y)
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = y != -1
    tok = logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
    return tok.sum(-1), mask.sum(-1)


def dpo_step(policy, ref, batch, tokenizer, args, device):
    pad = tokenizer.special_tokens[EOT]
    chosen = [encode_pair(tokenizer, r["prompt"], r["chosen"], policy.config.block_size) for r in batch]
    rejected = [encode_pair(tokenizer, r["prompt"], r["rejected"], policy.config.block_size) for r in batch]
    pc, nc = batch_logps(policy, chosen, pad, device)
    pr, _ = batch_logps(policy, rejected, pad, device)
    with torch.no_grad():
        rc, _ = batch_logps(ref, chosen, pad, device)
        rr, _ = batch_logps(ref, rejected, pad, device)
    margin = args.beta * ((pc - rc) - (pr - rr))
    loss = -F.logsigmoid(margin).mean()
    if args.sft_coef:
        loss = loss + args.sft_coef * (-pc / nc.clamp(min=1)).mean()
    return loss, (margin > 0).float().mean().item(), margin.mean().item()


@torch.no_grad()
def evaluate(policy, ref, rows, tokenizer, args, device):
    policy.eval()
    tot_loss = tot_acc = tot_margin = 0.0
    n = 0
    for i in range(0, len(rows), args.batch_size):
        b = rows[i:i + args.batch_size]
        loss, acc, margin = dpo_step(policy, ref, b, tokenizer, args, device)
        tot_loss, tot_acc, tot_margin, n = tot_loss + loss.item() * len(b), tot_acc + acc * len(b), \
            tot_margin + margin * len(b), n + len(b)
    policy.train()
    return tot_loss / n, tot_acc / n, tot_margin / n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="SFT checkpoint (policy init and frozen reference)")
    p.add_argument("--data", required=True, help="JSONL with prompt/chosen/rejected")
    p.add_argument("--val_data", default=None)
    p.add_argument("--out_dir", default="runs/dpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--sft_coef", type=float, default=0.2)
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    policy, tokenizer, ckpt = load_checkpoint(args.ckpt, device)
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    for m in policy.modules():                      # no dropout: policy and reference must agree at start
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
        if hasattr(m, "dropout") and isinstance(m.dropout, float):
            m.dropout = 0.0
    load = lambda path: [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]  # noqa: E731
    train = load(args.data)[: args.max_pairs]
    val = load(args.val_data) if args.val_data else train[: max(1, len(train) // 20)]

    (opt,) = policy.configure_optimizer(0.0, args.lr, (0.9, 0.99), device.split(":")[0])
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    os.makedirs(args.out_dir, exist_ok=True)
    vloss, vacc, vmargin = evaluate(policy, ref, val, tokenizer, args, device)
    print(f"{len(train)} pairs | start: val loss {vloss:.4f}, preference accuracy {vacc:.3f}")
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, args.lr, args.lr / 10, max(1, steps // 20), steps)
            loss, acc, margin = dpo_step(policy, ref, train[i:i + args.batch_size], tokenizer, args, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{steps} | loss {loss.item():.4f} | acc {acc:.2f} | margin {margin:.3f} "
                      f"| {time.time() - t0:.0f}s", flush=True)
    vloss, vacc, vmargin = evaluate(policy, ref, val, tokenizer, args, device)
    print(f"end: val loss {vloss:.4f}, preference accuracy {vacc:.3f}, mean margin {vmargin:.3f}")
    save_checkpoint(os.path.join(args.out_dir, "best.pt"), policy, tokenizer, iter=step,
                    args={**ckpt.get("args", {}), "dpo": vars(args)}, dpo_val={"loss": vloss, "accuracy": vacc})
    return {"val_loss": vloss, "val_accuracy": vacc}


if __name__ == "__main__":
    main()
