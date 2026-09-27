"""Pretrain GPT v2.

Examples
    python -m llm.prepare --input data/wizard_of_oz.txt --out_dir data/oz_bpe --vocab_size 2048
    python -m llm.train --data_dir data/oz_bpe --out_dir runs/oz --preset cpu-small
    python -m llm.train --data_dir data/owt_bpe --out_dir runs/owt --preset gpt-small \
        --batch_size 32 --grad_accum 4 --max_iters 50000 --compile
    python -m llm.train --out_dir runs/oz --resume            # continue from last.pt
    # modern recipe: Muon + EMA + QK-norm + logit soft-cap
    python -m llm.train --data_dir data/corpus_bpe4k --out_dir runs/c --optimizer muon --ema 0.999 \
        --qk_norm --logit_softcap 30
    # multi-GPU (DistributedDataParallel): one process per GPU
    torchrun --standalone --nproc_per_node 8 -m llm.train --data_dir data/owt_bpe --preset gpt-small
"""
import argparse
import copy
import csv
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import load_checkpoint, save_checkpoint
from .data import TokenDataset
from .model import GPT, GPTConfig
from .tokenizer import Tokenizer
from .utils import autocast_context, bits_per_char, cosine_lr, evaluate_full, get_device, resolve_dtype

PRESETS = {
    # ~0.4M non-embedding params, trains in minutes on a laptop CPU
    "cpu-tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, batch_size=32),
    # ~1.9M, still CPU friendly
    "cpu-small": dict(n_layer=6, n_head=6, n_kv_head=2, n_embd=192, block_size=128, batch_size=32),
    # ~10M, a single consumer GPU
    "gpu-medium": dict(n_layer=8, n_head=8, n_kv_head=4, n_embd=384, block_size=512, batch_size=32),
    # GPT-2 small shape (~85M non-embedding)
    "gpt-small": dict(n_layer=12, n_head=12, n_kv_head=4, n_embd=768, block_size=1024, batch_size=16),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None, help="output of llm.prepare (defaults to the one used by --resume)")
    p.add_argument("--out_dir", default="runs/default")
    p.add_argument("--preset", choices=list(PRESETS), default=None)
    # model
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_kv_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--pos_emb", choices=["rope", "learned"], default="rope")
    p.add_argument("--norm", choices=["rms", "layer"], default="rms")
    p.add_argument("--mlp", choices=["swiglu", "gelu", "relu"], default="swiglu")
    p.add_argument("--no_tie", action="store_true", help="separate embedding and output weights")
    p.add_argument("--qk_norm", action="store_true", help="RMS-normalise queries and keys")
    p.add_argument("--logit_softcap", type=float, default=0.0, help="cap*tanh(logits/cap); 0 = off")
    p.add_argument("--value_residual", action="store_true", help="mix first-layer values into every layer")
    p.add_argument("--unet_skips", action="store_true", help="skip connections between mirrored layers")
    p.add_argument("--n_experts", type=int, default=0, help="mixture-of-experts MLP (0 = dense)")
    p.add_argument("--moe_top_k", type=int, default=2)
    # optimisation
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step")
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=None, help="default lr/10")
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--lr_decay_iters", type=int, default=None, help="default max_iters")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                   help="muon: Muon for hidden matrices + AdamW for the rest")
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--ema", type=float, default=0.0,
                   help="keep an exponential moving average of the weights with this decay (e.g. 0.999); "
                        "evaluated and saved as the model. 0 = off")
    # evaluation / checkpoints
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--patience", type=int, default=0, help="stop after N evals without val improvement (0=off)")
    p.add_argument("--resume", action="store_true", help="continue from <out_dir>/last.pt")
    # system
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    defaults = vars(p.parse_args([]))
    if args.resume:  # reuse the previous run's settings for every option not given now
        saved = os.path.join(args.out_dir, "args.json")
        if os.path.exists(saved):
            with open(saved) as f:
                for k, v in json.load(f).items():
                    if k in defaults and k != "resume" and getattr(args, k) == defaults[k]:
                        setattr(args, k, v)
    elif args.preset:  # a preset overrides only the options left at their defaults
        for k, v in PRESETS[args.preset].items():
            if getattr(args, k) == defaults[k]:
                setattr(args, k, v)
    if args.min_lr is None:
        args.min_lr = args.lr / 10
    if args.lr_decay_iters is None or (args.resume and args.lr_decay_iters < args.max_iters):
        args.lr_decay_iters = args.max_iters
    return args


@torch.no_grad()
def estimate_loss(model, dataset, args, device, ctx, gen):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = dataset.get_batch(split, args.batch_size, model.config.block_size, device, gen)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def setup_distributed(args):
    """torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; without them this is a single process."""
    if int(os.environ.get("RANK", -1)) == -1:
        return False, 0, 1, get_device(args.device)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    device = f"cuda:{int(os.environ['LOCAL_RANK'])}" if backend == "nccl" else "cpu"
    if backend == "nccl":
        torch.cuda.set_device(device)
    return True, rank, world, device


@torch.no_grad()
def update_ema(ema_model, model, decay, step):
    d = min(decay, (1 + step) / (10 + step))      # warm up the average early in training
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.lerp_(p.detach(), 1 - d)


def main(argv=None):
    args = parse_args(argv)
    ddp, rank, world, device = setup_distributed(args)
    master = rank == 0
    say = print if master else (lambda *a, **k: None)
    torch.manual_seed(args.seed)
    dtype = resolve_dtype(args.dtype, device)
    ctx = autocast_context(device, dtype)
    os.makedirs(args.out_dir, exist_ok=True)
    last_path, best_path = os.path.join(args.out_dir, "last.pt"), os.path.join(args.out_dir, "best.pt")

    start_iter, best_val, ckpt = 0, float("inf"), None
    if args.resume:
        model, tokenizer, ckpt = load_checkpoint(last_path, device)
        start_iter, best_val = ckpt["iter"] + 1, ckpt.get("best_val_loss", best_val)
        args.data_dir = args.data_dir or ckpt["args"]["data_dir"]
        say(f"resumed from {last_path} at iter {start_iter}")
    dataset = TokenDataset(args.data_dir)
    if not args.resume:
        tokenizer = Tokenizer.load(os.path.join(args.data_dir, "tokenizer.json"))
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
            n_head=args.n_head, n_kv_head=args.n_kv_head, n_embd=args.n_embd, dropout=args.dropout,
            bias=args.bias, pos_emb=args.pos_emb, norm=args.norm, mlp=args.mlp, tie_weights=not args.no_tie,
            qk_norm=args.qk_norm, logit_softcap=args.logit_softcap, value_residual=args.value_residual,
            unet_skips=args.unet_skips, n_experts=args.n_experts, moe_top_k=args.moe_top_k)
        model = GPT(config).to(device)
    torch.manual_seed(args.seed + rank)            # different training batches per rank

    optimizers = model.configure_optimizer(args.weight_decay, args.lr, (0.9, args.beta2), device.split(":")[0],
                                           args.optimizer, args.muon_lr, args.muon_momentum)
    if ckpt is not None and "optimizer" in ckpt:
        states = ckpt["optimizer"] if isinstance(ckpt["optimizer"], list) else [ckpt["optimizer"]]
        for opt, st in zip(optimizers, states):
            opt.load_state_dict(st)
    ema_model = None
    if args.ema:
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        if ckpt is not None and "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"])
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[int(device.split(":")[1])] if device.startswith("cuda") else None)

    n_total = sum(p.numel() for p in raw_model.parameters())
    tokens_per_step = args.batch_size * args.grad_accum * raw_model.config.block_size * world
    say(f"device {device} x{world} ({dtype}) | {n_total / 1e6:.2f}M params ({raw_model.num_params() / 1e6:.2f}M "
        f"non-embedding) | {tokens_per_step:,} tokens/step | optimizer {args.optimizer}"
        + (f" | ema {args.ema}" if args.ema else ""))
    say(f"config: {raw_model.config}")
    if master:
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        log_file = open(os.path.join(args.out_dir, "log.csv"), "a", newline="")
        log = csv.writer(log_file)
        if start_iter == 0:
            log.writerow(["iter", "train_loss", "val_loss", "val_bpc", "lr", "elapsed_s"])

    gen = torch.Generator().manual_seed(args.seed + 1)   # fixed eval batches -> comparable curves
    evals_without_improvement = 0
    t0 = time.time()
    it = start_iter
    for it in range(start_iter, args.max_iters):
        lr = cosine_lr(it, args.lr, args.min_lr, args.warmup_iters, args.lr_decay_iters)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr / args.lr

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            gen.manual_seed(args.seed + 1)
            losses = estimate_loss(raw_model, dataset, args, device, ctx, gen)
            eval_model = raw_model
            if ema_model is not None and it > 0:
                gen.manual_seed(args.seed + 1)
                ema_losses = estimate_loss(ema_model, dataset, args, device, ctx, gen)
                say(f"           ema: train {ema_losses['train']:.4f} | val {ema_losses['val']:.4f}")
                if ema_losses["val"] < losses["val"]:
                    losses, eval_model = ema_losses, ema_model
            bpc = bits_per_char(losses["val"], dataset.meta)
            elapsed = time.time() - t0
            say(f"iter {it:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | val bpc {bpc:.3f}"
                f" | lr {lr:.2e} | {elapsed:.0f}s", flush=True)
            improved = losses["val"] < best_val
            best_val = min(best_val, losses["val"])
            state = dict(iter=it, best_val_loss=best_val, args=vars(args))
            if master:
                log.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{bpc:.4f}", f"{lr:.3e}",
                              f"{elapsed:.1f}"])
                log_file.flush()
                if improved:
                    save_checkpoint(best_path, eval_model, tokenizer, **state)
                extra = {"ema_model": ema_model.state_dict()} if ema_model is not None else {}
                save_checkpoint(last_path, raw_model, tokenizer, optimizers, **state, **extra)
            evals_without_improvement = 0 if improved else evals_without_improvement + 1
            if args.patience and evals_without_improvement >= args.patience:
                say(f"early stopping: no val improvement in {args.patience} evaluations")
                break

        for micro in range(args.grad_accum):
            x, y = dataset.get_batch("train", args.batch_size, raw_model.config.block_size, device)
            if ddp:  # only all-reduce gradients on the last micro-step
                model.require_backward_grad_sync = micro == args.grad_accum - 1
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
        if args.grad_clip > 0:
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if ema_model is not None:
            update_ema(ema_model, raw_model, args.ema, it)

    result = None
    if master:
        log_file.close()
        # final report on the full validation split with the best weights
        best, _, _ = load_checkpoint(best_path, device)
        val_loss = evaluate_full(best, dataset, "val", args.batch_size, best.config.block_size, device, ctx)
        result = {"val_loss": val_loss, "val_ppl": float(torch.exp(torch.tensor(val_loss))),
                  "val_bpc": bits_per_char(val_loss, dataset.meta), "iters": it + 1,
                  "train_time_s": time.time() - t0, "params": n_total}
        print("best checkpoint, full val:", json.dumps(result))
        with open(os.path.join(args.out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        x = torch.full((1, 1), tokenizer.eot_id if tokenizer.eot_id is not None else 0, dtype=torch.long,
                       device=device)
        sample = best.generate_all(x, 200, temperature=0.8, top_k=50)
        print("--- sample ---\n" + tokenizer.decode(sample[0, 1:].tolist()))
    if ddp:
        dist.destroy_process_group()
    return result


if __name__ == "__main__":
    main()
