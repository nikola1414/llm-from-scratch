"""Hyperparameter sweep: grid or random search over llm.train options.

Every trial is a normal training run in <out>/<trial-name>; results are ranked by the
full-validation bits per character written to result.json. Finished trials are skipped,
so an interrupted sweep can simply be restarted.

    python experiments/sweep.py --out runs/sweep --jobs 2 \
        --base "--data_dir data/corpus_bpe4k --preset cpu-tiny --max_iters 1500 --eval_interval 250" \
        --grid "lr=1e-3,2e-3" "dropout=0.0,0.1" "weight_decay=0.1"

    # random search: 8 trials, log-uniform lr, uniform dropout
    python experiments/sweep.py --out runs/rsweep --trials 8 --base "..." \
        --random "lr=log:3e-4:3e-3" "dropout=uni:0:0.3" "optimizer=choice:adamw,muon"
"""
import argparse
import itertools
import json
import math
import os
import random
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def parse_grid(specs):
    keys, values = [], []
    for spec in specs:
        k, v = spec.split("=", 1)
        keys.append(k)
        values.append(v.split(","))
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def sample_random(specs, n, seed):
    rng = random.Random(seed)
    trials = []
    for _ in range(n):
        t = {}
        for spec in specs:
            k, v = spec.split("=", 1)
            kind, rest = v.split(":", 1)
            if kind == "log":
                lo, hi = map(float, rest.split(":"))
                t[k] = f"{math.exp(rng.uniform(math.log(lo), math.log(hi))):.3g}"
            elif kind == "uni":
                lo, hi = map(float, rest.split(":"))
                t[k] = f"{rng.uniform(lo, hi):.3g}"
            elif kind == "int":
                lo, hi = map(int, rest.split(":"))
                t[k] = str(rng.randint(lo, hi))
            elif kind == "choice":
                t[k] = rng.choice(rest.split(","))
            else:
                raise ValueError(f"unknown distribution {kind!r}")
        trials.append(t)
    return trials


def trial_name(t):
    return "_".join(f"{k}={v}" for k, v in t.items()) or "base"


def to_flags(t):
    flags = []
    for k, v in t.items():
        if v in ("true", "True"):
            flags.append(f"--{k}")
        elif v not in ("false", "False"):
            flags += [f"--{k}", v]
    return flags


def run_trial(t, args):
    out = os.path.join(args.out, trial_name(t))
    if os.path.exists(os.path.join(out, "result.json")):
        return t, out
    cmd = [sys.executable, "-m", "llm.train", *shlex.split(args.base), "--out_dir", out, *to_flags(t)]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "stdout.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    print("finished", trial_name(t), flush=True)
    return t, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="", help="options shared by all trials")
    p.add_argument("--grid", nargs="*", default=[], help="key=v1,v2,...  (cartesian product)")
    p.add_argument("--random", nargs="*", default=[], help="key=log:lo:hi | uni:lo:hi | int:lo:hi | choice:a,b")
    p.add_argument("--trials", type=int, default=8, help="number of random trials")
    p.add_argument("--jobs", type=int, default=1, help="trials in parallel")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    trials = parse_grid(args.grid) if args.grid else []
    if args.random:
        extra = sample_random(args.random, args.trials, args.seed)
        trials = [{**g, **r} for g in (trials or [{}]) for r in extra]
    trials = trials or [{}]
    os.makedirs(args.out, exist_ok=True)
    env_threads = os.environ.get("OMP_NUM_THREADS")
    if env_threads is None and args.jobs > 1:
        os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 2) // args.jobs))
    print(f"{len(trials)} trials, {args.jobs} in parallel")
    with ThreadPoolExecutor(args.jobs) as pool:
        done = list(pool.map(lambda t: run_trial(t, args), trials))

    rows = []
    for t, out in done:
        path = os.path.join(out, "result.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append((json.load(f), t))
        else:
            print(f"trial failed: {trial_name(t)} (see {out}/stdout.log)")
    rows.sort(key=lambda r: r[0]["val_bpc"])
    keys = sorted({k for _, t in rows for k in t})
    print("\n| " + " | ".join(keys + ["val bits/char", "val loss", "min"]) + " |")
    print("|" + "---|" * (len(keys) + 3))
    for r, t in rows:
        print("| " + " | ".join([t.get(k, "") for k in keys] +
                                [f"**{r['val_bpc']:.4f}**", f"{r['val_loss']:.4f}", f"{r['train_time_s'] / 60:.1f}"]) + " |")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump([{"params": t, **r} for r, t in rows], f, indent=2)


if __name__ == "__main__":
    main()
