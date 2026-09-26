"""Print a markdown table of runs/<dir>/*/result.json (sorted by validation bits/char)."""
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "runs/ablation"
rows = []
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name, "result.json")
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        with open(os.path.join(root, name, "args.json")) as f:
            a = json.load(f)
        rows.append((name, r, a))
print("| run | data | params | steps | val loss/token | val bits/char | time |")
print("|---|---|---|---|---|---|---|")
for name, r, a in sorted(rows, key=lambda t: t[1]["val_bpc"]):
    print(f"| {name} | {os.path.basename(a['data_dir'])} | {r['params'] / 1e6:.2f}M | {r['iters']} | "
          f"{r['val_loss']:.3f} | **{r['val_bpc']:.3f}** | {r['train_time_s'] / 60:.1f} min |")
