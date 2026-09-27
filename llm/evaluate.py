"""Evaluate a checkpoint on the full validation (or train) split.

    python -m llm.evaluate --ckpt runs/oz/best.pt --data_dir data/oz_bpe

Reports cross-entropy per token, perplexity and bits per character. Bits per character
is the fair comparison between models with different tokenizers (a char model and a
BPE model predict different units, so their raw losses are not comparable).
"""
import argparse
import json
import math

from .checkpoint import load_checkpoint
from .data import TokenDataset
from .utils import autocast_context, bits_per_char, evaluate_full, get_device, resolve_dtype


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default=None, help="defaults to the data_dir stored in the checkpoint")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = get_device(args.device)
    model, _, ckpt = load_checkpoint(args.ckpt, device)
    data_dir = args.data_dir or ckpt["args"]["data_dir"]
    dataset = TokenDataset(data_dir)
    ctx = autocast_context(device, resolve_dtype("auto", device))
    loss = evaluate_full(model, dataset, args.split, args.batch_size, model.config.block_size, device, ctx, args.max_tokens)
    result = {"split": args.split, "loss": round(loss, 4), "perplexity": round(math.exp(loss), 2),
              "bits_per_char": round(bits_per_char(loss, dataset.meta, args.split), 4)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
