"""Train a tokenizer and write train.bin / val.bin / tokenizer.json / meta.json.

Examples
    # Wizard of Oz, BPE with 2048 tokens (same 80/20 split as v1)
    python -m llm.prepare --input data/wizard_of_oz.txt --out_dir data/oz_bpe --vocab_size 2048

    # character-level, for comparison with v1
    python -m llm.prepare --input data/wizard_of_oz.txt --out_dir data/oz_char --tokenizer char

    # OpenWebText after scripts/data_extract.py: separate val file, tokenizer trained on
    # a 50 MB sample, 8 worker processes
    python -m llm.prepare --input data/output_train.txt --val_input data/output_val.txt \
        --out_dir data/owt_bpe --vocab_size 16384 --tokenizer_sample_mb 50 --workers 8
"""
import argparse
import json
import os
import time

from .data import iter_text_chunks, token_dtype, write_tokens
from .tokenizer import BPETokenizer, CharTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="training text (or the whole corpus without --val_input)")
    p.add_argument("--val_input", default=None, help="separate validation text file")
    p.add_argument("--val_fraction", type=float, default=0.2, help="tail fraction used for val without --val_input")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe")
    p.add_argument("--vocab_size", type=int, default=2048)
    p.add_argument("--tokenizer_sample_mb", type=float, default=100, help="train BPE on the first N MB")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    small = args.val_input is None
    if small:  # corpus fits in memory: split by characters like v1
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
        n = int((1 - args.val_fraction) * len(text))
        train_texts, val_texts = [text[:n]], [text[n:]]
        sample = text[:n]
    else:
        sample, budget = [], int(args.tokenizer_sample_mb * 1e6)
        for chunk in iter_text_chunks(args.input):
            sample.append(chunk)
            budget -= len(chunk)
            if budget <= 0:
                break
        sample = "".join(sample)
        train_texts, val_texts = iter_text_chunks(args.input), iter_text_chunks(args.val_input)

    t0 = time.time()
    if args.tokenizer == "bpe":
        tok = BPETokenizer.train(sample, args.vocab_size, verbose=True)
    else:
        chars = set(sample)
        if not small:
            for chunk in iter_text_chunks(args.input):
                chars.update(chunk)
            for chunk in iter_text_chunks(args.val_input):
                chars.update(chunk)
        else:
            chars.update(val_texts[0])
        tok = CharTokenizer(chars)
    print(f"tokenizer: {args.tokenizer}, vocab {tok.vocab_size} ({time.time() - t0:.1f}s)")
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))

    meta = {"vocab_size": tok.vocab_size, "dtype": token_dtype(tok.vocab_size).__name__, "tokenizer": args.tokenizer}
    for split, texts in [("train", train_texts), ("val", val_texts)]:
        t0 = time.time()
        n_tok, n_chr = write_tokens(texts, os.path.join(args.out_dir, f"{split}.bin"), tok, args.workers)
        meta[f"{split}_tokens"], meta[f"{split}_chars"] = n_tok, n_chr
        print(f"{split}: {n_chr:,} chars -> {n_tok:,} tokens ({n_chr / max(n_tok, 1):.2f} chars/token, {time.time() - t0:.1f}s)")
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
