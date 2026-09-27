"""Download large Hugging Face datasets as plain text / JSONL for the GPU path.

    pip install datasets
    python data/download_hf.py fineweb --out data/fineweb --max_chars 10e9   # ~2.5B tokens
    python data/download_hf.py tinystories --out data/tinystories
    python data/download_hf.py dolly --out data/dolly.jsonl                  # instructions for SFT

* fineweb      HuggingFaceFW/fineweb-edu, sample-10BT: web pages filtered for educational
               quality. Streams, so no need to download the whole set first. ODC-BY licence.
* tinystories  roneneldan/TinyStories: short simple stories written by GPT-3.5/4. Small
               models learn fluent English from it very quickly. CDLA-Sharing licence.
* dolly        databricks/databricks-dolly-15k: 15k human-written instruction/response
               pairs, written to {"prompt", "response"} JSON lines for llm.finetune. CC BY-SA 3.0.

Text outputs are train.txt / val.txt with documents separated by <|endoftext|>, ready
for `python -m llm.prepare --input <out>/train.txt --val_input <out>/val.txt ...`.
"""
import argparse
import json
import os
import random

SEP = "\n<|endoftext|>\n"


def write_docs(docs, out_dir, val_every=200, max_chars=None):
    os.makedirs(out_dir, exist_ok=True)
    n = chars = 0
    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as tr, \
            open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8") as va:
        for text in docs:
            text = text.strip()
            if not text:
                continue
            f = va if n % val_every == 0 else tr
            f.write(text + SEP)
            n += 1
            chars += len(text)
            if n % 100_000 == 0:
                print(f"{n:,} documents, {chars / 1e9:.2f}B characters", flush=True)
            if max_chars and chars >= max_chars:
                break
    print(f"done: {n:,} documents, {chars:,} characters -> {out_dir}/train.txt, val.txt")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["fineweb", "tinystories", "dolly"])
    p.add_argument("--out", required=True)
    p.add_argument("--max_chars", type=float, default=None, help="stop after this many characters")
    p.add_argument("--val_every", type=int, default=200, help="every N-th document goes to val")
    args = p.parse_args()
    from datasets import load_dataset
    max_chars = int(args.max_chars) if args.max_chars else None

    if args.dataset == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        write_docs((row["text"] for row in ds), args.out, args.val_every, max_chars)
    elif args.dataset == "tinystories":
        train = load_dataset("roneneldan/TinyStories", split="train")
        val = load_dataset("roneneldan/TinyStories", split="validation")
        os.makedirs(args.out, exist_ok=True)
        for name, ds in (("train", train), ("val", val)):
            with open(os.path.join(args.out, f"{name}.txt"), "w", encoding="utf-8") as f:
                for row in ds:
                    f.write(row["text"].strip() + SEP)
        print(f"TinyStories: {len(train):,} train / {len(val):,} val stories -> {args.out}")
    else:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        rows = []
        for r in ds:
            prompt = r["instruction"].strip()
            if r.get("context"):
                prompt = f"Context: {r['context'].strip()}\n\nQuestion: {prompt}"
            rows.append({"prompt": prompt, "response": r["response"].strip()})
        random.Random(0).shuffle(rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Dolly: {len(rows):,} instruction pairs -> {args.out}")


if __name__ == "__main__":
    main()
