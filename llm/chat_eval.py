"""Score a chat model on held-out questions: an answer counts as correct if it contains
one of the expected keywords (data/chat_eval.jsonl). With --rag the question is answered
from the passage retrieved by BM25 (the same prompt format used in finetuning).

    python -m llm.chat_eval --ckpt models/oz-chat.pt
    python -m llm.chat_eval --ckpt runs/chat_dpo/best.pt --rag --index data/corpus/index.json
"""
import argparse
import json

import torch

from .checkpoint import load_checkpoint
from .generate import ASSISTANT, EOT, USER, chat_prompt
from .retrieval import BM25, rag_prompt
from .utils import get_device


@torch.no_grad()
def answer(model, tokenizer, question, index=None, k=1, max_new_tokens=80, device="cpu"):
    user = rag_prompt(question, [c for c, _, _ in index.search(question, k)]) if index else question
    ids = tokenizer.encode(chat_prompt([], user))[-(model.config.block_size - max_new_tokens):]
    x = torch.tensor([ids], device=device)
    stop = {tokenizer.special_tokens[EOT], tokenizer.special_tokens[USER]}
    out = model.generate_all(x, max_new_tokens, temperature=0, stop_ids=stop, repetition_penalty=1.1)
    return tokenizer.decode([i for i in out[0, len(ids):].tolist() if i not in stop]).strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval", default="data/chat_eval.jsonl")
    p.add_argument("--rag", action="store_true")
    p.add_argument("--index", default="data/corpus/index.json")
    p.add_argument("-k", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    device = get_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    index = BM25.load(args.index) if args.rag else None
    rows = [json.loads(line) for line in open(args.eval, encoding="utf-8") if line.strip()]
    correct = 0
    for r in rows:
        a = answer(model, tokenizer, r["question"], index, args.k, device=device)
        ok = any(kw.lower() in a.lower() for kw in r["keywords"])
        correct += ok
        if not args.quiet:
            print(f"[{'x' if ok else ' '}] {r['question']}\n    {a}")
    score = correct / len(rows)
    print(f"accuracy: {correct}/{len(rows)} = {score:.2%}")
    return score


if __name__ == "__main__":
    main()
