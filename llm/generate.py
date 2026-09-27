"""Sample from a checkpoint, one-shot or as an interactive loop, with streaming output.

    python -m llm.generate --ckpt runs/oz/best.pt --prompt "Dorothy looked at the Wizard"
    python -m llm.generate --ckpt runs/oz/best.pt                     # interactive completion
    python -m llm.generate --ckpt runs/oz_sft/best.pt --chat          # chat template (after llm.finetune)
"""
import argparse
import sys

import torch

from .checkpoint import load_checkpoint
from .utils import get_device

USER, ASSISTANT, EOT = "<|user|>", "<|assistant|>", "<|endoftext|>"


def chat_prompt(history, message):
    """history: list of (user, assistant) turns."""
    text = "".join(f"{USER}{u}{ASSISTANT}{a}{EOT}" for u, a in history)
    return text + f"{USER}{message}{ASSISTANT}"


def stream(model, tokenizer, prompt, args, device, stop_ids=None, out=sys.stdout):
    ids = tokenizer.encode(prompt) or [tokenizer.eot_id or 0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    new_ids, printed = [], ""
    for next_id in model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                                  top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                                  stop_ids=stop_ids):
        tok = next_id.item()
        if stop_ids and tok in stop_ids:
            break
        new_ids.append(tok)
        text = tokenizer.decode(new_ids)
        if not text.endswith("�"):        # wait until a multi-byte character is complete
            out.write(text[len(printed):])
            out.flush()
            printed = text
    out.write(tokenizer.decode(new_ids)[len(printed):] + "\n")
    return tokenizer.decode(new_ids)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default=None, help="complete this prompt and exit")
    p.add_argument("--chat", action="store_true", help="use the <|user|>/<|assistant|> template")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = get_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.ckpt, device)
    model.eval()
    stop_ids = {tokenizer.special_tokens[s] for s in (EOT, USER) if s in tokenizer.special_tokens} if args.chat else None

    if args.prompt is not None:
        prompt = chat_prompt([], args.prompt) if args.chat else args.prompt
        if not args.chat:
            sys.stdout.write(args.prompt)
        stream(model, tokenizer, prompt, args, device, stop_ids)
        return

    history = []
    print("Type a prompt (empty line to quit).")
    while True:
        try:
            message = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            break
        if args.chat:
            reply = stream(model, tokenizer, chat_prompt(history, message), args, device, stop_ids)
            history.append((message, reply))
            # keep the conversation inside the context window
            while history and len(tokenizer.encode(chat_prompt(history, ""))) > model.config.block_size // 2:
                history.pop(0)
        else:
            sys.stdout.write(message)
            stream(model, tokenizer, message, args, device)


if __name__ == "__main__":
    main()
