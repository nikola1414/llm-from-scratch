"""Sample from a checkpoint, one-shot or as an interactive loop, with streaming output.

    python -m llm.generate --ckpt runs/oz/best.pt --prompt "Dorothy looked at the Wizard"
    python -m llm.generate --ckpt runs/oz/best.pt                     # interactive completion
    python -m llm.generate --ckpt runs/oz_sft/best.pt --chat          # chat template (after llm.finetune)
    python -m llm.generate --ckpt models/oz-chat.pt --chat --rag       # answer from retrieved passages
    python -m llm.generate --ckpt models/oz-base.pt --prompt "Ozma" --num_samples 3   # batched sampling
    python -m llm.generate --ckpt models/oz-base.pt --int8 --context 512              # int8, longer context
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


def load_model(ckpt, device="cpu", int8=False, context=None, context_scaling="ntk"):
    """Load a checkpoint, optionally with int8 weights and/or a longer (RoPE-scaled) context."""
    model, tokenizer, ck = load_checkpoint(ckpt, device)
    model.eval()
    if context and context != model.config.block_size:
        model.extend_context(context, context_scaling)
    if int8:
        model = quantize_int8(model)
    return model, tokenizer, ck


def quantize_int8(model):
    """Dynamic int8 quantization of every Linear layer (weights stored as int8, activations
    quantized on the fly). ~3-4x smaller; faster on CPU once matrices are large."""
    import warnings
    from torch.ao.quantization import quantize_dynamic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return quantize_dynamic(model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)


def sample_batch(model, tokenizer, prompt, n, args, device):
    """n independent samples of the same prompt in one batched forward pass."""
    ids = tokenizer.encode(prompt) or [tokenizer.eot_id or 0]
    x = torch.tensor([ids] * n, dtype=torch.long, device=device)
    out = model.generate_all(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                             top_p=args.top_p, repetition_penalty=args.repetition_penalty)
    return [tokenizer.decode(row[len(ids):].tolist()) for row in out]


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
    p.add_argument("--num_samples", type=int, default=1, help="with --prompt: several samples in one batch")
    p.add_argument("--rag", action="store_true", help="chat: answer from passages retrieved with BM25")
    p.add_argument("--index", default="data/corpus/index.json", help="retrieval index (python -m llm.retrieval build)")
    p.add_argument("-k", type=int, default=1, help="passages to retrieve")
    p.add_argument("--int8", action="store_true", help="dynamic int8 quantization (CPU)")
    p.add_argument("--context", type=int, default=None, help="extend the context window (RoPE NTK scaling)")
    args = p.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cpu" if args.int8 else get_device(args.device)
    model, tokenizer, _ = load_model(args.ckpt, device, args.int8, args.context)
    stop_ids = {tokenizer.special_tokens[s] for s in (EOT, USER) if s in tokenizer.special_tokens} if args.chat else None
    index = None
    if args.rag:
        from .retrieval import BM25
        index = BM25.load(args.index)

    def user_turn(message):
        if index is None:
            return message
        from .retrieval import rag_prompt
        passages = [c for c, _, _ in index.search(message, args.k)]
        return rag_prompt(message, passages)

    if args.prompt is not None:
        prompt = chat_prompt([], user_turn(args.prompt)) if args.chat else args.prompt
        if args.num_samples > 1:
            for i, text in enumerate(sample_batch(model, tokenizer, prompt, args.num_samples, args, device)):
                print(f"--- sample {i + 1} ---\n{args.prompt if not args.chat else ''}{text}")
            return
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
            if index is not None:   # each RAG question is answered from fresh passages
                history = []
            reply = stream(model, tokenizer, chat_prompt(history, user_turn(message)), args, device, stop_ids)
            history.append((user_turn(message), reply))
            # keep the conversation inside the context window
            while history and len(tokenizer.encode(chat_prompt(history, ""))) > model.config.block_size // 2:
                history.pop(0)
        else:
            sys.stdout.write(message)
            stream(model, tokenizer, message, args, device)


if __name__ == "__main__":
    main()
