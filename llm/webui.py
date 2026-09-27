"""Browser chat UI (Gradio) for a trained checkpoint.

    pip install gradio
    python -m llm.webui --ckpt models/oz-chat.pt --rag          # then open http://127.0.0.1:7860

"Chat" mode uses the <|user|>/<|assistant|> template (optionally grounded with retrieved
passages, which are shown under the answer); "Complete" mode continues the text as a
plain language model. Replies stream token by token.
"""
import argparse

import torch

from .generate import ASSISTANT, EOT, USER, chat_prompt, load_model
from .utils import get_device


def build_app(model, tokenizer, index=None, device="cpu"):
    import gradio as gr

    stop = {tokenizer.special_tokens[s] for s in (EOT, USER) if s in tokenizer.special_tokens}

    def reply(message, history, mode, use_rag, temperature, top_p, max_new_tokens):
        passages = []
        if mode == "Chat":
            user = message
            if use_rag and index is not None:
                from .retrieval import rag_prompt
                passages = [c for c, _, _ in index.search(message, 1)]
                user = rag_prompt(message, passages)
            prompt, stop_ids = chat_prompt([], user), stop
        else:
            prompt, stop_ids = message, None
        ids = tokenizer.encode(prompt)[-(model.config.block_size - 1):] or [tokenizer.eot_id or 0]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        new = []
        for next_id in model.generate(x, int(max_new_tokens), temperature=temperature, top_k=50, top_p=top_p,
                                      repetition_penalty=1.1, stop_ids=stop_ids):
            tok = next_id.item()
            if stop_ids and tok in stop_ids:
                break
            new.append(tok)
            text = tokenizer.decode(new)
            yield (message + text) if mode == "Complete" else text
        text = tokenizer.decode(new)
        if passages:
            text += "\n\n---\n*Retrieved passage:* " + passages[0]
        yield (message + text) if mode == "Complete" else text

    return gr.ChatInterface(
        reply,
        title="LLM from scratch",
        description="A small GPT trained from scratch on public-domain Oz books.",
        additional_inputs=[
            gr.Radio(["Chat", "Complete"], value="Chat", label="Mode"),
            gr.Checkbox(value=index is not None, label="Retrieve passages (RAG)", interactive=index is not None),
            gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
            gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p"),
            gr.Slider(16, 400, value=150, step=8, label="Max new tokens"),
        ],
        examples=[["Who is Billina?"], ["Where does Ozma live?"], ["What do the Gargoyles fear most?"]],
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="models/oz-chat.pt")
    p.add_argument("--rag", action="store_true")
    p.add_argument("--index", default="data/corpus/index.json")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = p.parse_args(argv)
    device = "cpu" if args.int8 else get_device(args.device)
    model, tokenizer, _ = load_model(args.ckpt, device, args.int8)
    index = None
    if args.rag:
        from .retrieval import BM25
        index = BM25.load(args.index)
    build_app(model, tokenizer, index, device).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
