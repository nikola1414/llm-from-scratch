"""Interactive prompt -> completion with a trained model.

    python scripts/chatbot.py --model_path model-01.pkl --max_new_tokens 150

Type a prompt, get the model's continuation. Empty line or Ctrl-C to quit.
"""
import argparse
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt import get_device  # noqa: E402  (also makes gpt.* classes importable for unpickling)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='chat with the trained GPT')
    parser.add_argument('--model_path', default=os.path.join(ROOT, 'model-01.pkl'))
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--prompt', default=None, help='complete one prompt and exit')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f'loading model parameters from {args.model_path}')
    with open(args.model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    model, tokenizer = checkpoint['model'].to(device), checkpoint['tokenizer']
    model.eval()
    print('loaded successfully!')

    def complete(prompt):
        ids = tokenizer.encode(prompt) or [0]   # never feed an empty context
        context = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(context, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k)
        return tokenizer.decode(out[0].tolist())

    if args.prompt is not None:
        print(complete(args.prompt))
        return

    while True:
        try:
            prompt = input('Prompt:\n')
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        print(f'Completion:\n{complete(prompt)}\n')


if __name__ == '__main__':
    main()
