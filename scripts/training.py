"""Train (or resume training) the GPT language model.

Examples
    # small model on the Wizard of Oz, runs on a laptop CPU
    python scripts/training.py --data wizard --batch_size 32 --block_size 64 \
        --n_embd 128 --n_head 4 --n_layer 4 --max_iters 3000

    # OpenWebText (run scripts/data_extract.py first), GPU-sized model
    python scripts/training.py --data openwebtext --batch_size 64 --block_size 128 \
        --n_embd 384 --n_head 8 --n_layer 8 --max_iters 20000

    # keep training an existing model
    python scripts/training.py --data openwebtext --resume --model_path model-01.pkl
"""
import argparse
import mmap
import os
import pickle
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt import CharTokenizer, GPTConfig, GPTLanguageModel, get_device  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', choices=['wizard', 'openwebtext'], default='wizard')
    parser.add_argument('--data_dir', default=os.path.join(ROOT, 'data'))
    parser.add_argument('--batch_size', '-batch_size', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--max_iters', type=int, default=3000)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--learning_rate', '--lr', type=float, default=3e-4)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--model_path', default=os.path.join(ROOT, 'model-01.pkl'))
    parser.add_argument('--resume', action='store_true', help='load --model_path and keep training')
    parser.add_argument('--device', default=None, help='cuda | mps | cpu (auto by default)')
    parser.add_argument('--seed', type=int, default=1337)
    return parser.parse_args()


class InMemoryData:
    """ small corpus: the whole text is one tensor, 80/20 split """

    def __init__(self, path, tokenizer=None):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokenizer = tokenizer or CharTokenizer(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        n = int(0.8 * len(data))
        self.splits = {'train': data[:n], 'val': data[n:]}

    def get_batch(self, split, batch_size, block_size):
        data = self.splits[split]
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y


class MemoryMappedData:
    """ huge corpus: read a random chunk of the train/val file through mmap each batch """

    def __init__(self, data_dir, tokenizer=None):
        self.files = {
            'train': os.path.join(data_dir, 'output_train.txt'),
            'val': os.path.join(data_dir, 'output_val.txt'),
        }
        if tokenizer is None:
            with open(os.path.join(data_dir, 'vocab.txt'), 'r', encoding='utf-8') as f:
                tokenizer = CharTokenizer(f.read())
        self.tokenizer = tokenizer

    def get_random_chunk(self, split, batch_size, block_size):
        with open(self.files[split], 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                chunk_len = block_size * batch_size
                start_pos = random.randint(0, max(0, mm.size() - chunk_len))
                mm.seek(start_pos)
                block = mm.read(chunk_len - 1)
                decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
                return torch.tensor(self.tokenizer.encode(decoded_block), dtype=torch.long)

    def get_batch(self, split, batch_size, block_size):
        data = self.get_random_chunk(split, batch_size, block_size)
        # a chunk can shrink a little (dropped bytes/chars), re-read if it became too short
        while len(data) <= block_size + 1:
            data = self.get_random_chunk(split, batch_size, block_size)
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y


@torch.no_grad()
def estimate_loss(model, dataset, args, block_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = dataset.get_batch(split, args.batch_size, block_size)
            _, loss = model(X.to(device), Y.to(device))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = get_device(args.device)
    print(f'device: {device}')

    tokenizer = None
    if args.resume:
        print(f'loading model parameters from {args.model_path}')
        with open(args.model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        model, tokenizer = checkpoint['model'], checkpoint['tokenizer']

    if args.data == 'wizard':
        dataset = InMemoryData(os.path.join(args.data_dir, 'wizard_of_oz.txt'), tokenizer)
    else:
        dataset = MemoryMappedData(args.data_dir, tokenizer)
    tokenizer = dataset.tokenizer

    if not args.resume:
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size, block_size=args.block_size, n_embd=args.n_embd,
            n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout,
        )
        model = GPTLanguageModel(config)
    model = model.to(device)
    block_size = model.config.block_size
    print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, config: {model.config}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start = time.time()
    for iter in range(args.max_iters):
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss(model, dataset, args, block_size, device)
            msg = f"step {iter}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f} ({time.time() - start:.0f}s)"
            if device == 'cuda':
                msg += f', gpu mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB'
            print(msg, flush=True)

        xb, yb = dataset.get_batch('train', args.batch_size, block_size)
        _, loss = model(xb.to(device), yb.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f'final batch loss: {loss.item():.3f}')

    # pickle model + tokenizer on the CPU so the file loads on machines without a GPU
    model.to('cpu')
    with open(args.model_path, 'wb') as f:
        pickle.dump({'model': model, 'tokenizer': tokenizer}, f)
    print(f'model saved to {args.model_path}')

    model.to(device).eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print('--- sample ---')
    print(tokenizer.decode(model.generate(context, max_new_tokens=300)[0].tolist()))


if __name__ == '__main__':
    main()
