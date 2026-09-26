"""Token datasets: text -> binary token files -> random (x, y) batches via np.memmap.

Pre-tokenizing once and storing ids as uint16 (or uint32 for vocab > 65535) makes
training I/O trivial: a 10 GB text corpus becomes ~6 GB of tokens that the OS pages in
on demand, and every batch is a couple of array slices — no decoding in the hot loop.
"""
import json
import os
from multiprocessing import Pool

import numpy as np
import torch


def token_dtype(vocab_size):
    return np.uint16 if vocab_size < 2 ** 16 else np.uint32


_worker_tok = None


def _init_worker(tok_dict):
    global _worker_tok
    from .tokenizer import Tokenizer
    _worker_tok = Tokenizer.from_dict(tok_dict)


def _encode_worker(text):
    return _worker_tok.encode(text, allowed_special=False)


def iter_text_chunks(path, chunk_chars=1 << 20):
    """Read a (possibly huge) text file in ~1M-character pieces, split on newlines so a
    chunk boundary never cuts through a word."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        carry = ""
        while True:
            block = f.read(chunk_chars)
            if not block:
                if carry:
                    yield carry
                return
            block = carry + block
            cut = block.rfind("\n")
            if cut == -1:
                carry = block
                continue
            carry = block[cut + 1:]
            yield block[:cut + 1]


def write_tokens(texts, out_path, tokenizer, workers=1):
    """Tokenize an iterable of strings into a flat binary file; returns (#tokens, #chars)."""
    dtype = token_dtype(tokenizer.vocab_size)
    counts = {"tokens": 0, "chars": 0}

    def counted(it):  # count characters as the texts stream past (works with Pool.imap)
        for t in it:
            counts["chars"] += len(t)
            yield t

    pool = None
    if workers > 1:
        pool = Pool(workers, initializer=_init_worker, initargs=(tokenizer.to_dict(),))
        results = pool.imap(_encode_worker, counted(texts), chunksize=1)
    else:
        results = (tokenizer.encode(t, allowed_special=False) for t in counted(texts))
    with open(out_path, "wb") as f:
        for ids in results:
            np.asarray(ids, dtype=dtype).tofile(f)
            counts["tokens"] += len(ids)
    if pool is not None:
        pool.close()
        pool.join()
    return counts["tokens"], counts["chars"]


class TokenDataset:
    """Random contiguous windows from `<dir>/{train,val}.bin`."""

    def __init__(self, data_dir):
        with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.data_dir = data_dir
        self.dtype = np.dtype(self.meta["dtype"])

    def tokens(self, split):
        # re-open the memmap each call: avoids a known memory leak with long-lived memmaps
        return np.memmap(os.path.join(self.data_dir, f"{split}.bin"), dtype=self.dtype, mode="r")

    def get_batch(self, split, batch_size, block_size, device="cpu", generator=None):
        data = self.tokens(split)
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix.tolist()])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix.tolist()])
        if str(device).startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def iter_eval_batches(self, split, batch_size, block_size, max_tokens=None):
        """Deterministic, non-overlapping windows covering the split (for full evaluation)."""
        data = self.tokens(split)
        n = len(data) - 1
        if max_tokens:
            n = min(n, max_tokens)
        starts = list(range(0, n - block_size + 1, block_size))
        for b in range(0, len(starts), batch_size):
            s = starts[b:b + batch_size]
            x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in s])
            y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in s])
            yield x, y
