"""Tokenizers implemented from scratch.

* `BPETokenizer` — byte-level Byte Pair Encoding (the GPT-2/3/4 family of tokenizers).
  Text is split into chunks with a regex, each chunk is turned into UTF-8 bytes (so any
  string is representable and there is no <unk>), then the most frequent adjacent pair
  of symbols is merged repeatedly until the vocabulary reaches the requested size.
* `CharTokenizer` — one token per character (the v1 tokenizer), kept for comparison.

Both support special tokens (e.g. `<|endoftext|>`, `<|user|>`, `<|assistant|>`) that are
never split and can be used to build chat templates, and both serialise to plain JSON.
"""
import heapq
import json
import random
import re
from collections import Counter, defaultdict

try:
    import regex  # supports Unicode classes like \p{L}
except ImportError:  # pragma: no cover
    regex = None

# Pre-tokenization: text is split into chunks (contractions, words with their leading
# space, numbers, punctuation runs, whitespace) and merges never cross chunk boundaries,
# so tokens don't glue words to punctuation.
#   "gpt2"  — the exact GPT-2 pattern (Unicode-aware; needs the `regex` package). It is
#             what llama.cpp calls the "gpt-2" pre-tokenizer, so exported models tokenize
#             identically there.
#   "ascii" — a dependency-free approximation (ASCII letters, numbers of at most 3
#             digits); used by the first v2 checkpoints.
SPLIT_PATTERNS = {
    "gpt2": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    "ascii": r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""",
}
DEFAULT_PATTERN = "gpt2" if regex is not None else "ascii"


def compile_pattern(name):
    if name == "gpt2":
        if regex is None:
            raise ImportError("the 'gpt2' split pattern needs `pip install regex`")
        return regex.compile(SPLIT_PATTERNS[name])
    return re.compile(SPLIT_PATTERNS[name])


SPLIT_PATTERN = compile_pattern("ascii")  # kept for backwards compatibility

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]


class Tokenizer:
    """Common interface: encode / decode / vocab_size / special tokens / (de)serialise."""

    special_tokens: dict  # str -> id

    @property
    def eot_id(self):
        return self.special_tokens.get("<|endoftext|>")

    def _split_special(self, text, allowed_special):
        """Yield (is_special, piece) segments of `text`."""
        specials = [s for s in self.special_tokens if allowed_special and s in text]
        if not specials:
            yield False, text
            return
        pattern = "(" + "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")"
        for piece in re.split(pattern, text):
            if piece:
                yield piece in self.special_tokens, piece

    def encode(self, text, allowed_special=True):
        ids = []
        for is_special, piece in self._split_special(text, allowed_special):
            if is_special:
                ids.append(self.special_tokens[piece])
            else:
                ids.extend(self._encode_ordinary(piece))
        return ids

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def from_dict(d):
        kind = d["type"]
        if kind == "bpe":
            return BPETokenizer.from_dict(d)
        if kind == "char":
            return CharTokenizer.from_dict(d)
        raise ValueError(f"unknown tokenizer type {kind!r}")

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            return Tokenizer.from_dict(json.load(f))


class CharTokenizer(Tokenizer):
    def __init__(self, chars, special_tokens=DEFAULT_SPECIAL_TOKENS):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))
        self.special_tokens = {s: len(self.chars) + i for i, s in enumerate(special_tokens)}
        self.inverse_special = {i: s for s, i in self.special_tokens.items()}

    @property
    def vocab_size(self):
        return len(self.chars) + len(self.special_tokens)

    def _encode_ordinary(self, text):
        # unknown characters are dropped rather than raising
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(i) or self.inverse_special.get(i, "") for i in ids)

    def to_dict(self):
        return {"type": "char", "chars": "".join(self.chars), "special_tokens": list(self.special_tokens)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["chars"], d["special_tokens"])


class BPETokenizer(Tokenizer):
    def __init__(self, merges=None, special_tokens=DEFAULT_SPECIAL_TOKENS, pattern="ascii"):
        # merges: list of (left_id, right_id); merge i creates token id 256 + i
        self.pattern = pattern
        self.split = compile_pattern(pattern)
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        n = 256 + len(self.merges)
        self.special_tokens = {s: n + i for i, s in enumerate(special_tokens)}
        self.inverse_special = {i: s for s, i in self.special_tokens.items()}
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, text, vocab_size, special_tokens=DEFAULT_SPECIAL_TOKENS, verbose=False, pattern=None):
        """Learn `vocab_size - 256 - len(special_tokens)` merges from `text`."""
        pattern = pattern or DEFAULT_PATTERN
        num_merges = vocab_size - 256 - len(special_tokens)
        assert num_merges >= 0, "vocab_size must be at least 256 + number of special tokens"
        for s in special_tokens:  # special markers are never part of the learned merges
            text = text.replace(s, " ")

        # distinct chunks with their frequencies — merging works on these, not raw text
        chunk_counts = Counter(compile_pattern(pattern).findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        stats = defaultdict(int)          # pair -> weighted count
        where = defaultdict(set)          # pair -> indices of words containing it
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                stats[pair] += f
                where[pair].add(wi)
        # max-heap with lazy invalidation: entries are re-checked against `stats` on pop
        heap = [(-c, pair) for pair, c in stats.items()]
        heapq.heapify(heap)

        merges = []
        while len(merges) < num_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            if stats.get(pair, 0) != -neg_count or -neg_count <= 0:
                continue                   # stale entry
            new_id = 256 + len(merges)
            merges.append(pair)
            changed = defaultdict(int)
            for wi in list(where[pair]):
                w, f = words[wi], freqs[wi]
                if len(w) < 2:
                    continue
                for p in zip(w, w[1:]):    # remove this word's old pair counts
                    changed[p] -= f
                merged, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                words[wi] = merged
                for p in zip(merged, merged[1:]):   # add the new ones
                    changed[p] += f
                    where[p].add(wi)
            for p, delta in changed.items():
                if delta:
                    stats[p] += delta
                    if stats[p] > 0:
                        heapq.heappush(heap, (-stats[p], p))
            stats.pop(pair, None)
            where.pop(pair, None)
            if verbose and len(merges) % 500 == 0:
                print(f"  merge {len(merges)}/{num_merges}")
        return cls(merges, special_tokens, pattern)

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk, dropout=0.0, rng=None):
        if not dropout:
            cached = self._cache.get(chunk)
            if cached is not None:
                return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # merge the pair that was learned earliest (lowest rank), exactly as in training
            candidates = zip(ids, ids[1:])
            if dropout:
                # BPE-dropout (Provilkov et al. 2020): each possible merge is skipped with
                # probability `dropout`, so the same word gets different segmentations
                candidates = [p for p in candidates if p in self.ranks and rng.random() >= dropout]
                if not candidates:
                    break
            pair = min(candidates, key=lambda p: self.ranks.get(p, float("inf")))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            new_id, merged, i = 256 + rank, [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        if not dropout and len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text, dropout=0.0, rng=None):
        ids = []
        for chunk in self.split.findall(text):
            ids.extend(self._encode_chunk(chunk, dropout, rng))
        return ids

    def encode_with_dropout(self, text, dropout, seed=None, allowed_special=True):
        """Stochastic segmentation for training-time augmentation (see _encode_chunk)."""
        rng = random.Random(seed)
        ids = []
        for is_special, piece in self._split_special(text, allowed_special):
            ids.extend([self.special_tokens[piece]] if is_special else self._encode_ordinary(piece, dropout, rng))
        return ids

    def decode_bytes(self, ids):
        out = bytearray()
        for i in ids:
            if i in self.vocab:
                out += self.vocab[i]
            elif i in self.inverse_special:
                out += self.inverse_special[i].encode("utf-8")
        return bytes(out)

    def decode(self, ids):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def to_dict(self):
        return {"type": "bpe", "merges": [list(m) for m in self.merges],
                "special_tokens": list(self.special_tokens), "pattern": self.pattern}

    @classmethod
    def from_dict(cls, d):
        return cls(d["merges"], d["special_tokens"], d.get("pattern", "ascii"))
