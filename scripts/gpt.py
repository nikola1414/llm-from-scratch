"""Decoder-only GPT language model (character level) — shared by training.py and chatbot.py.

This is the model from notebooks/06_gpt_v1.ipynb, with hyperparameters moved into a
`GPTConfig` so they can come from the command line instead of globals.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128   # context length
    n_embd: int = 384       # embedding width
    n_head: int = 8         # attention heads per block
    n_layer: int = 8        # transformer blocks
    dropout: float = 0.2


class Head(nn.Module):
    """ one head of causal self-attention """

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                                  # (B, T, hs)
        q = self.query(x)                                                # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5              # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)                                                # (B, T, hs)
        return wei @ v                                                   # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """ several heads in parallel (ModuleList), concatenated and projected """

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(head_size * config.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """ per-token MLP: expand x4, ReLU, project back """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ pre-norm transformer block: communication (attention) then computation (MLP) """

    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()   # nn.Module.__init__ must run before any submodule is assigned
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)                                    # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device))  # (T, C)
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))                                            # (B, T, V)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, index, max_new_tokens, temperature=1.0, top_k=None):
        """Sample `max_new_tokens` tokens after the (B, T) context `index`."""
        for _ in range(max_new_tokens):
            # crop: position embeddings only exist for the last block_size positions
            index_cond = index[:, -self.config.block_size:]
            logits, _ = self(index_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index


class CharTokenizer:
    """ character-level encoder/decoder built from a vocabulary string """

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.string_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_string = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        # characters unseen in training are dropped instead of raising KeyError
        return [self.string_to_int[c] for c in s if c in self.string_to_int]

    def decode(self, ids):
        return ''.join(self.int_to_string[i] for i in ids)


def get_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
