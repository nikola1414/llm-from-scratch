"""GPT v2 — a modern decoder-only transformer.

Upgrades over v1 (`scripts/gpt.py`), each switchable in `GPTConfig` for ablations:

* **RoPE** rotary position embeddings instead of a learned position table
  (`pos_emb='rope'|'learned'`) — relative positions, no extra parameters.
* **RMSNorm** instead of LayerNorm (`norm='rms'|'layer'`).
* **SwiGLU** feed-forward instead of ReLU (`mlp='swiglu'|'gelu'|'relu'`).
* **Grouped-query attention** (`n_kv_head < n_head`) — fewer K/V heads, smaller KV cache.
* **Fused attention**: one QKV projection and `F.scaled_dot_product_attention`
  (FlashAttention kernels on GPU) instead of a Python loop over heads.
* **Weight tying** between the token embedding and the output layer.
* **Scaled residual init**: output projections use std 0.02/sqrt(2·n_layer) (GPT-2).
* **KV cache** for fast generation, plus temperature / top-k / top-p / repetition penalty.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 2048
    block_size: int = 256          # max context length
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = None          # None -> n_head (standard MHA); fewer -> GQA
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False             # biases in Linear layers
    pos_emb: str = "rope"          # 'rope' | 'learned'
    norm: str = "rms"              # 'rms' | 'layer'
    mlp: str = "swiglu"            # 'swiglu' | 'gelu' | 'relu'
    tie_weights: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        if self.pos_emb == "rope":
            assert (self.n_embd // self.n_head) % 2 == 0, "RoPE needs an even head size"

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    """x / sqrt(mean(x²) + eps) · g — LayerNorm without mean-centering or bias."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def make_norm(config):
    if config.norm == "rms":
        return RMSNorm(config.n_embd)
    return nn.LayerNorm(config.n_embd, bias=config.bias)


def rope_cache(seq_len, head_dim, theta, device=None):
    """cos/sin tables of shape (seq_len, head_dim/2) for rotary embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Rotate each (even, odd) half-pair of x (B, H, T, D) by a position-dependent angle.

    The dot product of two rotated vectors depends only on their *relative* position,
    which is how RoPE injects order into attention.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos, sin = cos[None, None], sin[None, None]           # (1, 1, T, D/2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_kv_head = config.n_head, config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.head_dim, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.proj.RESIDUAL_SCALE = True
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, rope=None, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(
            [self.n_head * self.head_dim, self.n_kv_head * self.head_dim, self.n_kv_head * self.head_dim], dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B, H, T, D)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, KV, T, D)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)

        past = 0
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                past = kv_cache["k"].shape[2]
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v

        if self.n_kv_head != self.n_head:                                  # GQA: share K/V heads
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # past == 0: ordinary causal attention. past > 0 with T == 1: the new token may
        # attend to everything in the cache, so no mask is needed.
        assert past == 0 or T == 1, "cached decoding feeds one token at a time"
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(past == 0 and T > 1),
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kind = config.mlp
        if self.kind == "swiglu":
            # 2/3 of 4·C keeps the parameter count equal to a 4x ReLU MLP; round up to 32
            hidden = int(2 * 4 * config.n_embd / 3)
            hidden = 32 * ((hidden + 31) // 32)
            self.gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        else:
            hidden = 4 * config.n_embd
        self.up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.down.RESIDUAL_SCALE = True
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.kind == "swiglu":
            h = F.silu(self.gate(x)) * self.up(x)
        elif self.kind == "gelu":
            h = F.gelu(self.up(x))
        else:
            h = F.relu(self.up(x))
        return self.dropout(self.down(h))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.norm2 = make_norm(config)
        self.mlp = MLP(config)

    def forward(self, x, rope=None, kv_cache=None):
        x = x + self.attn(self.norm1(x), rope, kv_cache)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd) if config.pos_emb == "learned" else None
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = make_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        if config.pos_emb == "rope":
            cos, sin = rope_cache(config.block_size, config.n_embd // config.n_head, config.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE", False):
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
            if not self.config.tie_weights:
                n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        """idx (B, T) -> logits (B, T, V). With `targets`, also returns the mean
        cross-entropy (positions whose target is -1 are ignored, e.g. padded or prompt
        tokens during finetuning). Without targets only the last position's logits are
        computed, which is all generation needs."""
        B, T = idx.shape
        assert start_pos + T <= self.config.block_size, \
            f"sequence of length {start_pos + T} exceeds block_size {self.config.block_size}"
        x = self.tok_emb(idx)
        rope = None
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(start_pos, start_pos + T, device=idx.device))
        else:
            rope = (self.rope_cos[start_pos:start_pos + T], self.rope_sin[start_pos:start_pos + T])
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, rope, kv_caches[i] if kv_caches is not None else None)
        x = self.norm_f(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, betas=(0.9, 0.95), device_type="cpu"):
        """AdamW with weight decay only on matrices (not on norms/biases)."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        extra = {"fused": True} if device_type == "cuda" else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, stop_ids=None, use_cache=True):
        """Autoregressive sampling. Yields each new token id (B must be 1 when streaming
        with stop_ids) and finally leaves the full sequence in `self.last_sequence`.

        With the KV cache, each step feeds only the newest token. When the context fills
        `block_size`, the cache is rebuilt from the most recent half-window.
        """
        was_training = self.training
        self.eval()
        caches, pos = None, 0
        stop_ids = set(stop_ids or [])
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _ = self(idx[:, -self.config.block_size:])
            elif caches is None or pos >= self.config.block_size:
                keep = self.config.block_size // 2 if caches is not None else self.config.block_size
                window = idx[:, -keep:]
                caches = [{} for _ in self.blocks]
                logits, _ = self(window, kv_caches=caches, start_pos=0)
                pos = window.shape[1]
            else:
                logits, _ = self(idx[:, -1:], kv_caches=caches, start_pos=pos)
                pos += 1
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                prev = idx[:, -self.config.block_size:]
                scores = logits.gather(1, prev)
                scores = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
                logits.scatter_(1, prev, scores)
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cum - F.softmax(sorted_logits, dim=-1) > top_p   # keep the first token over p
                    sorted_logits[remove] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            yield next_id
            if stop_ids and next_id.numel() == 1 and next_id.item() in stop_ids:
                break
        self.last_sequence = idx
        self.train(was_training)

    def generate_all(self, idx, max_new_tokens, **kwargs):
        """Non-streaming convenience wrapper: returns (B, T + new) token ids."""
        for _ in self.generate(idx, max_new_tokens, **kwargs):
            pass
        return self.last_sequence
