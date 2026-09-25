# 04 — Transformers, self-attention and the GPT architecture

## Transformer and self-attention

The bigram model looks at one character. To use *context*, each token must gather
information from previous tokens. **Self-attention** does this with data-dependent weights:

1. Every token emits three vectors via linear layers:
   * **query** `q` — "what am I looking for?"
   * **key** `k` — "what do I contain?"
   * **value** `v` — "what will I give you if you attend to me?"
2. Affinity between token *i* and *j* = `qᵢ · kⱼ` (a dot product).
3. Scale by `1/√d_k`, mask out the future (causal), softmax → weights that sum to 1.
4. Output for token *i* = weighted sum of the values: `Σⱼ wᵢⱼ vⱼ`.

```
Attention(Q, K, V) = softmax( mask(Q Kᵀ / √d_k) ) V
```

"Self" because Q, K and V all come from the same sequence (cross-attention takes K,V from
another sequence, e.g. an encoder).

## The original Transformer architecture ("Attention Is All You Need", 2017)

```
 inputs ─► embedding + positional encoding ─► [ Encoder × N ]──┐
                                                               │ K,V
 outputs(shifted) ─► embedding + pos ─► [ Decoder × N ] ◄──────┘ ─► Linear ─► Softmax
 Encoder block:  multi-head self-attention → add&norm → feed-forward → add&norm
 Decoder block:  masked self-attention → add&norm → cross-attention → add&norm → FFN → add&norm
```

It was built for translation (sequence → sequence).

## Building a GPT, not a Transformer

GPT (Generative Pre-trained Transformer) keeps **only the decoder**, and drops
cross-attention because there is no encoder. It is trained on one task: predict the next
token. That is all we need for text generation.

## GPT architecture (what we implement)

```
index (B,T)
  │ token_embedding_table (V, C)      +   position_embedding_table (block_size, C)
  ▼
x (B,T,C)
  │  Block × n_layer:
  │     x = x + MultiHeadAttention(LayerNorm(x))     # communication between tokens
  │     x = x + FeedForward(LayerNorm(x))            # computation per token
  ▼
LayerNorm → Linear (C → V) → logits (B,T,V) → cross-entropy / softmax sampling
```

* **Residual connections** (`x + f(x)`) give gradients a highway through deep stacks.
* **Pre-norm** (LayerNorm *before* each sub-layer, as in GPT-2) trains more stably than the
  original post-norm `LayerNorm(x + f(x))`.
* **Multi-head attention**: `n_head` heads of size `C / n_head` run in parallel, each can
  specialise (syntax, long-range, punctuation…); outputs are concatenated and projected.
* **FeedForward**: `Linear(C, 4C) → ReLU → Linear(4C, C) → Dropout`.
* **Dropout** randomly zeroes activations during training to fight overfitting.

## Positional encoding

Attention is permutation-invariant: it has no idea of order. We add a **learned
positional embedding** (`nn.Embedding(block_size, n_embd)`) to the token embedding. The
original paper used fixed sinusoids; newer models use RoPE/ALiBi.

## Why we scale by 1/√d_k

If `q` and `k` have unit-variance entries, `q·k` sums `d_k` terms and has variance `d_k`.
Large scores push softmax into a near one-hot distribution (one weight ≈ 1, the rest ≈ 0)
whose gradients vanish. Dividing by `√d_k` restores unit variance so softmax stays diffuse
at initialization. Notebook 05 demonstrates this numerically.

## Standard deviation for model parameters

Weights are initialised from `N(0, 0.02)` and biases to zero (as in GPT-2). Too large a std
explodes activations; too small makes all outputs identical and slows learning.

## `nn.Sequential` vs `nn.ModuleList`

* `nn.Sequential(*blocks)` — calls each module in order, output of one feeds the next.
  Perfect for the stack of Blocks (`self.blocks(x)`).
* `nn.ModuleList(heads)` — just a registered list; *you* decide how to call them. Heads run
  **in parallel** on the same input and are concatenated:
  `torch.cat([h(x) for h in self.heads], dim=-1)`.

Both register parameters so `.to(device)`, `.parameters()` and saving work.
