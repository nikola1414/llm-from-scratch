# 02 — Linear algebra & PyTorch cheat-sheet

## Shapes used everywhere

| symbol | meaning | wizard-of-oz value |
|---|---|---|
| `B` | batch size (independent sequences per step) | 32–64 |
| `T` | time / sequence length, ≤ `block_size` | 64–128 |
| `C` | channels = `n_embd` (width of each token vector) | 128–384 |
| `V` | vocabulary size | ~80 |

## Operations

* **Dot product** `a·b = Σ aᵢ bᵢ` — large when two vectors point the same way. Attention
  scores are dot products between queries and keys.
* **Matrix multiplication** `(m×n) @ (n×p) = (m×p)`; each output element is a dot product.
  `@` on 3-D tensors is a *batched* matmul: `(B,T,C) @ (B,C,T) → (B,T,T)`.
* **Linear layer** `y = x Wᵀ + b` — `nn.Linear(in, out)`.
* **Embedding** — row lookup, equivalent to `one_hot(ids) @ W`.
* **Transpose** `x.transpose(-2, -1)` swaps the last two dims.

## Int vs float

Token ids must be integer (`torch.long`) because they index tables. Everything that is
multiplied by weights must be floating point (`float32`, or `bfloat16`/`float16` for mixed
precision). Mixing dtypes in `@` raises an error.

## CPU vs GPU

* Move data and model to the same device: `model.to(device)`, `x.to(device)`.
* `device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'`
* GPU kernels are asynchronous — call `torch.cuda.synchronize()` before timing.
* GPUs win for large, parallel work; for tiny tensors the CPU can be faster because of
  kernel-launch and transfer overhead.
