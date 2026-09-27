# 09 — v2: making the model as good as possible

`scripts/gpt.py` (v1) follows the course. `llm/` (v2) applies what modern LLMs
(GPT-2/3, Llama, Mistral) learned since then. Every change can be switched off from the
command line, so you can measure it yourself (see `experiments/ablation.sh` and the
results in the README).

## Tokenizer: byte-level BPE, trained from scratch (`llm/tokenizer.py`)

* A regex splits text into words, numbers, punctuation and whitespace (GPT-2 style). Each
  piece is converted to UTF-8 **bytes**, so every string can be encoded and there are no
  unknown tokens.
* Training repeatedly merges the most frequent adjacent pair. A heap with lazy
  invalidation, plus an index from pair to words, keeps this fast: 2048 tokens train in
  under a second on the book, and a 50 MB sample trains in minutes.
* Encoding applies merges in the order they were learned, and caches each distinct
  piece.
* Special tokens `<|endoftext|>`, `<|user|>`, `<|assistant|>` are never split; they are
  used for document boundaries and the chat template.
* Why it helps: about 3 characters per token means the same `block_size` sees about 3×
  more text, and the model predicts whole word pieces instead of spelling them out.

## Architecture (`llm/model.py`)

| v1 | v2 | why |
|---|---|---|
| learned position table | **RoPE** (rotary embeddings) | rotating q/k by angles proportional to position makes attention depend on *relative* distance; no parameters; generalises better |
| LayerNorm | **RMSNorm** | cheaper (no mean, no bias), equally stable (Llama) |
| ReLU MLP (4×) | **SwiGLU** (`silu(xW₁) ⊙ xW₂`, hidden 8/3·C) | the gated MLP consistently lowers loss at equal parameters (PaLM, Llama) |
| one `Head` module per head, Python loop | **fused QKV + `F.scaled_dot_product_attention`** | one matmul for all heads; FlashAttention/memory-efficient kernels on GPU |
| n_head K/V heads | **grouped-query attention** (`--n_kv_head`) | several query heads share one K/V head, so the KV cache is smaller and generation faster (Llama 2/3, Mistral) |
| separate `lm_head` | **weight tying** | the input embedding and the output layer share one matrix; fewer parameters, better with little data |
| std 0.02 everywhere | **scaled residual init** 0.02/√(2·n_layer) on output projections | keeps the variance of the residual stream constant with depth (GPT-2) |
| biases everywhere | no biases | slightly faster, no loss in quality |

## Training (`llm/train.py`)

* **AdamW** with β₂ = 0.99 and **weight decay 0.1 on matrices only** (no decay on norms or
  biases).
* **Learning-rate schedule**: linear warm-up, then cosine decay to lr/10. The peak lr
  (1e-3) is higher than v1's constant 3e-4.
* **Gradient clipping** at norm 1.0, which prevents rare exploding steps.
* **Gradient accumulation** (`--grad_accum`) for large effective batches on small GPUs.
* **Mixed precision**: bf16 autocast on CUDA, or fp16 with a GradScaler on older GPUs.
* **`torch.compile`** (`--compile`) for fused kernels.
* **Pre-tokenized memmap data** (`llm/prepare.py`, `llm/data.py`): tokens are stored as
  uint16, so a batch is two array slices with no text decoding in the training loop.
  Tokenization can use several processes (`--workers`).
* **Checkpoints**: `last.pt` (with optimizer state, for `--resume`) and `best.pt` (lowest
  validation loss, which is effectively early stopping; `--patience` stops the run).
  Checkpoints hold only tensors and plain data and load with `weights_only=True`, which
  is safer than v1's pickled objects.
* **Metrics**: every run logs `log.csv`. The final `result.json` evaluates *all* of the
  validation text and reports loss, perplexity and **bits per character**. Bits per
  character is the fair way to compare a character model with a BPE model.
* **Presets**: `--preset cpu-tiny | cpu-small | gpu-medium | gpt-small`.

## Generation (`llm/generate.py`)

* **KV cache**: each new token runs one position through the network instead of the
  whole context. When the window fills, the cache is rebuilt from the most recent half.
* **Sampling**: temperature (0 = greedy), top-k, **top-p (nucleus)** and a **repetition
  penalty**.
* **Streaming** output, token by token, that waits for multi-byte characters to be
  complete before printing.
* **Chat mode** (`--chat`): wraps turns in the chat template, stops at `<|endoftext|>`,
  and keeps the conversation history within the context window.

## Finetuning (`llm/finetune.py`)

Instruction tuning on `{"prompt", "response"}` JSON lines. The loss covers **only the
response tokens** (prompt targets are −1, which `cross_entropy` ignores). The model
therefore learns to answer questions rather than to write them, and it learns to end
a turn with `<|endoftext|>`. `data/oz_sft.jsonl` has 50 Q&A pairs about the book as a
demo.

## Tests (`tests/`, run in CI)

* The tokenizer round-trips any Unicode text and handles special tokens.
* The model is causal: changing a future token never changes past logits.
* KV-cache decoding produces the same logits as a full forward pass, and greedy decoding
  gives the same output with and without the cache.
* RoPE preserves vector norms and depends only on relative position.
* The ignore-index masking is correct, weights are tied, and the model can overfit a
  single batch.
* An end-to-end run covers prepare → train → resume → evaluate → finetune → generate.

## Scaling up

On a GPU with OpenWebText:

```bash
python scripts/data_extract.py --src openwebtext --out_dir data            # text files
python -m llm.prepare --input data/output_train.txt --val_input data/output_val.txt \
    --out_dir data/owt_bpe --vocab_size 16384 --tokenizer_sample_mb 50 --workers 8
python -m llm.train --data_dir data/owt_bpe --out_dir runs/owt --preset gpt-small \
    --batch_size 16 --grad_accum 8 --max_iters 50000 --lr 6e-4 --warmup_iters 2000 --compile
```

With 16 × 8 × 1024 ≈ 131k tokens per step, 50k steps is about 6.5B tokens.
Chinchilla-optimal for ~100M parameters is about 2B tokens, so ~15k steps already get
most of the benefit.
