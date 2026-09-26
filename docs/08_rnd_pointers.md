# 08 — R&D pointers: where to go next

Ideas roughly in order of effort, each maps to a spot in `scripts/gpt.py` / `training.py`.

## Efficiency

* **Fused attention** — replace the per-head loop with one batched projection and
  `F.scaled_dot_product_attention(q, k, v, is_causal=True)` (FlashAttention kernels).
* **Mixed precision** — `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` and
  `torch.cuda.amp.GradScaler` for fp16: ~2× faster, half the memory.
* **`torch.compile(model)`** — graph compilation for another speed-up.
* **Gradient accumulation** — emulate large batches on small GPUs.
* **Quantization** (int8/int4) for inference; **KV-cache** so generation doesn't recompute
  the whole context every step.
* **Distributed training** — DDP / FSDP across GPUs.

## Better modelling

* **Subword tokenizer** (BPE via `tiktoken` or `sentencepiece`) — more text per context.
* **Learning-rate schedule** — linear warm-up + cosine decay; **gradient clipping**
  (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`); weight decay only on
  matrices.
* **Weight tying** — share `token_embedding_table.weight` with `lm_head.weight`.
* **GELU / SwiGLU** feed-forward, **RMSNorm**, **RoPE** or **ALiBi** positions (Llama-style).
* **Grouped-query / multi-query attention**, sliding-window attention, Mixture of Experts.
* **Sampling** — temperature, top-k (implemented), top-p/nucleus, repetition penalty.

## Data and evaluation

* Cleaning and de-duplication (MinHash), quality filtering, mixing sources.
* Scaling laws (Kaplan et al. 2020; Chinchilla, Hoffmann et al. 2022: ~20 tokens per
  parameter).
* Evaluate with perplexity (`exp(val_loss)`) and benchmarks (HellaSwag, MMLU, …).

## Papers to read

* *Attention Is All You Need* — Vaswani et al., 2017
* *Language Models are Unsupervised Multitask Learners* (GPT-2) — Radford et al., 2019
* *Language Models are Few-Shot Learners* (GPT-3) — Brown et al., 2020
* *A Survey of Large Language Models* — Zhao et al., 2023
* *LLaMA: Open and Efficient Foundation Language Models* — Touvron et al., 2023
* *FlashAttention* — Dao et al., 2022
* *LoRA: Low-Rank Adaptation of Large Language Models* — Hu et al., 2021
* *QLoRA* — Dettmers et al., 2023
* *Training language models to follow instructions with human feedback* (InstructGPT) — Ouyang et al., 2022
