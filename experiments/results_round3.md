# Round-3 results (CPU, 4 cores)

## Pretraining on the 18-book corpus (bits per character, full validation text)

Ablation: 4 layers / 4 heads / 128 dims, BPE 4096, context 128, 2000 steps
(`experiments/ablation_corpus.sh`, full table in `results_ablation_corpus.md`).

| change vs base | val bits/char | Δ |
|---|---|---|
| base (v2 recipe, AdamW) | 1.593 | — |
| + mixture of experts (4 experts, top-2; 2.2× params, ~1.5× compute) | **1.556** | −2.3% |
| + Muon optimizer | 1.576 | −1.1% |
| + QK-norm + logit soft-cap 30 | 1.588 | −0.3% |
| + value residual + U-Net skips | 1.593 | 0 |
| + EMA weights (0.998) | 1.593 | 0 (the cosine schedule already anneals) |
| + grouped-query attention (2 KV heads) | 1.601 | +0.5% (fewer params; for inference memory) |
| + BPE-dropout 0.1 | 1.716 | +7.7% (regularisation not needed with 20× data) |

Final base model `models/corpus-base.pt`: 6 layers / 6 heads / 192 dims, context 256,
Muon + QK-norm, dropout 0.1, 4000 steps (~1.9 h): **1.415 bits/char** (perplexity 28.7 per
BPE token). For reference, the v2 model on the single book scored 2.03 and v1 2.43 on
that book's validation text.

## Context extension (RoPE), no extra training, first 40k validation tokens

| context | scaling | val bits/char |
|---|---|---|
| 256 (trained) | – | 1.376 |
| 512 | none (extrapolate) | 1.429 |
| 512 | NTK | 1.429 |
| 512 | linear | 1.646 |
| 1024 | NTK | 1.537 |

## Chat: 26 held-out questions, answer must contain an expected keyword

| model | no retrieval | with retrieval (RAG, k=1) |
|---|---|---|
| `oz-chat.pt` (round 2: memorised 156 Q&A) | 19.2% | 19.2% |
| reading-comprehension SFT, first data version (3 epochs) | 19.2% | 34.6% |
| … + 3 more epochs | – | 34.6% |
| fixed data (dialogue answers, fewer refusals), 3 epochs | – | 42.3% |
| **fixed data, 6 epochs → `models/corpus-chat.pt`** | 11.5% | **57.7%** |
| … same, k=2 passages | – | 42.3% |
| … + DPO (synthetic pairs) | – | 46.2% |

Retrieval alone puts a keyword-bearing passage first for 22/26 questions (85%), which
is the ceiling for this setup.

DPO did not help: the synthetic rejected answers (a sentence from another passage, or a
repetition loop) were too easy — preference accuracy hit 100% within 100 steps — so the
update mostly moved the model away from the SFT solution. On-policy negatives (the
model's own wrong answers) would be the next thing to try.
