# 10 — Round 3: more data, modern optimizers, RAG chat, export

v2 (docs/09) reached the limit of a single 230 KB book: bigger models just memorised it.
This round works on each limiting factor in turn. As before, every change is a flag, and
the numbers come from runs you can reproduce (`experiments/ablation_corpus.sh`).

## 1. Data: 18 books instead of 1 (`data/download_corpus.py`)

* 12 Oz books, *The Sea Fairies* and *American Fairy Tales* by L. Frank Baum, plus *Alice in
  Wonderland*, Bryant's *Stories to Tell to Children*, Burgess's *Buster Brown* stories and
  Edgeworth's *The Parent's Assistant*: 4.8M characters, 20× more than before.
* Fetched from Project Gutenberg, with the GITenberg / NLTK mirrors on GitHub as
  fallbacks. Licence headers are stripped.
* The last 5% of **every** book goes to validation, so validation text comes from the
  same distribution as training text.
* Books are separated by `<|endoftext|>`, which `llm.prepare` turns into the special
  token, so the model learns where documents begin and end.

## 2. Tokenizer

* **Exact GPT-2 pre-tokenizer** (Unicode-aware, via the `regex` package). This is the
  pattern llama.cpp calls `gpt-2`, so exported models tokenize identically there. The
  pattern name is stored in `tokenizer.json`; older checkpoints keep their `ascii` pattern.
* **BPE-dropout** (`llm.prepare --bpe_dropout 0.1 --dropout_copies 4`): during encoding,
  each merge is skipped with probability p. The training split is written several times
  with different random segmentations. The model sees many ways of spelling the same
  word, which works as a regulariser and makes it more robust to rare words. Validation
  is always tokenized deterministically.

## 3. Optimizer and training (`llm/optim.py`, `llm/train.py`)

* **Muon** (`--optimizer muon`): for every hidden weight matrix, the momentum is replaced
  by the nearest orthogonal matrix, computed with 5 Newton–Schulz iterations. Every
  direction of the matrix is then updated equally. Embeddings, the output layer and
  norm gains stay on AdamW. Muon is the optimizer behind the modded-nanoGPT speedrun
  records.
* **EMA of weights** (`--ema 0.998`): a slowly moving average of the parameters is kept
  and evaluated alongside the raw weights; whichever is better is saved. The average sits
  in a flatter part of the loss surface, and it costs nothing at inference time.
* **Distributed training**: `torchrun --nproc_per_node N -m llm.train …` wraps the model
  in DistributedDataParallel. Gradients are only all-reduced on the last
  gradient-accumulation step, and only rank 0 writes logs and checkpoints. The tests run
  a real 2-process job on CPU (gloo).
* **Hyperparameter sweeps** (`experiments/sweep.py`): grid or random search (log-uniform,
  uniform, int, choice), parallel and resumable, ranked by validation bits per character.

## 4. Architecture options (`llm/model.py`)

| flag | what | from |
|---|---|---|
| `--qk_norm` | RMSNorm on each head's queries and keys; attention logits stay bounded | OLMo 2, Gemma 3, Qwen 3 |
| `--logit_softcap 30` | `30·tanh(logits/30)` on the output | Gemma 2, modded-nanoGPT |
| `--value_residual` | every layer mixes in layer 0's values with a learned λ | ResFormer |
| `--unet_skips` | layer *n−1−i* gets a learned-weight skip from layer *i* | modded-nanoGPT |
| `--n_experts 4 --moe_top_k 2` | sparse mixture of experts with a Switch load-balancing loss | Switch, Mixtral |
| `--n_kv_head` | grouped-query attention | Llama 2/3 |
| `extend_context(n, 'ntk'/'linear')` | run at a longer context than trained by rescaling RoPE | NTK-aware / position interpolation |

## 5. Chat that answers new questions: RAG + reading comprehension + DPO

The v2 chat model memorised 156 answers. It could not answer anything else, because a
1M-parameter model cannot store a library of facts. The fix is to **stop asking it to
remember**:

1. **Retrieval** (`llm/retrieval.py`): a BM25 index over 15k passages of the books.
   The question's best passage is put in the prompt: `Context: … Question: …`.
2. **Reading-comprehension finetuning** (`data/make_chat_data.py`):
   - about 5,000 synthetic examples: a passage, a question about a character or place
     in it, and as answer the sentence of the passage that describes it;
   - "unanswerable" examples (unrelated passage → *The text doesn't say.*);
   - the hand-written Q&A pairs, grounded the same way.

   The model learns to *find and copy* the relevant sentence. That skill transfers to
   questions it has never seen.
3. **DPO** (`llm/dpo.py`): Direct Preference Optimization on about 5,000 pairs. The
   chosen answer is the grounded one; the rejected answer is a fluent answer about the
   wrong passage, or a degenerate repetition. A frozen copy of the SFT model is the
   reference, and an optional NLL term (`--sft_coef`, RPO) keeps the chosen likelihood up.
4. **Evaluation** (`llm/chat_eval.py`): 26 held-out questions, none of them in the
   training data, scored by expected keywords, with and without retrieval.

## 6. Using the model

* `python -m llm.generate … --num_samples 4`: batched sampling.
* `--int8`: dynamic int8 quantization of every Linear layer. The model gets about 3×
  smaller (3.5 → 1.2 MB). On a model this tiny it is not faster, because the matrices are
  too small for int8 kernels to pay off; the gain appears at around 100M parameters.
* `--context 512`: NTK-scaled RoPE to read longer prompts than the training context.
* `--chat --rag`: answer from retrieved passages.
* `python -m llm.webui --rag`: a Gradio chat in the browser that streams replies and
  shows the retrieved passage.
* `python -m llm.export --format hf|gguf`:
  - **Hugging Face**: the architecture maps exactly onto Llama (Qwen3 with QK-norm). The
    tests check that 🤗 transformers produces the same token ids and logits.
  - **GGUF**: for llama.cpp, Ollama and LM Studio, with a generated Ollama `Modelfile`.

## 7. GPU path

`experiments/gpu_pipeline.sh` runs the whole thing at web scale:

1. Download ~2.5B tokens of FineWeb-Edu and train a 32k BPE tokenizer.
2. Pretrain a GPT-2-small-shaped model (Muon, EMA, QK-norm, bf16, `torch.compile`,
   optionally multi-GPU).
3. Finetune on Dolly-15k plus the grounded Oz data, then run DPO.
4. Evaluate, and export to GGUF and Hugging Face format.

`data/download_hf.py` also fetches TinyStories, which gives fluent small models very
quickly.

## Results

The full tables are in `experiments/results_round3.md`. In short:

* **Data** was the biggest lever. The same 1.3M-parameter recipe goes from 2.03 bits/char
  on one book to 1.59 on 18 books, and the shipped 3.4M-parameter model reaches 1.415.
* **MoE** (−2.3%) and **Muon** (−1.1%) help. QK-norm with soft-cap helps a little. Value
  residual, U-Net skips and EMA make no difference at this scale. GQA costs a little
  quality in exchange for a smaller cache. BPE-dropout hurts once data is plentiful.
* **RAG chat** answers 58% of held-out questions, against 19% for memorised Q&A. The
  retrieval ceiling is 85%.
* **DPO** with easy synthetic negatives lowered the score to 46%. Preference data has to be
  hard, ideally the model's own mistakes, to teach anything.
* **Context extension** without extra training costs quality: 1.38 bits/char at the
  trained 256 tokens, 1.43 at 512 tokens with NTK scaling, 1.65 with linear scaling.
