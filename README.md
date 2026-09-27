# LLM from scratch

A GPT language model built from first principles in Python and PyTorch. It exists in two
versions, and v2 has been improved over two rounds:

* **v1 (`notebooks/`, `scripts/`)** follows the course step by step. It starts with a
  bigram model trained on a Wizard of Oz book and ends with a character-level
  transformer that trains on OpenWebText and completes prompts.
* **v2 (`llm/`)** is the improved model, built with what modern LLMs use:
  - a byte-level BPE tokenizer trained from scratch,
  - RoPE, RMSNorm, SwiGLU, grouped-query attention, fused attention and weight tying,
  - warm-up plus cosine learning rate, gradient clipping and accumulation, mixed
    precision and `torch.compile`,
  - memory-mapped token datasets, safe checkpoints and a KV cache,
  - top-p sampling, instruction finetuning and a chat mode,
  - a test suite that runs in CI.

  On the same model size and number of steps it reaches **15% lower bits per character**
  than v1 (2.07 vs 2.43), and the tuned version reaches **2.03**. See [v2](#v2-the-improved-model).
* **Round 3** adds:
  - 20× more training data (18 public-domain books);
  - the Muon optimizer, EMA, multi-GPU (DDP) and hyperparameter sweeps;
  - QK-norm, mixture of experts and other architecture options;
  - a **retrieval-augmented chat model** that answers questions it has never seen;
  - int8 quantization, a web UI, and export to Hugging Face / llama.cpp / Ollama;
  - a ready-to-run GPU pipeline for web-scale data.

  The shipped base model reaches **1.415 bits/char**, and the chat model answers
  **58%** of held-out questions, against 19% before. See [Round 3](#round-3-more-data-rag-chat-export).

Everything is written by hand: the tokenizers, data loaders, attention, transformer
blocks, training loops, checkpointing and sampling. Only `torch` is used for tensors,
autograd and the optimizer.

```
Dorothy said ─► [char tokenizer] ─► [token + position embeddings]
                                  ─► [ N × (LayerNorm → multi-head causal self-attention → +residual
                                            LayerNorm → feed-forward             → +residual) ]
                                  ─► LayerNorm ─► Linear ─► softmax ─► sample next char ─► repeat
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. train a small GPT on the Wizard of Oz text (~6 min on a laptop CPU)
python scripts/training.py --data wizard --batch_size 32 --block_size 64 \
    --n_embd 128 --n_head 4 --n_layer 4 --max_iters 3000

# 2. chat with it
python scripts/chatbot.py --model_path model-01.pkl
```

For the full dataset, see [Training on OpenWebText](#training-on-openwebtext).

To chat with the best shipped model in the browser (it needs the corpus for retrieval,
about a 1-minute download):

```bash
python data/download_corpus.py && python -m llm.retrieval build
pip install gradio && python -m llm.webui --ckpt models/corpus-chat.pt --rag   # http://127.0.0.1:7860
```

## Repository layout

```
├── data/
│   ├── download_wizard_of_oz.py     re-download the corpus
│   └── wizard_of_oz.txt             ~232k characters, public domain
├── notebooks/                       the step-by-step build, all executed with outputs
│   ├── 01_data_and_tokenizer.ipynb
│   ├── 02_pytorch_basics.ipynb
│   ├── 03_bigram.ipynb
│   ├── 04_normalization_and_activations.ipynb
│   ├── 05_self_attention.ipynb
│   ├── 06_gpt_v1.ipynb
│   ├── 07_gpt_openwebtext.ipynb
│   └── 08_gpt_v2.ipynb              tour of v2: BPE, RoPE, checkpoints, KV cache, chat
├── scripts/                         v1
│   ├── gpt.py                       model + tokenizer (shared module)
│   ├── data_extract.py              OpenWebText .xz → train/val/vocab
│   ├── training.py                  CLI training / resume / pickle
│   └── chatbot.py                   prompt → completion
├── llm/                             v2
│   ├── tokenizer.py                 byte-level BPE + char tokenizers
│   ├── model.py                     GPT: RoPE, RMSNorm, SwiGLU, GQA, SDPA, KV cache
│   ├── data.py, prepare.py          text → uint16 token files, memmap batches
│   ├── train.py                     pretraining loop
│   ├── finetune.py                  instruction finetuning (prompt-masked loss)
│   ├── evaluate.py                  full-split loss / perplexity / bits per char
│   ├── generate.py                  streaming sampling + chat
│   └── checkpoint.py, utils.py
│   ├── optim.py                     Muon optimizer
│   ├── retrieval.py                 BM25 search for retrieval-augmented chat
│   ├── dpo.py                       Direct Preference Optimization
│   ├── chat_eval.py                 held-out question answering score
│   ├── webui.py                     Gradio chat in the browser
│   └── export.py                    Hugging Face (Llama/Qwen3) and GGUF export
├── models/                          shipped checkpoints
│   ├── oz-base.pt, oz-chat.pt       round 2: one book, 0.87M params (3.5 MB)
│   ├── corpus-base.pt               round 3: 18 books, 3.4M params (14 MB)
│   └── corpus-chat.pt               round 3: + reading-comprehension finetuning (use with --rag)
├── data/
│   ├── download_corpus.py           18 public-domain books → data/corpus/
│   ├── make_chat_data.py            grounded chat data + DPO pairs from the books
│   ├── download_hf.py               FineWeb-Edu / TinyStories / Dolly (GPU path)
│   ├── oz_sft.jsonl                 156 hand-written Q&A pairs
│   └── chat_eval.jsonl              26 held-out evaluation questions
├── experiments/                     ablations, sweeps, GPU pipeline, result tables
├── tests/                           pytest suite (also run by GitHub Actions)
├── docs/                            concept notes for each stage
└── requirements.txt
```

## Curriculum map

Every topic below is covered either in a notebook (code you can run) or a doc (notes).

| # | Topic | Where |
|---|---|---|
| 1 | Install libraries, pylzma build tools, Jupyter Notebook | [docs/00_setup.md](docs/00_setup.md), [requirements.txt](requirements.txt) |
| 2 | Download Wizard of Oz | [data/download_wizard_of_oz.py](data/download_wizard_of_oz.py) |
| 3 | Experimenting with the text file, character-level tokenizer | [01_data_and_tokenizer](notebooks/01_data_and_tokenizer.ipynb) |
| 4 | Types of tokenizers | [docs/01_tokenizers.md](docs/01_tokenizers.md), notebook 01 |
| 5 | Tensors instead of arrays, linear algebra heads-up | notebook 01, [docs/02](docs/02_linear_algebra_and_pytorch.md) |
| 6 | Train and validation splits, premise of the bigram model | notebook 01 |
| 7 | Inputs and targets (concept and implementation), batch size, `get_batch` | notebook 01 |
| 8 | Switching from CPU to CUDA | notebooks 01 and 02, `get_device()` in `gpt.py` |
| 9 | PyTorch overview, CPU vs GPU performance, more PyTorch functions | [02_pytorch_basics](notebooks/02_pytorch_basics.ipynb) |
| 10 | Embedding vectors and implementation | notebook 02 |
| 11 | Dot product and matrix multiplication, matmul, int vs float | notebook 02 |
| 12 | Recap and `get_batch`, `nn.Module` subclass | [03_bigram](notebooks/03_bigram.ipynb) |
| 13 | Gradient descent, logits and reshaping, logits dimensionality | notebook 03, [docs/03](docs/03_training_and_optimizers.md) |
| 14 | Generate function and giving the model context | notebook 03 |
| 15 | Training loop, optimizer, `zero_grad` | notebook 03 |
| 16 | Optimizers overview and applications | [docs/03](docs/03_training_and_optimizers.md), notebook 03 |
| 17 | Loss reporting, train vs eval mode | notebook 03 (`estimate_loss`) |
| 18 | Normalization overview; ReLU, Sigmoid and Tanh activations | [04_normalization_and_activations](notebooks/04_normalization_and_activations.ipynb) |
| 19 | Transformer and self-attention, Transformer architecture | [docs/04_transformer_and_gpt.md](docs/04_transformer_and_gpt.md) |
| 20 | Building a GPT rather than a full Transformer, GPT architecture | docs/04 |
| 21 | Self-attention deep dive, why we scale by 1/√dₖ | [05_self_attention](notebooks/05_self_attention.ipynb) |
| 22 | Switching to a MacBook (the `mps` backend) | notebook 06, docs/00 |
| 23 | Positional encoding, `GPTLanguageModel` init and forward pass | [06_gpt_v1](notebooks/06_gpt_v1.ipynb) |
| 24 | Standard deviation for model parameters (std 0.02 init) | notebook 06, docs/04 |
| 25 | Transformer blocks, feed-forward network | notebook 06 |
| 26 | Multi-head attention, dot-product attention | notebook 06 |
| 27 | `nn.Sequential` vs `nn.ModuleList` | notebook 06, docs/04 |
| 28 | Hyperparameters, fixing errors, beginning training | notebook 06 |
| 29 | OpenWebText download and the Survey of LLMs paper | [docs/05_openwebtext.md](docs/05_openwebtext.md) |
| 30 | How the dataloader has to change, extracting with WinRAR | docs/05 |
| 31 | Python data extractor, adjusting train/val splits | [scripts/data_extract.py](scripts/data_extract.py) |
| 32 | Adding the dataloader, training on OpenWebText | [07_gpt_openwebtext](notebooks/07_gpt_openwebtext.ipynb), `training.py` |
| 33 | Model loading/saving, pickling | notebook 07, `training.py`, [docs/06](docs/06_scripts_and_cli.md) |
| 34 | Fixing errors and GPU memory in Task Manager | docs/06 |
| 35 | Command-line argument parsing, porting code to a script | [scripts/training.py](scripts/training.py), docs/06 |
| 36 | Prompt → completion feature | [scripts/chatbot.py](scripts/chatbot.py) |
| 37 | `nn.Module` inheritance, generation cropping | `scripts/gpt.py`, docs/06 |
| 38 | Pretraining vs finetuning | [docs/07_pretraining_vs_finetuning.md](docs/07_pretraining_vs_finetuning.md) |
| 39 | R&D pointers | [docs/08_rnd_pointers.md](docs/08_rnd_pointers.md) |
| 40 | v2: applying the R&D pointers | [llm/](llm), [08_gpt_v2](notebooks/08_gpt_v2.ipynb), [docs/09_v2_improvements.md](docs/09_v2_improvements.md) |
| 41 | Round 3: data, Muon, RAG chat, DPO, export, GPU path | [docs/10_round3_upgrades.md](docs/10_round3_upgrades.md), [experiments/results_round3.md](experiments/results_round3.md) |

## Results (v1)

These results come from runs on a 4-core CPU with the Wizard of Oz text, as saved in the
notebooks:

| model | params | steps | train loss | val loss |
|---|---|---|---|---|
| random guess (ln 80) | – | – | 4.38 | 4.38 |
| bigram (notebook 03) | 6.4k | 10,000 | 2.42 | 2.47 |
| GPT: 4 layers, 4 heads, n_embd 128, block 64 (notebook 06) | 0.82M | 3,000 | 1.33 | 1.56 |

Sample from the GPT after about 6 minutes of CPU training:

```
"I've to you othese some of this lighousily us," cymed a dear.
... the Prince piglets of the on and the adven buggy. Wizard to his was etreed ...
the Wizard ustaned Dorothy, ruled apped the tup inny, and I as the wood ...
```

It has learned words, names from the book, dialogue punctuation and paragraph structure.
A bigger model on a GPU gets much further. Try
`--batch_size 64 --block_size 256 --n_embd 384 --n_head 8 --n_layer 8 --max_iters 5000`.

## v2: the improved model

### Use it

```bash
# chat with the shipped model (no training needed)
python -m llm.generate --ckpt models/oz-chat.pt --chat
python -m llm.generate --ckpt models/oz-base.pt --prompt "Dorothy looked at the Wizard"

# reproduce it: tokenize → pretrain → finetune → evaluate  (~15 min on a laptop CPU)
python -m llm.prepare  --input data/wizard_of_oz.txt --out_dir data/oz_bpe512 --vocab_size 512
python -m llm.train    --data_dir data/oz_bpe512 --out_dir runs/oz --preset cpu-tiny \
                       --dropout 0.3 --weight_decay 0.5 --max_iters 1500 --eval_interval 100
python -m llm.finetune --ckpt runs/oz/best.pt --data data/oz_sft.jsonl --out_dir runs/oz_chat \
                       --epochs 25 --dropout 0.1 --lr 5e-4
python -m llm.evaluate --ckpt runs/oz/best.pt
python -m llm.generate --ckpt runs/oz_chat/last.pt --chat

python -m pytest -q    # full test suite
```

What changed and why is explained in [docs/09_v2_improvements.md](docs/09_v2_improvements.md).
Every change has a command-line switch (`--pos_emb`, `--norm`, `--mlp`, `--n_kv_head`,
`--no_tie`, `--tokenizer`, …), so each one can be tested on its own.

### Ablation: what each change buys

All runs use 4 layers, 4 heads, 128 dimensions and 2000 steps on a CPU, with the same
80/20 split as v1. The metric is **bits per character on the whole validation text**
(lower is better). It is the fair comparison between character and BPE models. Script:
[experiments/ablation.sh](experiments/ablation.sh).

| run | tokenizer | val bits/char | vs v1 |
|---|---|---|---|
| v1 baseline (v1 architecture and training recipe) | char | 2.431 | — |
| v1 architecture + v2 training recipe (warm-up/cosine lr, AdamW β₂ 0.99, wd 0.1, clipping, 128 context) | char | 2.113 | −13.1% |
| v2 architecture + v2 recipe | char | **2.070** | −14.9% |
| v2, BPE 512 / 1024 / 2048 | bpe | 2.124 / 2.130 / 2.133 | −12.6% |

Tuning on top of that ([experiments/results_final.md](experiments/results_final.md)):

| run | params | val bits/char |
|---|---|---|
| **v2, BPE 512, dropout 0.3, weight decay 0.5** (shipped as `models/oz-base.pt`) | 0.87M | **2.034** (−16.3% vs v1) |
| v2, char, 6 layers / 192 dims / GQA | 1.87M | 2.064 |
| v2, BPE 512, 6 layers / 192 dims / GQA, regularized | 2.46M | 2.075 |

What the experiments show:

* **The training recipe is the biggest single win**, and the modern architecture adds
  to it.
* **This corpus is small: one book, 230 KB.** BPE models see about 2–3× more text per
  step and memorize it within a few hundred steps: training loss keeps falling while
  validation loss rises. With strong regularization (dropout 0.3, weight decay 0.5), BPE
  becomes the best option. Larger models get worse here, because the data, not the
  model size, is what limits results. On OpenWebText the opposite holds: use BPE with a
  16k–32k vocabulary and the `gpu-medium` / `gpt-small` presets.
* The KV cache gives a **2.4× faster** generation even on this small model (notebook 08).

### Chat model: what to expect

After instruction finetuning, the model answers in full sentences and stops at the end
of its turn:

```
> Who is Jim?
Jim is the cab-horse. He is old and very thin, but he can talk once they reach the fairy countries.
> Can you tell me about the Braided Man?
The Braided Man lives in Pyramid Mountain. His hair and his beard are braided, and he makes Assorted Flutters and Rustles.
```

With about 1M parameters trained on a single book, it **recalls** answers it saw in
finetuning, including rephrased versions of those questions. It cannot reason about
genuinely new questions: ask "Where does Ozma live?" and you get a fluent but wrong
answer. Real assistant behaviour needs orders of magnitude more pretraining data and
parameters. The code here scales to that (see
[Training on OpenWebText](#training-on-openwebtext) and docs/09).

## Round 3: more data, RAG chat, export

The round-2 experiments showed that data, not code, was holding the model back. Round 3
fixes that first, then adds the rest of the improvements list. Details are in
[docs/10_round3_upgrades.md](docs/10_round3_upgrades.md) and all the numbers are in
[experiments/results_round3.md](experiments/results_round3.md).

### Use it

```bash
python data/download_corpus.py                       # 18 books, 4.8M characters
python -m llm.retrieval build                        # BM25 index for RAG

python -m llm.generate --ckpt models/corpus-base.pt --prompt "Dorothy looked at the Scarecrow and said"
python -m llm.generate --ckpt models/corpus-chat.pt --chat --rag      # ask anything about the books
python -m llm.webui    --ckpt models/corpus-chat.pt --rag             # same, in the browser
python -m llm.chat_eval --ckpt models/corpus-chat.pt --rag            # 26 held-out questions

python -m llm.export --ckpt models/corpus-chat.pt --format gguf --out export/corpus-chat.gguf
cd export && ollama create oz-chat -f Modelfile && ollama run oz-chat  # or llama.cpp / LM Studio
```

Reproduce the models (about 2.5 hours on a 4-core CPU):

```bash
python -m llm.prepare --input data/corpus/train.txt --val_input data/corpus/val.txt \
    --out_dir data/corpus_bpe4k --vocab_size 4096 --workers 4
python -m llm.train --data_dir data/corpus_bpe4k --out_dir runs/base --n_layer 6 --n_head 6 --n_embd 192 \
    --block_size 256 --batch_size 24 --max_iters 4000 --eval_interval 250 --lr 1e-3 --dropout 0.1 \
    --optimizer muon --qk_norm --patience 3
python data/make_chat_data.py
python -m llm.finetune --ckpt runs/base/best.pt --data data/chat/sft_train.jsonl --out_dir runs/chat \
    --epochs 6 --batch_size 16 --lr 3e-4 --dropout 0.1 --val_fraction 0.05
```

### What each change bought (4 layers / 128 dims, 2000 steps, 18-book corpus)

| change | val bits/char |
|---|---|
| base (v2 recipe) | 1.593 |
| mixture of experts, 4 experts / top-2 | **1.556** |
| Muon optimizer | 1.576 |
| QK-norm + logit soft-cap | 1.588 |
| value residual + U-Net skips / EMA weights | 1.593 / 1.593 (no change) |
| grouped-query attention, 2 KV heads | 1.601 |
| BPE-dropout | 1.716 (worse: with 20× data it only costs capacity) |

**Shipped `corpus-base.pt`:** 6 layers, 192 dimensions, 256-token context, Muon and
QK-norm, trained for 4000 steps (~2 h on CPU). It reaches **1.415 bits/char**. It uses
only options that can be exported (MoE and soft-cap can't be expressed in the Llama/Qwen3
formats). Sample:

```
Dorothy looked at the Scarecrow and said:
"There's a mistake, my dear."
"I don't know," replied the Scarecrow. "I'm afraid of that!"
"What is it?" asked the Scarecrow.
"Don't know," answered the Scarecrow, calmly. "But we must take place for
somewhere else. That is why we're lost."
"And where did you happen to be here?" inquired the Tin Woodman, wonderingly.
```

### A chat model that answers new questions

A few-million-parameter model can't memorise a library. So `corpus-chat.pt` was taught
**reading comprehension** instead:

1. BM25 retrieval finds the best passage.
2. The prompt becomes `Context: … Question: …`.
3. The model was finetuned on about 6,000 synthetic examples to answer from the passage.

On 26 held-out questions that appear nowhere in its training data:

| model | accuracy |
|---|---|
| round-2 `oz-chat.pt` (memorised Q&A) | 19% |
| `corpus-chat.pt` without retrieval | 12% |
| **`corpus-chat.pt` with retrieval** | **58%** (the retrieval ceiling is 85%) |
| + DPO on synthetic preference pairs | 46% (the pairs were too easy; see the results file) |

```
> What is the Love Magnet?
All I want is to have people love me; and as long as I own the Love Magnet everyone I meet is sure to love me dearly.
> Who is Cap'n Bill?
"That might o' been, Trot, that might o' been," answered Cap'n Bill.
```

It answers by quoting the most relevant sentence of the passage. Often that is the
answer; sometimes it is only related (as for Cap'n Bill), and when retrieval picks the
wrong passage the answer is wrong. Generating a free-form answer needs a larger model
trained on real instruction data, which is what the GPU pipeline below does.

### Also new

* **Export:**
  - `python -m llm.export --format hf` writes a model that loads in 🤗 transformers as
    Llama (or as Qwen3 with QK-norm), with identical logits.
  - `--format gguf` writes a file for llama.cpp, Ollama and LM Studio, plus an Ollama
    Modelfile. Both paths are covered by tests that run the exported model and compare
    its output.
* **Generation options:** `--int8` (about 3× smaller), `--num_samples N`,
  `--context 512` (RoPE NTK scaling; costs a little quality without extra training), and
  `--chat --rag`.
* **Training:**
  - `torchrun --nproc_per_node N -m llm.train …` for multi-GPU;
  - `--optimizer muon`, `--ema`;
  - `--qk_norm --logit_softcap --value_residual --unet_skips --n_experts`;
  - `experiments/sweep.py` for grid and random hyperparameter search.
* **Tokenizer:** the exact GPT-2 pre-tokenizer (Unicode-aware) and BPE-dropout.
* **Tests:** 70 tests, including a real 2-process DDP run and exported models checked
  in transformers and llama.cpp.

### GPU path

`experiments/gpu_pipeline.sh` is ready to run on a CUDA machine:

1. Download about 2.5B tokens of FineWeb-Edu and train a 32k BPE tokenizer.
2. Pretrain a GPT-2-small-shaped model with Muon, EMA, QK-norm, bf16 and
   `torch.compile`, optionally on several GPUs.
3. Finetune on Dolly-15k plus the grounded Oz data, then run DPO.
4. Evaluate and export to GGUF and Hugging Face.

Expect roughly a day on one RTX 4090, or 2–3 hours on 8×A100. It hasn't been run here
because no GPU was available.

## Training on OpenWebText

1. Download `openwebtext.tar.xz` from <https://skylion007.github.io/OpenWebTextCorpus/>
   and unpack the outer archive (WinRAR/7-Zip, or `tar -xf openwebtext.tar.xz`).
2. Extract it into text files and build the vocabulary:
   ```bash
   python scripts/data_extract.py --src openwebtext --out_dir data            # all of it
   python scripts/data_extract.py --src openwebtext --out_dir data --max_files 500  # a sample
   ```
3. Train on a GPU. The loader memory-maps the files, so RAM use stays small:
   ```bash
   python scripts/training.py --data openwebtext --batch_size 64 --block_size 128 \
       --n_embd 384 --n_head 8 --n_layer 8 --max_iters 20000 --model_path model-01.pkl
   ```
4. Continue training later with `--resume`, which keeps the architecture stored in the
   pickle:
   ```bash
   python scripts/training.py --data openwebtext --resume --max_iters 10000
   ```
5. Chat with the model:
   ```bash
   python scripts/chatbot.py --max_new_tokens 200 --temperature 0.8 --top_k 20
   ```

If you run out of GPU memory, lower `--batch_size` first, then `--block_size`.

The v2 pipeline for the same data is faster and gives much better results:

```bash
python -m llm.prepare --input data/output_train.txt --val_input data/output_val.txt \
    --out_dir data/owt_bpe --vocab_size 16384 --tokenizer_sample_mb 50 --workers 8
python -m llm.train --data_dir data/owt_bpe --out_dir runs/owt --preset gpt-small \
    --batch_size 16 --grad_accum 8 --max_iters 20000 --lr 6e-4 --warmup_iters 2000 --compile
```

## Script reference

`python scripts/training.py --help`

| flag | default | meaning |
|---|---|---|
| `--data` | `wizard` | `wizard` (in memory) or `openwebtext` (memory-mapped) |
| `--batch_size` | 32 | sequences per step |
| `--block_size` | 128 | context length |
| `--n_embd` / `--n_head` / `--n_layer` | 384 / 8 / 8 | model size |
| `--dropout` | 0.2 | dropout rate |
| `--learning_rate` | 3e-4 | AdamW learning rate |
| `--max_iters`, `--eval_interval`, `--eval_iters` | 3000, 500, 100 | training schedule |
| `--model_path` | `model-01.pkl` | where to save the pickle, or load it with `--resume` |
| `--device` | auto | `cuda`, `mps` or `cpu` |

`python scripts/chatbot.py --help` takes `--model_path`, `--max_new_tokens`,
`--temperature`, `--top_k`, `--prompt` (answer one prompt and exit) and `--device`.

## Notes

* v1 checkpoints are pickled as `{'model', 'tokenizer'}`, so only load pickles you trust.
  v2 checkpoints hold only tensors and plain data and load with `weights_only=True`.
* The blocks use pre-norm (LayerNorm before attention and before the feed-forward
  layer), as in GPT-2. Post-norm, as in the original Transformer, is described in
  docs/04.
* The committed corpus is *Dorothy and the Wizard in Oz* (L. Frank Baum, 1908, public
  domain). Any plain-text file works, because the vocabulary is built from the data.

## Credits

The syllabus follows freeCodeCamp's *Create a Large Language Model from Scratch with
Python* course by Elliot Arledge. The architecture follows Andrej Karpathy's nanoGPT
lectures and the papers listed in [docs/08_rnd_pointers.md](docs/08_rnd_pointers.md).
