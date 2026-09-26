# LLM from scratch

A GPT language model built from first principles in Python and PyTorch, in two versions:

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

For the full dataset, see [Training on OpenWebText](#training-on-openwebtext). To chat
with the shipped v2 model right away, run `python -m llm.generate --ckpt models/oz-chat.pt --chat`.

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
├── models/                          shipped v2 checkpoints (3.5 MB each)
│   ├── oz-base.pt                   pretrained on the book
│   └── oz-chat.pt                   + instruction finetuned on data/oz_sft.jsonl
├── data/oz_sft.jsonl                156 Q&A pairs about the book (with paraphrases)
├── experiments/                     ablation script + result tables
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

python -m pytest -q    # 31 tests
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
