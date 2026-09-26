# LLM from scratch

A character-level GPT built from first principles in Python and PyTorch. It starts with
a bigram model trained on a Wizard of Oz book and ends with a decoder-only transformer that
trains on OpenWebText and completes prompts in an interactive chatbot.

Everything is written by hand: the tokenizer, data loaders, attention heads, transformer
blocks, training loop, checkpointing and sampling. Only `torch` is used for tensors,
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
│   └── 07_gpt_openwebtext.ipynb
├── scripts/
│   ├── gpt.py                       model + tokenizer (shared module)
│   ├── data_extract.py              OpenWebText .xz → train/val/vocab
│   ├── training.py                  CLI training / resume / pickle
│   └── chatbot.py                   prompt → completion
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

## Results

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

* Checkpoints are pickled as `{'model', 'tokenizer'}`. Only load pickles you trust.
* The blocks use pre-norm (LayerNorm before attention and before the feed-forward
  layer), as in GPT-2. Post-norm, as in the original Transformer, is described in
  docs/04.
* The committed corpus is *Dorothy and the Wizard in Oz* (L. Frank Baum, 1908, public
  domain). Any plain-text file works, because the vocabulary is built from the data.

## Credits

The syllabus follows freeCodeCamp's *Create a Large Language Model from Scratch with
Python* course by Elliot Arledge. The architecture follows Andrej Karpathy's nanoGPT
lectures and the papers listed in [docs/08_rnd_pointers.md](docs/08_rnd_pointers.md).
