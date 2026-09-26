# 06 — From notebook to scripts: CLI, pickling, and common errors

## Porting code to a script

Notebooks are great for exploring but awkward for long training runs (kernel restarts,
hidden state, no CLI). The model classes moved to `scripts/gpt.py`; training and chatting
are separate entry points:

| file | purpose |
|---|---|
| `scripts/gpt.py` | `GPTConfig`, `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPTLanguageModel`, `CharTokenizer` |
| `scripts/data_extract.py` | OpenWebText `.xz` → `output_train.txt` / `output_val.txt` / `vocab.txt` |
| `scripts/training.py` | train or resume, evaluate, pickle the model |
| `scripts/chatbot.py` | interactive prompt → completion |

## Command-line argument parsing

`argparse` replaces hard-coded hyperparameters:

```python
parser = argparse.ArgumentParser(description='This is a demonstration program')
parser.add_argument('-batch_size', type=int, required=True, help='Please provide a batch_size')
args = parser.parse_args()
print(f'batch size: {args.batch_size}')
```

`python scripts/training.py --help` lists every option (`--batch_size`, `--block_size`,
`--n_embd`, `--n_layer`, `--max_iters`, `--lr`, `--data`, `--resume`, …).

## Model loading/saving and pickling

`pickle.dump(obj, f)` serialises an arbitrary Python object — here
`{'model': model, 'tokenizer': tokenizer}` — so training can stop and resume
(`--resume`) and the chatbot can load the result. Notes:

* Unpickling needs the **class definitions importable** under the same module name
  (`gpt.GPTLanguageModel`); that is why both scripts put `scripts/` on `sys.path`.
* We move the model to the CPU before pickling so the file loads on machines without CUDA.
* Only unpickle files you trust — pickle can execute arbitrary code.
* The PyTorch-native alternative is `torch.save(model.state_dict(), 'model.pt')` +
  `model.load_state_dict(torch.load('model.pt'))`, which stores weights only.

## `nn.Module` inheritance

Every layer subclasses `nn.Module` and must call `super().__init__()` **before** assigning
sub-modules; otherwise PyTorch raises
`AttributeError: cannot assign module before Module.__init__() call`. Registering
sub-modules as attributes (or in `nn.ModuleList` / `nn.Sequential`) is what makes
`.parameters()`, `.to(device)`, `.train()`/`.eval()` and pickling see them. Call the module
(`model(x)`), not `model.forward(x)`, so hooks run.

## Generation cropping

The position table has `block_size` rows. Once prompt + generated tokens exceed that,
`position_embedding_table(torch.arange(T))` would index out of range
(`IndexError: index out of range in self`). `generate` therefore feeds only the last
`block_size` tokens: `index_cond = index[:, -block_size:]`.

## Prompt → completion

`chatbot.py` encodes the prompt, runs `generate`, decodes the whole sequence. Errors met
along the way and their fixes:

| error | cause | fix |
|---|---|---|
| `KeyError: 'é'` in `encode` | prompt char not in training vocab | `CharTokenizer.encode` skips unknown chars |
| `IndexError: index out of range in self` | context longer than `block_size` | crop the context in `generate` |
| `RuntimeError: Expected all tensors to be on the same device` | prompt tensor on CPU, model on GPU | create tensors with `device=device` |
| `IndexError: index -1 is out of bounds for dimension 1 with size 0` | empty prompt (context has no tokens) | fall back to a single token |
| `AttributeError: Can't get attribute 'GPTLanguageModel'` on unpickling | class not importable | import `gpt` before `pickle.load` |

## Fixing errors + GPU memory

* `CUDA out of memory` — activations scale with `batch_size × block_size × n_embd ×
  n_layer`, and attention with `block_size²`. Lower `batch_size`, then `block_size`.
* Watch **Task Manager → Performance → GPU → Dedicated GPU memory** (Windows) or
  `nvidia-smi` (Linux); `training.py` also prints `torch.cuda.max_memory_allocated()`.
* `mmap` "cannot mmap an empty file" — the extractor has not run or `--data_dir` is wrong.
* Very high initial loss (≫ ln(vocab)) — check the weight init (std 0.02) and that
  targets are shifted by one.
