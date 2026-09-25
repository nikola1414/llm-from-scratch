# 05 — Scaling up the data: OpenWebText

## Download

[OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) is an open reproduction of
the WebText corpus used to train GPT-2: ~8 M documents / ~40 GB of text scraped from
Reddit-linked URLs. Download `openwebtext.tar.xz` (≈12 GB) from the site or from the
Hugging Face dataset `Skylion007/openwebtext`.

## Survey of LLMs paper

"A Survey of Large Language Models" (Zhao et al., 2023, arXiv:2303.18223) is a great map of
the field: pre-training corpora (web text, books, code, Wikipedia), data cleaning and
de-duplication, architectures (decoder-only vs encoder-decoder, normalization and
positional-encoding variants), scaling laws, instruction tuning, RLHF and evaluation. Its
data section explains why web corpora like OpenWebText are the backbone of pre-training.

## Extract the corpus (WinRAR / 7-Zip / tar)

The outer archive is a tar compressed with xz. Unpack it once:

* **Windows** — right click → *WinRAR → Extract here* (or 7-Zip → *Extract here* twice).
* **Linux / macOS** — `tar -xf openwebtext.tar.xz`

You get a folder `openwebtext/` with ~20 000 `urlsf_subset*.xz` files. Each is again an
xz-compressed tar of small `.txt` documents; we do **not** unpack them by hand.

## Python data extractor

```bash
python scripts/data_extract.py --src openwebtext --out_dir data            # everything
python scripts/data_extract.py --src openwebtext --out_dir data --max_files 500   # a sample
```

It streams every inner archive with the built-in `lzma`/`tarfile` modules and writes

* `data/output_train.txt` / `data/output_val.txt` — **90 / 10 split by archive** so no
  document is in both sets,
* `data/vocab.txt` — all distinct characters (our character-level vocabulary).

## How the dataloader / batch getter has to change

The Wizard of Oz fits in memory as one tensor (`torch.tensor(encode(text))`). A 40 GB text
file does not. Instead we **memory-map** the file and read a random slice each batch:

```python
def get_random_chunk(split):
    filename = 'data/output_train.txt' if split == 'train' else 'data/output_val.txt'
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start_pos = random.randint(0, mm.size() - block_size * batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
```

`mmap` lets the OS page in only the bytes we read, so memory use is independent of file
size. `errors='ignore'` drops half-cut multi-byte UTF-8 characters at the slice edges.
