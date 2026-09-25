# 00 — Setup: libraries, build tools and Jupyter

## 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Registering the venv as a Jupyter kernel lets notebooks use exactly these packages:

```bash
pip install ipykernel
python -m ipykernel install --user --name=llm-from-scratch --display-name "LLM from scratch"
```

## 2. Install libraries

```bash
pip install -r requirements.txt
```

For an NVIDIA GPU install the CUDA build of PyTorch instead (pick the command for
your CUDA version at <https://pytorch.org/get-started/locally/>), e.g.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check it:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

On Apple Silicon, PyTorch ships with the `mps` backend (`torch.backends.mps.is_available()`).

## 3. pylzma and build tools

OpenWebText is distributed as thousands of `.xz` files. Python's standard library
already contains the `lzma` module, which is what `scripts/data_extract.py` uses, so
**no extra install is needed**.

If you want the third-party `pylzma` package anyway, it is a C extension and must be
compiled:

* **Windows** – install "Visual Studio Build Tools" with the *Desktop development with
  C++* workload, then `pip install pylzma`.
* **Linux** – `sudo apt install build-essential python3-dev`, then `pip install pylzma`.
* **macOS** – `xcode-select --install`, then `pip install pylzma`.

## 4. Jupyter Notebook

```bash
pip install jupyter
jupyter notebook          # opens the browser; open notebooks/ and pick the kernel
```

Notebooks are ideal for experimenting cell-by-cell; once the code is stable we port it
to plain scripts in `scripts/` (see docs/06_scripts_and_cli.md).
