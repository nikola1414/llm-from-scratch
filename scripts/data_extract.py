"""Extract the OpenWebText corpus into train/val text files and a character vocabulary.

OpenWebText (https://skylion007.github.io/OpenWebTextCorpus/) downloads as
`openwebtext.tar.xz`. Unpack the outer archive once (WinRAR / 7-Zip on Windows,
`tar -xf openwebtext.tar.xz` elsewhere) to get a folder full of `urlsf_subset*.xz`
files — each is itself an xz-compressed tar of small .txt documents. This script
reads those inner archives directly with Python's built-in `lzma`/`tarfile`
modules, so nothing else needs unpacking.

Output (in --out_dir, default data/):
    output_train.txt   90 % of the archives
    output_val.txt     10 % of the archives
    vocab.txt          every distinct character, one file, used as the vocabulary

Usage:
    python scripts/data_extract.py --src openwebtext --out_dir data
    python scripts/data_extract.py --src openwebtext --max_files 200   # quick subset
"""
import argparse
import lzma
import os
import random
import tarfile

from tqdm import tqdm


def xz_files_in_dir(directory):
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".xz") and os.path.isfile(os.path.join(directory, f))
    )


def read_archive(path):
    """Yield the text of every document inside one .xz file.

    Handles both an xz-compressed tar of documents (the OpenWebText format) and a
    plain xz-compressed text file.
    """
    try:
        with tarfile.open(path, mode="r:xz") as tar:
            for member in tar:
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        yield f.read().decode("utf-8", errors="ignore")
    except tarfile.ReadError:
        with lzma.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            yield f.read()


def process_files(directory, files, output_file, vocab):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in tqdm(files, desc=os.path.basename(output_file)):
            for doc in read_archive(os.path.join(directory, filename)):
                outfile.write(doc)
                if not doc.endswith("\n"):
                    outfile.write("\n")
                vocab.update(doc)
    return vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="openwebtext", help="folder with the urlsf_subset*.xz files")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_files", type=int, default=None, help="only use this many archives")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    files = xz_files_in_dir(args.src)
    if not files:
        raise SystemExit(f"no .xz files found in {args.src!r}")
    random.Random(args.seed).shuffle(files)
    if args.max_files:
        files = files[: args.max_files]

    # split by archive so no document appears in both sets
    split_index = max(1, int(len(files) * (1 - args.val_fraction)))
    files_train = files[:split_index]
    files_val = files[split_index:] or files[-1:]

    os.makedirs(args.out_dir, exist_ok=True)
    vocab = set()
    vocab = process_files(args.src, files_train, os.path.join(args.out_dir, "output_train.txt"), vocab)
    vocab = process_files(args.src, files_val, os.path.join(args.out_dir, "output_val.txt"), vocab)

    with open(os.path.join(args.out_dir, "vocab.txt"), "w", encoding="utf-8") as vfile:
        vfile.write("".join(sorted(vocab)))
    print(f"train archives: {len(files_train)}, val archives: {len(files_val)}, vocab size: {len(vocab)}")


if __name__ == "__main__":
    main()
