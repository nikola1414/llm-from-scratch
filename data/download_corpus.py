"""Build a bigger public-domain training corpus: the Oz books plus other classic
children's literature (~4 MB, ~17x the single book used by v1).

Sources are Project Gutenberg texts, fetched from Gutenberg itself or from the GITenberg
and NLTK mirrors on GitHub. Licence headers/footers are stripped. For each book the last
`--val_fraction` of the text goes to the validation file, so train and val come from the
same distribution. Books are separated by `<|endoftext|>`.

    python data/download_corpus.py                       # -> data/corpus/{train,val}.txt
    python -m llm.prepare --input data/corpus/train.txt --val_input data/corpus/val.txt \
        --out_dir data/corpus_bpe4k --vocab_size 4096
"""
import argparse
import io
import os
import re
import urllib.request
import zipfile

GITENBERG = "https://raw.githubusercontent.com/GITenberg/{repo}/master/{file}"
GUTENBERG = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# (title, gutenberg id, GITenberg repo)
BAUM = [
    ("The Wonderful Wizard of Oz", 55, "The-Wonderful-Wizard-of-Oz_55"),
    ("The Marvelous Land of Oz", 54, "The-Marvelous-Land-of-Oz_54"),
    ("Ozma of Oz", 486, "Ozma-of-Oz_486"),
    ("Dorothy and the Wizard in Oz", 22566, "Dorothy-and-the-Wizard-in-Oz_22566"),
    ("The Road to Oz", 26624, "The-Road-to-Oz_26624"),
    ("The Emerald City of Oz", 517, "The-Emerald-City-of-Oz_517"),
    ("The Patchwork Girl of Oz", 955, "The-Patchwork-Girl-of-Oz_955"),
    ("Tik-Tok of Oz", 956, "Tik-Tok-of-Oz_956"),
    ("The Scarecrow of Oz", 957, "The-Scarecrow-of-Oz_957"),
    ("Rinkitink in Oz", 25581, "Rinkitink-in-Oz_25581"),
    ("The Lost Princess of Oz", 959, "The-Lost-Princess-of-Oz_959"),
    ("The Magic of Oz", 419, "The-Magic-of-Oz_419"),
    ("The Sea Fairies", 4358, "The-Sea-Fairies_4358"),
    ("American Fairy Tales", 4357, "American-Fairy-Tales_4357"),
]
NLTK_ZIP = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"
NLTK_BOOKS = ["carroll-alice.txt", "bryant-stories.txt", "burgess-busterbrown.txt", "edgeworth-parents.txt"]


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def strip_boilerplate(text):
    text = text.replace("\r\n", "\n")
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    if start and end and end.start() > start.end():
        text = text[start.end():end.start()]
    else:  # older headers
        s = re.search(r"\*END\*THE SMALL PRINT[^\n]*\n", text)
        e = re.search(r"End of (the )?Project Gutenberg", text, re.I)
        text = text[s.end() if s else 0: e.start() if e else len(text)]
    return re.sub(r"\n{4,}", "\n\n\n", text).strip() + "\n"


def get_baum(title, gid, repo):
    for url in (GUTENBERG.format(id=gid), GITENBERG.format(repo=repo, file=f"{gid}.txt")):
        try:
            return strip_boilerplate(fetch(url).decode("utf-8-sig", errors="replace"))
        except Exception:
            continue
    raise RuntimeError(f"could not download {title}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--no_nltk", action="store_true", help="only the Baum books")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out_dir, "books"), exist_ok=True)

    books = []
    for title, gid, repo in BAUM:
        text = get_baum(title, gid, repo)
        books.append((title, text))
        print(f"{len(text):>9,} chars  {title}")
    if not args.no_nltk:
        z = zipfile.ZipFile(io.BytesIO(fetch(NLTK_ZIP)))
        for name in NLTK_BOOKS:
            text = z.read(f"gutenberg/{name}").decode("latin-1").replace("\r\n", "\n").strip() + "\n"
            books.append((name, text))
            print(f"{len(text):>9,} chars  {name} (NLTK)")

    sep = "\n<|endoftext|>\n"
    train, val = [], []
    for title, text in books:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        with open(os.path.join(args.out_dir, "books", slug + ".txt"), "w", encoding="utf-8") as f:
            f.write(text)
        # cut at a paragraph boundary near the split point
        cut = text.rfind("\n\n", 0, int(len(text) * (1 - args.val_fraction))) + 2
        train.append(text[:cut])
        val.append(text[cut:])
    for name, parts in (("train", train), ("val", val)):
        with open(os.path.join(args.out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(sep.join(parts))
    total = sum(len(t) for _, t in books)
    print(f"{len(books)} books, {total:,} characters -> {args.out_dir}/train.txt, val.txt")


if __name__ == "__main__":
    main()
