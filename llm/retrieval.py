"""Retrieval-augmented generation (RAG): a small BM25 search engine over text chunks.

A small model cannot memorise every fact in its training books, and it cannot know facts
it never saw. Retrieval moves the knowledge into the prompt: find the passages most
relevant to the question, put them in front of it, and let the model read the answer
from the context.

    python -m llm.retrieval build --books data/corpus/books --out data/corpus/index.json
    python -m llm.retrieval search --index data/corpus/index.json "Where does Ozma live?"
"""
import argparse
import glob
import json
import math
import os
import re
from collections import Counter

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its
this that these those i you he she we they me him her us them my your his our their what which who whom whose
where when why how do does did done have has had not no so than then there here into out up down about over
tell know please can could would should will shall may might must describe""".split())


def words(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS]


def chunk_text(text, max_chars=500, min_chars=120):
    """Split on blank lines, then pack paragraphs into chunks of at most `max_chars`
    (longer paragraphs are split at sentence boundaries, never before `min_chars`).
    Fragments shorter than 20 characters (page numbers, "* * *") are dropped."""
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        while len(para) > max_chars:
            cut = max(para.rfind(". ", 0, max_chars), para.rfind('" ', 0, max_chars))
            cut = cut + 1 if cut > min_chars else max_chars
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        pieces.append(para)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) >= 20]


class BM25:
    """Okapi BM25: score(q, d) = Σ idf(w) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))."""

    def __init__(self, chunks, sources=None, k1=1.5, b=0.75):
        self.chunks, self.sources = chunks, sources or [""] * len(chunks)
        self.k1, self.b = k1, b
        self.tfs = [Counter(words(c)) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / max(1, len(self.lens))
        df = Counter(w for tf in self.tfs for w in tf)
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.postings = {}
        for i, tf in enumerate(self.tfs):
            for w in tf:
                self.postings.setdefault(w, []).append(i)

    def search(self, query, k=3):
        scores = Counter()
        for w in set(words(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in self.postings[w]:
                tf = self.tfs[i][w]
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
        return [(self.chunks[i], s, self.sources[i]) for i, s in scores.most_common(k)]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "sources": self.sources}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["chunks"], d.get("sources"))

    @classmethod
    def from_books(cls, paths, max_chars=500):
        chunks, sources = [], []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for c in chunk_text(f.read(), max_chars):
                    chunks.append(c)
                    sources.append(os.path.basename(path))
        return cls(chunks, sources)


def rag_prompt(question, passages):
    """The user turn used for retrieval-augmented chat (same format in finetuning)."""
    context = "\n\n".join(passages)
    return f"Context: {context}\n\nQuestion: {question}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--books", default="data/corpus/books", help="folder of .txt files")
    b.add_argument("--out", default="data/corpus/index.json")
    b.add_argument("--max_chars", type=int, default=500)
    s = sub.add_parser("search")
    s.add_argument("--index", default="data/corpus/index.json")
    s.add_argument("-k", type=int, default=3)
    s.add_argument("query")
    args = p.parse_args()
    if args.cmd == "build":
        index = BM25.from_books(sorted(glob.glob(os.path.join(args.books, "*.txt"))), args.max_chars)
        index.save(args.out)
        print(f"{len(index.chunks)} chunks -> {args.out}")
    else:
        for chunk, score, src in BM25.load(args.index).search(args.query, args.k):
            print(f"[{score:.2f}] ({src}) {chunk}\n")


if __name__ == "__main__":
    main()
