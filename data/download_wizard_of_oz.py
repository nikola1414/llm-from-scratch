"""Download an L. Frank Baum Oz book (public domain) as plain text -> data/wizard_of_oz.txt.

Project Gutenberg's "The Wonderful Wizard of Oz" (#55) is tried first; the fallback
mirror serves "Dorothy and the Wizard in Oz" (1908), which is the copy committed here.
Any plain-text book works — the model is character-level and builds its vocab from it.

Usage:  python data/download_wizard_of_oz.py
"""
import os
import re
import urllib.request

URLS = [
    "https://www.gutenberg.org/cache/epub/55/pg55.txt",
    "https://www.gutenberg.org/files/55/55-0.txt",
    "https://raw.githubusercontent.com/Infatoshi/fcc-intro-to-llms/main/wizard_of_oz.txt",
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wizard_of_oz.txt")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only the book itself (drop the Project Gutenberg header/licence)."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start() - (start.end() if start else 0)]
    return text.strip() + "\n"


def main():
    for url in URLS:
        try:
            print(f"trying {url}")
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read().decode("utf-8-sig")
            text = strip_gutenberg_boilerplate(raw).replace("\r\n", "\n")
            with open(OUT, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"saved {len(text):,} characters to {OUT}")
            return
        except Exception as e:  # try the next mirror
            print(f"  failed: {e}")
    raise SystemExit("could not download the book from any mirror")


if __name__ == "__main__":
    main()
