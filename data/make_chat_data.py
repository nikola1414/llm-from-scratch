"""Generate retrieval-grounded chat data for finetuning (and preference pairs for DPO).

Instead of memorising a few hand-written answers, the model is taught *reading
comprehension*: each example gives a passage from the books plus a question about a
character or place in it; the answer is the sentence of the passage that talks about
it. A model trained this way can answer questions it has never seen, as long as
retrieval (llm/retrieval.py) finds the right passage.

Also included:
  * the hand-written Q&A pairs in data/oz_sft.jsonl, with a retrieved passage as context;
  * "unanswerable" examples whose passage is unrelated -> "The text doesn't say.";
  * DPO pairs (chosen = the grounded answer, rejected = a fluent answer about something
    else, or a degenerate repetition).

    python data/make_chat_data.py        # -> data/chat/{sft_train,sft_val,dpo_train,dpo_val}.jsonl
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.retrieval import BM25, chunk_text, rag_prompt  # noqa: E402

TEMPLATES = ["Who is {e}?", "Tell me about {e}.", "What does the story say about {e}?", "What do you know about {e}?",
             "Describe {e}.", "What happens with {e}?", "who is {e}", "Can you tell me about {e}?"]
NO_ANSWER = "The text doesn't say."
NOT_NAMES = set("""I A The And But Then When What Why How Where Who So Oh Yes No Now Well Do Did Is It He She They We
You My His Her Our Your Their This That There Here If In On At For With As Not All One Mr Mrs Miss Sir Madam God
Chapter CHAPTER Contents Illustration Aunt Uncle King Queen Princess Prince Good Great Little Old New Please Some
Of To By From Let Oz""".split())
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')


def find_entities(texts, min_count=25):
    """Capitalised words (and two-word names) that appear mid-sentence often enough."""
    single, double = Counter(), Counter()
    for text in texts:
        for m in re.finditer(r"(?<=[a-z,;] )([A-Z][a-z]+(?:-[A-Z][a-z]+)?)(?: ([A-Z][a-z]+))?", text):
            a, b = m.group(1), m.group(2)
            if a not in NOT_NAMES:
                single[a] += 1
            if b and b not in NOT_NAMES:
                double[f"{a} {b}"] += 1
    names = {n for n, c in double.items() if c >= min_count // 2}
    names |= {n for n, c in single.items() if c >= min_count}
    names = {n for n in names if "Gutenberg" not in n}
    return sorted(names, key=len, reverse=True)


def answer_sentence(chunk, entity):
    """The first narrative sentence about `entity`; if there is none, the first sentence
    of dialogue about it (retrieved passages often mention a character only in speech)."""
    sentences = [s.strip() for s in SENT.split(chunk)]
    about = [s for s in sentences if entity in s and 30 <= len(s) <= 220]
    narrative = [s for s in about if '"' not in s and "'" not in s[:1]]
    return (narrative or about or [None])[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--books", default="data/corpus/books")
    p.add_argument("--handwritten", default="data/oz_sft.jsonl")
    p.add_argument("--out_dir", default="data/chat")
    p.add_argument("--index_out", default="data/corpus/index.json")
    p.add_argument("--max_chars", type=int, default=400)
    p.add_argument("--n", type=int, default=6000, help="synthetic grounded examples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.books, "*.txt")))
    texts = [open(pth, encoding="utf-8").read() for pth in paths]
    index = BM25.from_books(paths, args.max_chars)
    index.save(args.index_out)
    entities = find_entities(texts)
    print(f"{len(index.chunks)} chunks, {len(entities)} entities, e.g. {entities[:12]}")

    candidates = []
    for ci, chunk in enumerate(index.chunks):
        for e in entities:
            if e in chunk:
                ans = answer_sentence(chunk, e)
                if ans:
                    candidates.append((ci, e, ans))
                break
    rng.shuffle(candidates)
    examples, pairs = [], []
    for ci, e, ans in candidates[: args.n]:
        q = rng.choice(TEMPLATES).format(e=e)
        prompt = rag_prompt(q, [index.chunks[ci]])
        examples.append({"prompt": prompt, "response": ans})
        # DPO: rejected = grounded-looking answer from a different passage, or a degenerate loop
        other = rng.choice(candidates)[2]
        rejected = other if rng.random() < 0.7 else " ".join([ans.split(",")[0]] * 4)
        if rejected != ans:
            pairs.append({"prompt": prompt, "chosen": ans, "rejected": rejected})
    # unanswerable: unrelated passage
    for _ in range(args.n // 25):
        ci, e, _ = rng.choice(candidates)
        other = rng.randrange(len(index.chunks))
        if e in index.chunks[other]:
            continue
        prompt = rag_prompt(rng.choice(TEMPLATES).format(e=e), [index.chunks[other]])
        examples.append({"prompt": prompt, "response": NO_ANSWER})
        pairs.append({"prompt": prompt, "chosen": NO_ANSWER, "rejected": rng.choice(candidates)[2]})
    # hand-written Q&A, grounded with the best passage
    with open(args.handwritten, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            passages = [c for c, _, _ in index.search(r["prompt"], 1)]
            examples.append({"prompt": rag_prompt(r["prompt"], passages), "response": r["response"]})

    rng.shuffle(examples)
    rng.shuffle(pairs)
    n_val = len(examples) // 20
    splits = {"sft_train": examples[n_val:], "sft_val": examples[:n_val],
              "dpo_train": pairs[len(pairs) // 20:], "dpo_val": pairs[: len(pairs) // 20]}
    for name, rows in splits.items():
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(rows)}")
    print("example:", json.dumps(examples[0], indent=1)[:700])


if __name__ == "__main__":
    main()
