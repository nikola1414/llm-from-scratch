# 01 — Tokenizers

A language model only sees integers. A **tokenizer** defines the mapping text ⇄ token ids
and therefore the size of the vocabulary (the number of rows in the embedding table and
the number of outputs of the final layer).

| Type | How it splits | Pros | Cons |
|---|---|---|---|
| **Character-level** (used here) | one token per character | tiny vocab, no out-of-vocabulary words, trivial to implement | long sequences, the model must learn spelling |
| **Word-level** | whitespace/punctuation | short sequences | enormous vocab, unknown words, bad at morphology |
| **Subword – BPE** (GPT-2/3/4, Llama) | repeatedly merge the most frequent pair of symbols | balanced vocab/sequence length, handles any text (byte-level) | needs a training step |
| **Subword – WordPiece** (BERT) | merges chosen by likelihood | as BPE | as BPE |
| **Subword – Unigram / SentencePiece** (T5, Llama) | prune a large vocab by likelihood; language agnostic | works on raw text with no pre-tokenization | as BPE |

Trade-off: a bigger vocabulary means fewer tokens per sentence (more text fits in the
context window) but a bigger embedding/output matrix and rarer tokens per id.

Our encoder/decoder:

```python
chars = sorted(set(text))
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join(int_to_string[i] for i in l)
```

To try a production tokenizer: `pip install tiktoken` then
`tiktoken.get_encoding("gpt2").encode("hello world")`.
