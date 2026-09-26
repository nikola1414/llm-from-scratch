# 07 — Pretraining vs finetuning

## Pretraining (what this repo does)

* **Objective**: next-token prediction on a huge, unlabeled corpus (OpenWebText, books,
  code…). Loss = cross-entropy of the next token.
* **Result**: a *base model* that has absorbed grammar, facts and style, and continues any
  text — but does not follow instructions or hold a conversation. Ask it a question and it
  may answer with more questions.
* **Cost**: by far the most compute (GPT-3: ~3.6k petaflop/s-days).

## Finetuning

Start from the pretrained weights and keep training on a much smaller, curated dataset,
usually with a lower learning rate:

* **Supervised finetuning / instruction tuning** — (prompt, ideal response) pairs, often
  formatted with special tokens, e.g. `<|user|> … <|assistant|> …`. The loss is typically
  computed only on the response tokens. This turns a base model into an assistant.
* **Preference tuning** — RLHF (train a reward model on human rankings, optimise with PPO)
  or DPO (optimise directly on preferred/rejected pairs).
* **Domain adaptation** — continue pretraining on medical, legal or code text.
* **Parameter-efficient finetuning** — LoRA/QLoRA train small low-rank adapters instead of
  all weights, so billion-parameter models fit on one GPU.

## Finetuning this model

The training code barely changes: load the pickled model with `--resume`, point the data
loader at a finetuning dataset, lower `--lr` (e.g. `1e-4`), and — if you want a chatbot —
add prompt/response markers to the text. With a character tokenizer the markers can be
plain strings such as `### Question:` / `### Answer:`.
