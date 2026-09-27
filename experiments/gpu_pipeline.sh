#!/usr/bin/env bash
# Full GPU pipeline: web-scale pretraining -> chat finetuning -> preference tuning -> export.
# Needs: a CUDA GPU (24 GB+ for the defaults), ~60 GB disk, `pip install datasets`.
# Rough cost on one RTX 4090: tokenizing ~1 h (CPU), pretraining ~20-24 h. On 8xA100, use
#   NGPU=8 (torchrun) and it takes ~2-3 h. Every step can be re-run on its own.
set -euo pipefail
NGPU=${NGPU:-1}
CHARS=${CHARS:-10e9}          # ~2.5B tokens of FineWeb-Edu (Chinchilla-optimal for ~120M params)
ITERS=${ITERS:-20000}         # 20k steps x 128k tokens/step = 2.6B tokens
WORKERS=${WORKERS:-$(nproc)}

# 1. data
[ -f data/fineweb/train.txt ] || python data/download_hf.py fineweb --out data/fineweb --max_chars "$CHARS"
[ -f data/fineweb_bpe32k/meta.json ] || python -m llm.prepare --input data/fineweb/train.txt \
    --val_input data/fineweb/val.txt --out_dir data/fineweb_bpe32k --vocab_size 32768 \
    --tokenizer_sample_mb 200 --workers "$WORKERS"

# 2. pretraining: GPT-2-small shape, modern recipe (Muon + EMA + QK-norm, bf16, torch.compile)
LAUNCH="python"
[ "$NGPU" -gt 1 ] && LAUNCH="torchrun --standalone --nproc_per_node $NGPU"
$LAUNCH -m llm.train --data_dir data/fineweb_bpe32k --out_dir runs/fineweb --preset gpt-small \
    --batch_size 16 --grad_accum $((8 / NGPU > 0 ? 8 / NGPU : 1)) --max_iters "$ITERS" --lr 6e-4 \
    --warmup_iters 700 --optimizer muon --ema 0.9995 --qk_norm --dropout 0.0 --weight_decay 0.1 \
    --eval_interval 1000 --eval_iters 50 --compile

# 3. chat: general instructions (Dolly) + grounded reading comprehension on the Oz books
[ -f data/dolly.jsonl ] || python data/download_hf.py dolly --out data/dolly.jsonl
[ -f data/chat/sft_train.jsonl ] || { python data/download_corpus.py; python data/make_chat_data.py; }
cat data/dolly.jsonl data/chat/sft_train.jsonl | shuf --random-source=<(yes) > data/chat/sft_mix.jsonl
python -m llm.finetune --ckpt runs/fineweb/best.pt --data data/chat/sft_mix.jsonl --out_dir runs/fineweb_sft \
    --epochs 3 --lr 1e-4 --batch_size 32 --dropout 0.0
python -m llm.dpo --ckpt runs/fineweb_sft/best.pt --data data/chat/dpo_train.jsonl \
    --val_data data/chat/dpo_val.jsonl --out_dir runs/fineweb_dpo --beta 0.1 --lr 5e-6

# 4. evaluate + export
python -m llm.evaluate --ckpt runs/fineweb/best.pt
python -m llm.chat_eval --ckpt runs/fineweb_dpo/best.pt --rag
python -m llm.export --ckpt runs/fineweb_dpo/best.pt --format gguf --out export/fineweb-chat.gguf
python -m llm.export --ckpt runs/fineweb_dpo/best.pt --format hf --out export/fineweb-chat-hf
echo "done: chat with  python -m llm.webui --ckpt runs/fineweb_dpo/best.pt --rag"
echo "      or         cd export && ollama create fineweb-chat -f Modelfile && ollama run fineweb-chat"
