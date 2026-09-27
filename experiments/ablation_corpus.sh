#!/usr/bin/env bash
# Ablation of the round-3 upgrades on the 18-book corpus (CPU friendly).
# Base: v2 recipe, 4 layers / 4 heads / 128 dims, BPE 4096, context 128, 2000 steps.
# Each run changes one thing. Results: runs/ablation_corpus/<name>/result.json.
set -euo pipefail
ITERS=${ITERS:-2000}
JOBS=${JOBS:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

[ -f data/corpus/train.txt ] || python data/download_corpus.py
[ -f data/corpus_bpe4k/meta.json ] || python -m llm.prepare --input data/corpus/train.txt \
    --val_input data/corpus/val.txt --out_dir data/corpus_bpe4k --vocab_size 4096 --workers 4
[ -f data/corpus_bpe4k_drop/meta.json ] || python -m llm.prepare --input data/corpus/train.txt \
    --val_input data/corpus/val.txt --out_dir data/corpus_bpe4k_drop --vocab_size 4096 --workers 4 \
    --bpe_dropout 0.1 --dropout_copies 4

export COMMON="--n_layer 4 --n_head 4 --n_embd 128 --block_size 128 --batch_size 32 --max_iters $ITERS \
 --eval_interval 250 --eval_iters 40 --lr 1e-3 --dropout 0.1"
run() {
  local name=$1; shift
  [ -f runs/ablation_corpus/$name/result.json ] && { echo "skip $name"; return; }
  python -m llm.train --out_dir runs/ablation_corpus/$name $COMMON "$@" > runs/ablation_corpus/$name.log 2>&1
  echo "done $name"
}
mkdir -p runs/ablation_corpus
export -f run

cat <<JOBS | xargs -P "$JOBS" -I{} bash -c "run {}"
a_base             --data_dir data/corpus_bpe4k
b_muon             --data_dir data/corpus_bpe4k --optimizer muon
c_ema              --data_dir data/corpus_bpe4k --ema 0.998
d_bpe_dropout      --data_dir data/corpus_bpe4k_drop
e_qknorm_softcap   --data_dir data/corpus_bpe4k --qk_norm --logit_softcap 30
f_valres_unet      --data_dir data/corpus_bpe4k --value_residual --unet_skips
g_moe4             --data_dir data/corpus_bpe4k --n_experts 4 --moe_top_k 2
h_gqa2             --data_dir data/corpus_bpe4k --n_kv_head 2
JOBS
python experiments/summarize.py runs/ablation_corpus
