#!/usr/bin/env bash
# Ablation on the Wizard of Oz corpus (CPU friendly). Every run uses the same model shape
# (4 layers, 4 heads, 128 dims) and the same number of steps; only one thing changes at a
# time. Results: runs/ablation/<name>/result.json, summarised by experiments/summarize.py.
set -euo pipefail
ITERS=${ITERS:-2000}
JOBS=${JOBS:-2}                         # runs in parallel
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

python -m llm.prepare --input data/wizard_of_oz.txt --out_dir data/oz_char --tokenizer char
for V in 512 1024 2048; do
  python -m llm.prepare --input data/wizard_of_oz.txt --out_dir data/oz_bpe$V --vocab_size $V
done

COMMON="--n_layer 4 --n_head 4 --n_embd 128 --batch_size 32 --max_iters $ITERS --eval_interval 200 --eval_iters 40"
V1_ARCH="--pos_emb learned --norm layer --mlp relu --no_tie --bias"
V1_RECIPE="--lr 3e-4 --min_lr 3e-4 --warmup_iters 0 --weight_decay 0.01 --beta2 0.999 --grad_clip 0 --dropout 0.2"

run() {  # name, args...
  local name=$1; shift
  python -m llm.train --out_dir runs/ablation/$name $COMMON "$@" > runs/ablation/$name.log 2>&1
  echo "done $name"
}
mkdir -p runs/ablation
export -f run; export COMMON

cat <<JOBS | xargs -P "$JOBS" -I{} bash -c "run {}"
1_v1_baseline        --data_dir data/oz_char    $V1_ARCH $V1_RECIPE --block_size 64
2_v1_arch_v2_recipe  --data_dir data/oz_char    $V1_ARCH --block_size 128
3_v2_arch_char       --data_dir data/oz_char    --block_size 128
4_v2_bpe512          --data_dir data/oz_bpe512  --block_size 128
5_v2_bpe1024         --data_dir data/oz_bpe1024 --block_size 128
6_v2_bpe2048         --data_dir data/oz_bpe2048 --block_size 128
JOBS
python experiments/summarize.py runs/ablation
