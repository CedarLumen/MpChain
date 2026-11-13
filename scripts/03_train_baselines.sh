#!/usr/bin/env bash
# Train the two baselines at matched parameter count (paper sections 5.1 and 5.6).
#   mixture of experts : N=40 experts, d=128, mult=16  -> 21.5M parameters
#   dense Transformer  : 12 layers, d=384, ffn 1536    -> 23.0M parameters
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

echo "== mixture of experts"
python src/train_moe.py --steps 100000 --N 40 --d 128 --mult 16 --batch 128 --lr 1e-3 \
  --eval 5000 --ckpt ck_moe.pt 2>&1 | tee results/train_moe.log

echo "== dense Transformer"
python src/train_tf.py --steps 100000 --batch 128 --d 384 --layers 12 --lr 3e-4 \
  --eval 2500 --ckpt ck_tf100k.pt 2>&1 | tee results/train_tf.log

echo "== done: ck_moe.pt ck_tf100k.pt"
