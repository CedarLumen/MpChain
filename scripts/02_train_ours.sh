#!/usr/bin/env bash
# Train the chain with the paper's main configuration, three seeds (paper section 5.1).
# Configuration: N=40 experts, d=128, memory width 128, T=6 calls, batch 128, lr 1e-3, 100k steps.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

COMMON="--steps 100000 --N 40 --d 128 --dmp 128 --T 6 --batch 128 --lr 1e-3 --eval 5000"

for SEED in 0 1 2; do
  echo "== seed ${SEED}"
  python src/train_tr.py ${COMMON} --seed "${SEED}" --ckpt "ck_seed${SEED}.pt" \
    2>&1 | tee "results/train_seed${SEED}.log"
done

echo "== done: ck_seed0.pt ck_seed1.pt ck_seed2.pt"
