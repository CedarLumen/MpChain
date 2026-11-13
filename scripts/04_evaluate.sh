#!/usr/bin/env bash
# Evaluate every checkpoint and compute the cross-seed statistics (paper sections 5.1, 5.6).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

echo "== chain (three seeds) and mixture of experts"
python src/eval_bleu_any.py ck_seed0.pt ck_seed1.pt ck_seed2.pt ck_moe.pt \
  2>&1 | tee results/EVAL_UN.txt

echo "== dense baseline"
python src/eval_tf_bleu.py ck_tf100k.pt 2>&1 | tee results/EVAL_TF.txt

echo "== paired statistics across seeds (reads evaldump/)"
python src/stats_seeds.py 2>&1 | tee results/STATS.txt

echo "== done. Per-sentence records are in evaldump/, summary tables in results/"
