#!/usr/bin/env bash
# The four mechanism ablations and the two routing controls (paper sections 3.5, 5.2, 5.4, 5.5).
# Each ablation is a full training run with one switch flipped, then the same evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

MAIN="--steps 100000 --N 40 --d 128 --dmp 128 --T 6 --batch 128 --lr 1e-3 --eval 5000"

echo "== 5.2 memory removed"
python src/train_tr.py ${MAIN} --no_mp 1 --ckpt ck_ab_nomp.pt 2>&1 | tee results/train_abl_nomp.log

echo "== 5.2 routing replaced by a fixed permutation"
python src/train_tr.py ${MAIN} --rand_route 1 --ckpt ck_ab_randroute.pt 2>&1 | tee results/train_abl_randroute.log

echo "== 5.2 output taken from the last call only"
python src/train_tr.py ${MAIN} --last_only 1 --ckpt ck_ab_lastonly.pt 2>&1 | tee results/train_abl_lastonly.log

echo "== evaluate the three arms"
python src/eval_bleu_any.py ck_seed0.pt ck_ab_nomp.pt ck_ab_randroute.pt ck_ab_lastonly.pt \
  2>&1 | tee results/EVAL_ABLATION.txt

echo "== 5.4 which memory slots carry the metric"
python src/slot_ablation.py 2>&1 | tee results/SLOT_ABLATION.txt

echo "== 5.5 perturbation of the routing decision"
python src/route_sensitivity.py --ckpt ck_seed0.pt --n 60 2>&1 | tee results/route_sens.txt

echo "== 5.5 permutation control: usage preserved, content correspondence broken"
python src/route_shuffle.py --ckpt ck_seed0.pt --n 500 2>&1 | tee results/route_shuffle.txt

echo "== 3.5 re-instantiation with a longer chain"
python src/eval_extend_t.py 2>&1 | tee results/EXTEND_T.txt

echo "== done"
