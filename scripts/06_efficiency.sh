#!/usr/bin/env bash
# Efficiency: FLOPs, wall-clock along the optimisation ladder, and the three device classes
# (paper section 6). The parts that need a second device are marked; skip them on one machine.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

echo "== 6.1 FLOPs per token"
python src/flops_report.py 2>&1 | tee results/flops.txt

echo "== 6.2 optimisation ladder (GPU)"
python src/bench_infer2.py --ckpt ck_seed0.pt --reps 20 2>&1 | tee results/BENCH_LADDER.txt

echo "== 6.2 both architectures at the same optimisation level (GPU)"
python src/bench_both.py --ckpt ck_seed0.pt --nt 32 --reps 15 2>&1 | tee results/BENCH_BOTH.txt

echo "== 6.4 laptop GPU and CPU: latency, board power, energy (needs nvidia-smi)"
python src/bench_edge.py --ckpt ck_seed0.pt --runs 3 2>&1 | tee results/EDGE_LOCAL.txt

echo "== 6.4 Raspberry Pi 5, 2 GB, CPU only (run this on the board)"
echo "    MPCHAIN_DATA=data/pi_sub.npz python src/pi_bench.py --threads 4"
echo "    MPCHAIN_DATA=data/pi_sub.npz python src/pi_bench.py --threads 1"

echo "== done"
