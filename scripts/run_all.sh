#!/usr/bin/env bash
# Full pipeline on one CUDA machine: data, three training runs, baselines, evaluation, statistics.
# Expect about three days of GPU time on a single card; each stage logs to results/.
set -euo pipefail
cd "$(dirname "$0")"

bash 01_prepare_data.sh
bash 02_train_ours.sh
bash 03_train_baselines.sh
bash 04_evaluate.sh

echo
echo "Next: bash 05_ablations.sh   (further training runs)"
echo "      bash 06_efficiency.sh  (FLOPs and wall-clock)"
