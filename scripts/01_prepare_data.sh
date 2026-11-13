#!/usr/bin/env bash
# Build the character-id archives. Requires the parquet files described in docs/data.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

: "${MPCHAIN_SRC:?set MPCHAIN_SRC to the United Nations en-zh parquet}"
export MPCHAIN_OUT="${MPCHAIN_OUT:-$PWD/data/un_sub.npz}"
mkdir -p "$(dirname "$MPCHAIN_OUT")"

echo "== main corpus"
python src/data.py

if [[ -n "${MPCHAIN_OPUS_SRC:-}" ]]; then
  echo "== second corpus"
  export MPCHAIN_OPUS_OUT="${MPCHAIN_OPUS_OUT:-$PWD/data/opus_train.npz}"
  python src/data_opus.py
else
  echo "== SKIP second corpus (MPCHAIN_OPUS_SRC not set)"
fi

echo "== reduced archive for the single-board benchmark"
python src/make_pi_data.py
