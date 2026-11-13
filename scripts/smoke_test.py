"""Check that this repository runs, without the corpora and without training anything useful.

Builds a tiny synthetic archive with the schema of docs/data.md (800 rows, 34 English and 62
Chinese symbols), then runs src/train_tr.py for 20 steps on a 4-expert chain of width 32.
It needs a CUDA device, because the training script places the model there unconditionally.
Four seconds on a laptop GPU; the accuracy is meaningless on random data, and that is fine.

    python scripts/smoke_test.py

The synthetic archive is written to data/un_sub.npz, which is ignored by git.
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

N, VE, VZ = 800, 34, 62
rng = np.random.default_rng(0)
en_len = rng.integers(4, 40, N)
zh_len = rng.integers(3, 20, N)
en = np.zeros((N, 100), np.int16)
zh = np.zeros((N, 32), np.int16)
for i in range(N):
    en[i, :en_len[i]] = rng.integers(2, VE, en_len[i])
    zh[i, :zh_len[i]] = rng.integers(2, VZ, zh_len[i])

os.makedirs(DATA, exist_ok=True)
np.savez_compressed(
    os.path.join(DATA, "un_sub.npz"),
    en=en, zh=zh,
    en_len=en_len.astype(np.int8), zh_len=zh_len.astype(np.int8),
    split=(np.arange(N) % 2 == 0).astype(np.int8),   # half validation, the evaluator draws 6x256
    es=np.array([" "] * (VE - 2)), zs=np.array([" "] * (VZ - 2)),
)
print("synthetic archive:", os.path.join(DATA, "un_sub.npz"))

cmd = [sys.executable, os.path.join("src", "train_tr.py"),
       "--steps", "20", "--N", "4", "--d", "32", "--dmp", "32", "--T", "3",
       "--batch", "8", "--eval", "10", "--log", "10", "--compile", "0",
       "--ckpt", os.path.join(DATA, "smoke.pt")]
env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
print("$", " ".join(cmd))
r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
print("\n".join((r.stdout + r.stderr).strip().splitlines()[-8:]))
ok = r.returncode == 0 and "FINAL" in r.stdout
print("\nsmoke test:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
