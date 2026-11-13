"""Shrink the dataset for the Raspberry Pi: keep only the vocab + 600 test sentences.

Why: in un_sub.npz, EN is (608k, 100) int64 ≈ 486 MB, which together with ZH is about 650 MB;
on a 2 GB Pi merely loading the data eats half the memory, and the SD card slows it down beyond bearing.
Timing only needs fixed-shape inputs, so keeping 600 test sentences (all marked split=1) is enough,
the vocab is kept as it is and the shapes do not change.

python make_pi_data.py     # -> pi_bundle/data/un_sub.npz
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "un_sub.npz")
DST = os.path.join(ROOT, "pi_bundle", "data", "un_sub.npz")
NROW = 600

D = np.load(SRC)
print("source keys:", sorted(D.files))
VA = np.where(D["split"] == 1)[0]
rows = VA[:NROW]
out = {}
for k in D.files:
    v = D[k]
    if v.ndim >= 1 and v.shape[0] == D["en"].shape[0]:
        out[k] = v[rows]
    else:
        out[k] = v
out["split"] = np.ones(len(rows), dtype=D["split"].dtype)
os.makedirs(os.path.dirname(DST), exist_ok=True)
np.savez_compressed(DST, **out)
sz = os.path.getsize(DST) / 1024 ** 2
print("wrote %s: %d rows, %.2f MB (original %.1f MB)" % (DST, len(rows), sz, os.path.getsize(SRC) / 1024 ** 2))
print("en shape %s / zh shape %s / vocab %d %d"
      % (out["en"].shape, out["zh"].shape, len(out["es"]) + 2, len(out["zs"]) + 2))
