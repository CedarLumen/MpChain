"""T=3 vs T=6(seed0) 的配对比较：BLEU 差值的自举 95% CI。"""
import json
import math
import os

import numpy as np

DUMP = "evaldump"
TAG = "un_sub"


def load(name):
    p = os.path.join(DUMP, f"{name}__{TAG}.jsonl")
    recs = [json.loads(l) for l in open(p, encoding="utf-8")]
    recs.sort(key=lambda r: r["idx"])
    return recs


def bleu_from_sums(m1, t1, m2, t2, hl, rl):
    p1 = m1 / t1 if t1 else 0.0
    p2 = m2 / t2 if t2 else 0.0
    if min(p1, p2) <= 0:
        return 0.0
    bp = min(1.0, math.exp(1 - rl / max(hl, 1)))
    return bp * math.exp((math.log(p1) + math.log(p2)) / 2)


def corpus(recs):
    return bleu_from_sums(sum(r["m1"] for r in recs), sum(r["t1"] for r in recs),
                          sum(r["m2"] for r in recs), sum(r["t2"] for r in recs),
                          sum(r["len"] for r in recs), sum(r["rlen"] for r in recs))


a = load("ck_seed0.pt")          # T=6
b = load("ck_t3_long.pt")        # T=3
assert [r["idx"] for r in a] == [r["idx"] for r in b]
print(f"T=6 seed0 BLEU-2 {corpus(a):.4f}   tok_acc {sum(r['hit'] for r in a)/sum(r['len'] for r in a):.4f}  "
      f"平均步数 {np.mean([r['steps'] for r in a]):.2f}")
print(f"T=3        BLEU-2 {corpus(b):.4f}   tok_acc {sum(r['hit'] for r in b)/sum(r['len'] for r in b):.4f}  "
      f"平均步数 {np.mean([r['steps'] for r in b]):.2f}")

n = len(a)
A = {k: np.array([r[k] for r in a], float) for k in ("m1", "t1", "m2", "t2", "len")}
B = {k: np.array([r[k] for r in b], float) for k in ("m1", "t1", "m2", "t2", "len")}
rl = sum(r["rlen"] for r in a)
rng = np.random.default_rng(0)
ds = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    ds.append(bleu_from_sums(B["m1"][idx].sum(), B["t1"][idx].sum(), B["m2"][idx].sum(),
                             B["t2"][idx].sum(), B["len"][idx].sum(), rl)
              - bleu_from_sums(A["m1"][idx].sum(), A["t1"][idx].sum(), A["m2"][idx].sum(),
                               A["t2"][idx].sum(), A["len"][idx].sum(), rl))
ds = np.array(ds)
lo, hi = np.percentile(ds, [2.5, 97.5])
frac = float((ds <= 0).mean())
print(f"ΔBLEU (T=3 − T=6) = {corpus(b)-corpus(a):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
      f"p={max(2*min(frac,1-frac), 0.001):.3g}")
