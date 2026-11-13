"""Multi-seed statistics: mean±std of the main model / each ablation arm, plus per-seed paired Δ and sign consistency.

Input is evaldump/<ckpt>__<tag>.jsonl written by eval_bleu_any.py (the same 500 sentences, pairable per sentence).

Pairing: the arm of seed i vs the main model of seed i (same seed), giving Δ_i; we report the mean±std of the 3 Δ
and the 3/3 same-sign count, plus a sentence-level paired bootstrap 95% CI for seed0 (same protocol as the single-seed version).
"""
import glob
import json
import math
import os

import numpy as np

DUMP = "evaldump"

# tag -> (baseline list, {arm name: that arm's ckpt list (in seed 0/1/2 order)})
CONFIG = {
    "un_sub": (
        ["ck_seed0.pt", "ck_seed1.pt", "ck_seed2.pt"],
        {
            "w/o Mp": ["ck_ab_nomp.pt", "ck_ab_nomp_s1.pt", "ck_ab_nomp_s2.pt"],
            "random routing": ["ck_ab_randroute.pt", "ck_ab_randroute_s1.pt", "ck_ab_randroute_s2.pt"],
            "last-step-only supervision": ["ck_ab_lastonly.pt", "ck_ab_lastonly_s1.pt", "ck_ab_lastonly_s2.pt"],
            "standard MoE": ["ck_moe.pt", "ck_moe_s1.pt", "ck_moe_s2.pt"],
        },
    ),
    "opus_test": (
        ["ck_seed0.pt", "ck_seed1.pt", "ck_seed2.pt"],
        {
            "w/o Mp": ["ck_ab_nomp.pt", "ck_ab_nomp_s1.pt", "ck_ab_nomp_s2.pt"],
            "random routing": ["ck_ab_randroute.pt", "ck_ab_randroute_s1.pt", "ck_ab_randroute_s2.pt"],
            "last-step-only supervision": ["ck_ab_lastonly.pt", "ck_ab_lastonly_s1.pt", "ck_ab_lastonly_s2.pt"],
            "standard MoE": ["ck_moe.pt", "ck_moe_s1.pt", "ck_moe_s2.pt"],
        },
    ),
    "opus_train": (
        ["ck_opus_s0.pt", "ck_opus_s1.pt", "ck_opus_s2.pt"],
        {"standard MoE": ["ck_opus_moe_s0.pt", "ck_opus_moe_s1.pt", "ck_opus_moe_s2.pt"]},
    ),
}


def load(model, tag):
    p = os.path.join(DUMP, f"{model}__{tag}.jsonl")
    if not os.path.exists(p):
        return None
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


def corpus_bleu(recs):
    return bleu_from_sums(sum(r["m1"] for r in recs), sum(r["t1"] for r in recs),
                          sum(r["m2"] for r in recs), sum(r["t2"] for r in recs),
                          sum(r["len"] for r in recs), sum(r["rlen"] for r in recs))


def paired_bootstrap(ref, arm, nboot=1000, seed=0):
    n = len(ref)
    assert [r["idx"] for r in ref] == [r["idx"] for r in arm], "sentence mismatch"
    a = {k: np.array([r[k] for r in ref], float) for k in ("m1", "t1", "m2", "t2", "len")}
    b = {k: np.array([r[k] for r in arm], float) for k in ("m1", "t1", "m2", "t2", "len")}
    rl = sum(r["rlen"] for r in ref)
    d0 = corpus_bleu(arm) - corpus_bleu(ref)
    rng = np.random.default_rng(seed)
    ds = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        ds[i] = (bleu_from_sums(b["m1"][idx].sum(), b["t1"][idx].sum(), b["m2"][idx].sum(),
                                b["t2"][idx].sum(), b["len"][idx].sum(), rl)
                 - bleu_from_sums(a["m1"][idx].sum(), a["t1"][idx].sum(), a["m2"][idx].sum(),
                                  a["t2"][idx].sum(), a["len"][idx].sum(), rl))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    frac = float((ds <= 0).mean())
    return d0, lo, hi, max(2 * min(frac, 1 - frac), 1.0 / nboot)


def report(tag):
    base_names, arms = CONFIG[tag]
    base = [load(n, tag) for n in base_names]
    if not any(base):
        print(f"\n## {tag}: no dump found, skipping")
        return
    print(f"\n## dataset `{tag}` (500 sentences, pairable per sentence)\n")
    ok = [(n, b) for n, b in zip(base_names, base) if b]
    bb = [corpus_bleu(b) for _, b in ok]
    print("| model | BLEU-2 (per seed) | mean ± std |")
    print("|---|---|---|")
    print(f"| main model | {' / '.join(f'{x:.4f}' for x in bb)} | "
          f"**{np.mean(bb):.4f} ± {np.std(bb, ddof=1) if len(bb) > 1 else 0:.4f}** |")
    for name, lst in arms.items():
        rs = [load(m, tag) for m in lst]
        pairs = [(a, b) for a, b in zip(rs, base) if a is not None and b is not None]
        if not pairs:
            print(f"| {name} | (missing) | n/a |")
            continue
        vals = [corpus_bleu(a) for a, _ in pairs]
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        print(f"| {name} | {' / '.join(f'{x:.4f}' for x in vals)} | "
              f"{np.mean(vals):.4f} ± {sd:.4f} |")

    print("\n| arm | ΔBLEU per seed (vs main model of same seed) | mean ± std | same sign | seed0 paired 95% CI | p |")
    print("|---|---|---|---|---|---|")
    for name, lst in arms.items():
        rs = [load(m, tag) for m in lst]
        pairs = [(a, b) for a, b in zip(rs, base) if a is not None and b is not None]
        if not pairs:
            continue
        ds = [corpus_bleu(a) - corpus_bleu(b) for a, b in pairs]
        same = sum(1 for d in ds if (d > 0) == (ds[0] > 0))
        d0, lo, hi, p = paired_bootstrap(pairs[0][1], pairs[0][0])
        sd = np.std(ds, ddof=1) if len(ds) > 1 else 0.0
        print(f"| {name} | {' / '.join(f'{d:+.4f}' for d in ds)} | "
              f"{np.mean(ds):+.4f} ± {sd:.4f} | {same}/{len(ds)} | "
              f"[{lo:+.4f}, {hi:+.4f}] | {p:.3g} |")


if __name__ == "__main__":
    tags = [t for t in CONFIG if glob.glob(os.path.join(DUMP, f"*__{t}.jsonl"))]
    if not tags:
        print("no matching dump in evaldump")
    for t in tags:
        report(t)
