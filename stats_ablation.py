"""消融/基线对照的统计：均值±std、配对自举 95% CI、配对检验。

输入：eval_bleu_any.py 落盘的 evaldump/<ckpt>__<dataset>.jsonl（500 句，同 seed，
      所有模型跑的是同一批句子 ⇒ 可以逐句配对）。

输出：每张表给出
  - 每个模型的语料级 BLEU-2 / token acc
  - 主模型 3 seed 的 mean ± std（run-to-run 噪声底）
  - 各对照臂相对 baseline 的 ΔBLEU 与句子级配对自举 95% CI（1000 次）
  - 配对检验：Wilcoxon signed-rank（逐句 BLEU-2）、配对 t（逐句 token 命中率）
用法: python stats_ablation.py [--ref ck_seed0.pt]
"""
import glob
import json
import math
import os
import sys

import numpy as np

DUMP = "evaldump"
MAIN = ["ck_seed0.pt", "ck_seed1.pt", "ck_seed2.pt"]
ARMS = [
    ("ck_ab_nomp.pt", "w/o Mp（去掉公共存储）"),
    ("ck_ab_randroute.pt", "随机路由"),
    ("ck_ab_lastonly.pt", "只监督最后一步"),
    ("ck_moe.pt", "标准 MoE 基线（token-level top-1）"),
]
NBOOT = 1000


def load(model, tag):
    p = os.path.join(DUMP, f"{model}__{tag}.jsonl")
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p, encoding="utf-8")]
    recs.sort(key=lambda r: r["idx"])
    return recs


def corpus_bleu(recs):
    m1 = sum(r["m1"] for r in recs); t1 = sum(r["t1"] for r in recs)
    m2 = sum(r["m2"] for r in recs); t2 = sum(r["t2"] for r in recs)
    hl = sum(r["len"] for r in recs); rl = sum(r["rlen"] for r in recs)
    return bleu_from_sums(m1, t1, m2, t2, hl, rl)


def bleu_from_sums(m1, t1, m2, t2, hl, rl):
    p1 = m1 / t1 if t1 else 0.0
    p2 = m2 / t2 if t2 else 0.0
    if min(p1, p2) <= 0:
        return 0.0
    bp = min(1.0, math.exp(1 - rl / max(hl, 1)))
    return bp * math.exp((math.log(p1) + math.log(p2)) / 2)


def tok_acc(recs):
    h = sum(r["hit"] for r in recs); n = sum(r["len"] for r in recs)
    return h / max(n, 1)


def paired_bootstrap(ref, arm, nboot=NBOOT, seed=0):
    """句子级配对自举：返回 (Δ, 95% CI 下限, 上限, 双侧 p)。"""
    n = len(ref)
    assert [r["idx"] for r in ref] == [r["idx"] for r in arm], "句子不一致，无法配对"
    a = {k: np.array([r[k] for r in ref], dtype=np.float64) for k in ("m1", "t1", "m2", "t2", "len", "rlen")}
    b = {k: np.array([r[k] for r in arm], dtype=np.float64) for k in ("m1", "t1", "m2", "t2", "len")}
    rl = a["rlen"].sum()
    d0 = corpus_bleu(arm) - corpus_bleu(ref)
    rng = np.random.default_rng(seed)
    deltas = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        br = bleu_from_sums(a["m1"][idx].sum(), a["t1"][idx].sum(), a["m2"][idx].sum(),
                            a["t2"][idx].sum(), a["len"][idx].sum(), rl)
        ba = bleu_from_sums(b["m1"][idx].sum(), b["t1"][idx].sum(), b["m2"][idx].sum(),
                            b["t2"][idx].sum(), b["len"][idx].sum(), rl)
        deltas[i] = ba - br
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    frac = float((deltas <= 0).mean())
    p = 2 * min(frac, 1 - frac)
    p = max(p, 1.0 / nboot)
    return d0, lo, hi, p


def wilcoxon_p(d):
    """Wilcoxon signed-rank 的正态近似（不依赖 scipy）。"""
    d = np.asarray(d, dtype=np.float64)
    d = d[d != 0]
    n = len(d)
    if n < 10:
        return float("nan"), n
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    # 处理并列：平均秩
    absd = np.abs(d)
    for v in np.unique(absd):
        m = absd == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    wp = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (wp - mu) / sd
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), n


def paired_t(d):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = d.mean()
    s = d.std(ddof=1)
    if s == 0:
        return float("nan"), float("nan"), n
    t = m / (s / math.sqrt(n))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p, n


def report(tag, ref_name="ck_seed0.pt"):
    main = {m: load(m, tag) for m in MAIN}
    if not any(main.values()):
        print(f"\n## {tag}: 没有记录，跳过")
        return
    print(f"\n## 数据集 `{tag}`（500 句，全部模型同一批句子）\n")
    print("| 模型 | BLEU-2 | tok_acc | 备注 |")
    print("|---|---|---|---|")
    bs = []
    for m in MAIN:
        r = main[m]
        if r is None:
            continue
        b = corpus_bleu(r)
        bs.append(b)
        print(f"| {m} | {b:.4f} | {tok_acc(r):.4f} | 主模型 seed |")
    if len(bs) > 1:
        bs = np.array(bs)
        print(f"| **主模型 mean±std** | **{bs.mean():.4f} ± {bs.std(ddof=1):.4f}** | "
              f"— | n={len(bs)} |")
    ref = main[ref_name] or next(v for v in main.values() if v)
    rb = corpus_bleu(ref)
    print(f"\n噪声底：3 seed 之间 BLEU-2 std = {np.std(bs, ddof=1):.4f}"
          f"（对照 = {ref_name}）\n" if len(bs) > 1 else f"\n对照 = {ref_name}\n")

    print("| 对照臂 | BLEU-2 | ΔBLEU vs 对照 | 95% CI（配对自举） | 自举 p | Wilcoxon p | 配对 t p | Δ采 token acc |")
    print("|---|---|---|---|---|---|---|---|")
    for name, desc in ARMS:
        r = load(name, tag)
        if r is None:
            print(f"| {desc} | (缺) | — | — | — | — | — | — |")
            continue
        b = corpus_bleu(r)
        d, lo, hi, pboot = paired_bootstrap(ref, r)
        dbleu = [a["bleu2"] - b0["bleu2"] for a, b0 in zip(r, ref)]
        _, nz = wilcoxon_p(dbleu)
        wp, _ = wilcoxon_p(dbleu)
        dacc = [a["hit"] / max(a["len"], 1) - b0["hit"] / max(b0["len"], 1) for a, b0 in zip(r, ref)]
        _, tp, _ = paired_t(dacc)
        print(f"| {desc} | {b:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {pboot:.3g} | "
              f"{wp:.3g} (n={nz}) | {tp:.3g} | {np.mean(dacc)*100:+.2f} pt |")

    # 逐句配对细节
    print(f"\n逐句配对（{desc if False else '对照'} vs 各臂，n=500）：")
    for name, desc in ARMS:
        r = load(name, tag)
        if r is None:
            continue
        dbleu = np.array([a["bleu2"] - b0["bleu2"] for a, b0 in zip(r, ref)])
        dacc = np.array([a["hit"] / max(a["len"], 1) - b0["hit"] / max(b0["len"], 1)
                         for a, b0 in zip(r, ref)])
        print(f"  - {desc}: ΔBLEU 均值 {dbleu.mean():+.4f} | 赢 {int((dbleu>0).sum())} / "
              f"平 {int((dbleu==0).sum())} / 输 {int((dbleu<0).sum())} 句 | "
              f"Δtoken acc {dacc.mean()*100:+.2f} pt, Cohen's d "
              f"{(dacc.mean()/dacc.std(ddof=1) if dacc.std(ddof=1) else float('nan')):+.2f}")


if __name__ == "__main__":
    ref = "ck_seed0.pt"
    if "--ref" in sys.argv:
        ref = sys.argv[sys.argv.index("--ref") + 1]
    tags = sorted({os.path.basename(p).split("__", 1)[1].replace(".jsonl", "")
                   for p in glob.glob(os.path.join(DUMP, "*.jsonl"))})
    if not tags:
        print("evaldump 为空：先把 eval_bleu_any.py 跑一遍")
        sys.exit(1)
    for t in tags:
        report(t, ref)
