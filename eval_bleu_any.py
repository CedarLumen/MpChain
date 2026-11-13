"""BLEU 评估（自动识别 ckpt 类型：MpChainTr / 标准 MoE）+ 逐句记录落盘。

python eval_bleu_any.py ck1.pt ck2.pt ...
环境变量 MPCHAIN_DATA 指定数据集（默认 data/un_sub.npz）。

除汇总表外，每个 ckpt 还会把逐句记录写到 evaldump/<ckpt名>__<数据集名>.jsonl，
供 stats_ablation.py 做配对检验（同 seed、同 VA、同 500 句，索引一一对应）。
"""
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import torch

from mpchain_tr import MpChainTr
from train_moe import MoEEncoder

LEN, LZH = 100, 32
DATA = os.environ.get("MPCHAIN_DATA", "data/un_sub.npz")
D = np.load(DATA)
ZS = [str(c) for c in D["zs"]]
EN, ZH, EL, ZL, SP = D["en"], D["zh"], D["en_len"], D["zh_len"], D["split"]
VE, VZ = len(D["es"]) + 2, len(D["zs"]) + 2
VA = np.where(SP == 1)[0]
TAG = os.path.basename(DATA).replace(".npz", "")
DUMP_DIR = "evaldump"


def pick_step(stops, T):
    p = torch.stack(stops, 1)
    pick = (p > 0).float().argmax(1)
    pick[(p <= 0).all(1)] = T - 1
    return pick


@torch.no_grad()
def generate_mp(net, rows, maxlen=32, ban=(0, 1)):
    B = len(rows)
    en = np.zeros((B, LEN), np.int64)
    env = np.zeros((B, LEN, 1), bool)
    tl = np.zeros(B, np.int64)
    for i, j in enumerate(rows):
        ne = int(EL[j])
        en[i, LEN - ne:] = EN[j, :ne]
        env[i, LEN - ne:] = True
        tl[i] = int(ZL[j])
    en, env = torch.from_numpy(en).cuda(), torch.from_numpy(env).cuda()
    seq = np.zeros((B, LZH), np.int64)
    lens = np.ones(B, np.int64)
    out = np.zeros((B, maxlen), np.int64)
    steps = np.zeros(B, np.int64)
    T = net.T
    for s in range(maxlen):
        zh = np.zeros((B, LZH), np.int64)
        zhv = np.zeros((B, LZH, 1), bool)
        for i in range(B):
            L = int(lens[i])
            zh[i, LZH - L:] = seq[i, :L]
            zhv[i, LZH - L:] = True
        outs, stops, _, _ = net(en, env, torch.from_numpy(zh).cuda(), torch.from_numpy(zhv).cuda())
        pick = pick_step(stops, T)
        if s == 0:
            steps = (pick + 1).cpu().numpy()
        lg = torch.stack(outs, 1).gather(
            1, pick[:, None].unsqueeze(-1).expand(-1, 1, VZ)).squeeze(1).clone()
        for b in ban:
            lg[:, b] = -1e9
        nxt = lg.argmax(-1)
        out[:, s] = nxt.cpu().numpy()
        for i in range(B):
            if lens[i] < LZH:
                seq[i, lens[i]] = nxt[i].item()
                lens[i] += 1
    return out, tl, steps


@torch.no_grad()
def generate_moe(net, rows, maxlen=32, ban=(0, 1)):
    B = len(rows)
    en = np.zeros((B, LEN), np.int64)
    env = np.zeros((B, LEN, 1), bool)
    tl = np.zeros(B, np.int64)
    for i, j in enumerate(rows):
        ne = int(EL[j])
        en[i, :ne] = EN[j, :ne]
        env[i, :ne] = True
        tl[i] = int(ZL[j])
    en, env = torch.from_numpy(en).cuda(), torch.from_numpy(env).cuda()
    seq = np.zeros((B, LZH), np.int64)
    lens = np.ones(B, np.int64)
    out = np.zeros((B, maxlen), np.int64)
    for s in range(maxlen):
        zh = np.zeros((B, LZH), np.int64)
        zhv = np.zeros((B, LZH, 1), bool)
        for i in range(B):
            L = int(lens[i])
            zh[i, :L] = seq[i, :L]
            zhv[i, :L] = True
        logits, _ = net(en, env, torch.from_numpy(zh).cuda(), torch.from_numpy(zhv).cuda())
        lg = logits.clone()
        for b in ban:
            lg[:, b] = -1e9
        nxt = lg.argmax(-1)
        out[:, s] = nxt.cpu().numpy()
        for i in range(B):
            if lens[i] < LZH:
                seq[i, lens[i]] = nxt[i].item()
                lens[i] += 1
    return out, tl, np.ones(B, np.int64)


def to_str(ids):
    return "".join(ZS[c - 2] if c >= 2 else "" for c in ids)


def sent_stats(h, r, n=2):
    """单句：BLEU-2 + n-gram 命中/总数（配对 & 自举用）。"""
    match, total = [0] * n, [0] * n
    for k in range(1, n + 1):
        hg = Counter(h[i:i + k] for i in range(max(len(h) - k + 1, 0)))
        rg = Counter(r[i:i + k] for i in range(max(len(r) - k + 1, 0)))
        match[k - 1] = sum(min(c, rg[g]) for g, c in hg.items())
        total[k - 1] = max(len(h) - k + 1, 0)
    ps = [(match[i] / total[i] if total[i] else 0.0) for i in range(n)]
    if min(ps) <= 0:
        return 0.0, ps, match, total
    bp = min(1.0, math.exp(1 - len(r) / max(len(h), 1)))
    return bp * math.exp(sum(math.log(p) for p in ps) / n), ps, match, total


def corpus_bleu(hyps, refs, n=2):
    match, total = [0] * n, [0] * n
    hl = rl = 0
    for h, r in zip(hyps, refs):
        hl += len(h)
        rl += len(r)
        for k in range(1, n + 1):
            hg = Counter(h[i:i + k] for i in range(max(len(h) - k + 1, 0)))
            rg = Counter(r[i:i + k] for i in range(max(len(r) - k + 1, 0)))
            match[k - 1] += sum(min(c, rg[g]) for g, c in hg.items())
            total[k - 1] += max(len(h) - k + 1, 0)
    ps = [(match[i] / total[i] if total[i] else 0.0) for i in range(n)]
    if min(ps) <= 0:
        return 0.0, ps
    bp = min(1.0, math.exp(1 - rl / max(hl, 1)))
    return bp * math.exp(sum(math.log(p) for p in ps) / n), ps


def eval_one(path, n=500, seed=11):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    a = ck["args"]
    if "mult" in a:
        arch = "moe"
        net = MoEEncoder(N=a["N"], d=a["d"], mult=a["mult"]).cuda()
    else:
        arch = "mpchain"
        net = MpChainTr(N=a["N"], Ve=VE, Vz=VZ, d=a["d"], dmp=a.get("dmp", 0) or a["d"],
                        T=a["T"]).cuda()
    net.load_state_dict(ck["model"])
    net.eval()
    rng = np.random.default_rng(seed)
    rows = rng.choice(VA, n, replace=False)
    gen, tl, steps = (generate_moe if arch == "moe" else generate_mp)(net, rows)
    hyps = [to_str(gen[i][:tl[i]]) for i in range(n)]
    refs = [to_str(ZH[j, :int(ZL[j])]) for j in rows]
    b, ps = corpus_bleu(hyps, refs, 2)
    nt = sum(int(tl[i]) for i in range(n))
    hit = sum(1 for i in range(n) for k in range(int(tl[i])) if gen[i][k] == ZH[rows[i], k])

    # 逐句记录（配对检验用）——rows 完全由 seed 决定，所有模型同一批句子
    os.makedirs(DUMP_DIR, exist_ok=True)
    out_path = os.path.join(DUMP_DIR, f"{os.path.basename(path)}__{TAG}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            sb, sps, m, t = sent_stats(hyps[i], refs[i], 2)
            h_len = int(tl[i])
            h_hit = sum(1 for k in range(h_len) if gen[i][k] == ZH[rows[i], k])
            f.write(json.dumps({
                "idx": int(rows[i]), "hyp": hyps[i], "ref": refs[i],
                "bleu2": sb, "p1": sps[0], "p2": sps[1],
                "m1": m[0], "t1": t[0], "m2": m[1], "t2": t[1],
                "rlen": len(refs[i]),
                "hit": h_hit, "len": h_len, "steps": int(steps[i]),
            }, ensure_ascii=False) + "\n")
    return (b, ps, hit / max(nt, 1), ck.get("step"), arch, out_path)


if __name__ == "__main__":
    files = sys.argv[1:]
    print(f"{'ckpt':<22}{'arch':<9}{'step':>8}{'BLEU-2':>9}{'p1':>7}{'p2':>7}{'tok_acc':>9}")
    for f in files:
        if not os.path.exists(f):
            print(f"{f:<22}(missing)")
            continue
        b, ps, acc, step, arch, dump = eval_one(f)
        print(f"{os.path.basename(f):<22}{arch:<9}{str(step):>8}{b:>9.4f}"
              f"{ps[0]:>7.3f}{ps[1]:>7.3f}{acc:>9.4f}", flush=True)
    print(f"\n逐句记录 -> {os.path.abspath(DUMP_DIR)}/*.jsonl", flush=True)
