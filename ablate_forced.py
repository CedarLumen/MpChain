"""评估期消融（不训练）：对已训好的 ckpt 直接改推理行为，检验两个卖点。

① 条件计算：自适应早停（平均 3.5 步）vs 固定跑满 6 步 vs 只用第 1 步
   —— 如果"少跑几步不掉点"，才配叫 conditional compute。
② 槽位贡献：mask 掉第 j 次写入后再读，看哪一步的写入真正被用到。

python ablate_forced.py ck_seed0.pt ck_seed1.pt ck_seed2.pt
"""
import math
import os
import sys
from collections import Counter

import numpy as np
import torch

from mpchain_tr import MpChainTr

LEN, LZH = 100, 32
D = np.load(os.environ.get("MPCHAIN_DATA", "data/un_sub.npz"))
ZS = [str(c) for c in D["zs"]]
EN, ZH, EL, ZL, SP = D["en"], D["zh"], D["en_len"], D["zh_len"], D["split"]
VE, VZ = len(D["es"]) + 2, len(D["zs"]) + 2
VA = np.where(SP == 1)[0]
N_EVAL = 500
SEED = 11


def pick_step(stops, T):
    p = torch.stack(stops, 1)
    pick = (p > 0).float().argmax(1)
    pick[(p <= 0).all(1)] = T - 1
    return pick


def to_str(ids):
    return "".join(ZS[c - 2] if c >= 2 else "" for c in ids)


def bleu(hyps, refs, n=2):
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
        return 0.0
    bp = min(1.0, math.exp(1 - rl / max(hl, 1)))
    return bp * math.exp(sum(math.log(p) for p in ps) / n)


@torch.no_grad()
def generate(net, rows, mode, k=0, slot=None, maxlen=32, ban=(0, 1)):
    """mode: 'adapt' 自适应早停 | 'force' 固定用第 k 步（1-based）
    slot: 0-based 槽位索引，None 表示不屏蔽"""
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
    T = net.T
    used = np.zeros(B, np.int64)
    for s in range(maxlen):
        zh = np.zeros((B, LZH), np.int64)
        zhv = np.zeros((B, LZH, 1), bool)
        for i in range(B):
            L = int(lens[i])
            zh[i, LZH - L:] = seq[i, :L]
            zhv[i, LZH - L:] = True
        ms = (slot,) if slot is not None else ()
        outs, stops, _, _ = net(en, env, torch.from_numpy(zh).cuda(),
                                torch.from_numpy(zhv).cuda(), mask_slots=ms)
        if mode == "adapt":
            pick = pick_step(stops, T)
            if s == 0:
                used = (pick + 1).cpu().numpy()
            lg = torch.stack(outs, 1).gather(
                1, pick[:, None].unsqueeze(-1).expand(-1, 1, VZ)).squeeze(1).clone()
        else:
            if s == 0:
                used[:] = k
            lg = outs[k - 1].clone()
        for b in ban:
            lg[:, b] = -1e9
        nxt = lg.argmax(-1)
        out[:, s] = nxt.cpu().numpy()
        for i in range(B):
            if lens[i] < LZH:
                seq[i, lens[i]] = nxt[i].item()
                lens[i] += 1
    return out, tl, used


@torch.no_grad()
def evaluate(net, rows, mode, k=0, slot=None):
    gen, tl, used = generate(net, rows, mode, k=k, slot=slot)
    hyps = [to_str(gen[i][:tl[i]]) for i in range(len(rows))]
    refs = [to_str(ZH[j, :int(ZL[j])]) for j in rows]
    nt = sum(int(tl[i]) for i in range(len(rows)))
    hit = sum(1 for i in range(len(rows)) for p in range(int(tl[i])) if gen[i][p] == ZH[rows[i], p])
    return bleu(hyps, refs, 2), hit / max(nt, 1), float(used.mean())


def load(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    a = ck["args"]
    net = MpChainTr(N=a["N"], Ve=VE, Vz=VZ, d=a["d"], dmp=a.get("dmp", 0) or a["d"], T=a["T"]).cuda()
    net.load_state_dict(ck["model"])
    net.eval()
    return net, a["T"]


if __name__ == "__main__":
    files = sys.argv[1:] or ["ck_seed0.pt"]
    rng = np.random.default_rng(SEED)
    rows = rng.choice(VA, N_EVAL, replace=False)
    print(f"{'ckpt':<14}{'设置':<22}{'BLEU-2':>9}{'tok_acc':>9}{'平均步数':>9}")
    for f in files:
        if not os.path.exists(f):
            print(f"{f} (missing)"); continue
        net, T = load(f)
        # ① 条件计算
        for mode, k, label in [("adapt", 0, "自适应早停（原始）"),
                               ("force", 1, "只跑第 1 步"),
                               ("force", T, f"固定跑满第 {T} 步")] + \
                              [("force", kk, f"固定第 {kk} 步") for kk in range(2, T)]:
            b, acc, st = evaluate(net, rows, mode, k=k)
            print(f"{os.path.basename(f):<14}{label:<22}{b:>9.4f}{acc:>9.4f}{st:>9.2f}", flush=True)
        # ② 槽位贡献（第 j 次写入，1-based 标注，传 0-based 索引）
        for j in range(1, T + 1):
            b, acc, st = evaluate(net, rows, "adapt", slot=j - 1)
            print(f"{os.path.basename(f):<14}{'屏蔽第 %d 次写入' % j:<18}{b:>9.4f}{acc:>9.4f}{st:>9.2f}", flush=True)
        del net
        torch.cuda.empty_cache()
