"""Routing sensitivity probe (internal experiment, not part of the paper).

Question: for the same English sentence with only one detail changed (one digit / one character), does the chain pick the same small models?

Four conditions:
  A the original sentence unchanged (determinism baseline, must be 100% identical)
  B change one digit
  C change one non-digit character
  D swap in a completely unrelated sentence (the reference for "genuinely different")

Metrics:
  (1) fraction of fully identical routing sequences
  (2) per-chain-step expert agreement (averaged over the T-1 steps)
  (3) JS divergence of the expert usage distribution (40-dim)
"""
import argparse
import math
from collections import Counter

import numpy as np
import torch

from mpchain_tr import MpChainTr

LEN, LZH = 100, 32
dev = "cuda"


def build_net(ck_path):
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    ar = ck["args"]
    net = MpChainTr(ar["N"], 153, 2048, ar["d"], ar.get("dmp") or ar["d"], ar["T"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


def decode_ids(ids, zs):
    return "".join(zs[t - 2] for t in ids if t >= 2)


def encode(en_ids, zh_ids, k):
    en = np.zeros((1, LEN), np.int64)
    env = np.zeros((1, LEN, 1), bool)
    ne = len(en_ids)
    en[0, LEN - ne:] = en_ids
    env[0, LEN - ne:] = True
    zh = np.zeros((1, LZH), np.int64)
    zhv = np.zeros((1, LZH, 1), bool)
    if k:
        zh[0, LZH - k:] = zh_ids[:k]
        zhv[0, LZH - k:] = True
    return (torch.from_numpy(en).cuda(), torch.from_numpy(env).cuda(),
            torch.from_numpy(zh).cuda(), torch.from_numpy(zhv).cuda())


@torch.no_grad()
def routes(net, en_ids, zh_ids, k):
    en, env, zh, zhv = encode(en_ids, zh_ids, k)
    outs, stops, _, idxs = net(en, env, zh, zhv)
    return [int(i[0]) for i in idxs]          # routing sequence of each chain


def js(p, q, N=40):
    a = np.array([p.get(i, 0) for i in range(N)], float)
    b = np.array([q.get(i, 0) for i in range(N)], float)
    a = a / max(a.sum(), 1e-9) + 1e-12
    b = b / max(b.sum(), 1e-9) + 1e-12
    m = 0.5 * (a + b)
    return 0.5 * np.sum(a * np.log(a / m)) + 0.5 * np.sum(b * np.log(b / m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ck_seed0.pt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--data", default="data/un_sub.npz")
    a = ap.parse_args()

    D = np.load(a.data)
    zs = [str(c) for c in D["zs"]]
    es = [str(c) for c in D["es"]]
    EN, ZH, EL, ZL, SP = D["en"], D["zh"], D["en_len"], D["zh_len"], D["split"]
    VA = np.where(SP == 1)[0]
    net = build_net(a.ckpt)
    rng = np.random.default_rng(0)

    DIGITS = "0123456789"
    stats = {t: {"seq": 0, "agree": [], "js": [], "step": []} for t in "ABCD"}
    examples = []

    rows = rng.choice(VA, a.n, replace=False)
    for r in rows:
        ne, nz = int(EL[r]), int(ZL[r])
        if nz < 4:
            continue
        k = nz // 2
        base_en = [int(t) for t in EN[r, :ne]]
        base_zh = [int(t) for t in ZH[r, :nz]]
        ref = routes(net, base_en, base_zh, k)

        # A run the original once more, unchanged
        same = routes(net, base_en, base_zh, k)
        # B change one digit (skip this condition if no digit is found)
        en_txt = [es[t - 2] for t in base_en]
        pos_b = next((i for i, c in enumerate(en_txt) if c in DIGITS), None)
        var_b = None
        if pos_b is not None:
            v = base_en.copy()
            v[pos_b] = es.index(str((int(en_txt[pos_b]) + 1) % 10)) + 2
            var_b = routes(net, v, base_zh, k)
        # C change one non-digit, non-space character
        pos_c = next((i for i, c in enumerate(en_txt)
                      if c not in DIGITS and c.strip()), None)
        var_c = None
        if pos_c is not None:
            v = base_en.copy()
            alt = next(ch for ch in "abcdefghijklmnopqrstuvwxyz" if ch != en_txt[pos_c])
            v[pos_c] = es.index(alt) + 2
            var_c = routes(net, v, base_zh, k)
        # D swap in an unrelated sentence
        r2 = int(rng.choice(VA))
        other_en = [int(t) for t in EN[r2, :int(EL[r2])]]
        var_d = routes(net, other_en, base_zh, k)

        for tag, seq in (("A", same), ("B", var_b), ("C", var_c), ("D", var_d)):
            if seq is None:
                continue
            stats[tag]["seq"] += int(seq == ref)
            stats[tag]["agree"].append(float(np.mean([x == y for x, y in zip(ref, seq)])))
            stats[tag]["js"].append(float(js(Counter(ref), Counter(seq))))
            stats[tag]["step"].append([float(x == y) for x, y in zip(ref, seq)])
        if var_b is not None and len(examples) < 5:
            examples.append((decode_ids(base_en, es), decode_ids(base_zh, zs),
                             ref, var_b))

    print(f"ckpt {a.ckpt}   sentences {a.n}   chain length cap T={net.T}")
    print(f"{'set':<4}{'description':<22}{'seq identical':>14}{'step agree':>14}{'JS div':>10}")
    desc = {"A": "repeat original (determinism)", "B": "change one digit", "C": "change one non-digit char",
            "D": "swap in unrelated sentence"}
    for tag in "ABCD":
        s = stats[tag]
        n = len(s["agree"])
        if not n:
            print(f"{tag:<4}{desc[tag]:<22}{'—':>14}{'—':>14}{'—':>10}")
            continue
        print(f"{tag:<4}{desc[tag]:<22}{s['seq']/n:>13.0%}"
              f"{np.mean(s['agree']):>13.3f}{np.mean(s['js']):>10.4f}")

    print("\nPer-chain-step expert agreement (step 1 is always expert 0, so counting starts at step 2):")
    print(f"{'set':<4}" + "".join(f"{'step'+str(i+1):>8}" for i in range(net.T - 1)))
    for tag in "ABCD":
        s = stats[tag]
        if not s["step"]:
            continue
        arr = np.array(s["step"]).mean(0)
        print(f"{tag:<4}" + "".join(f"{v:>8.2f}" for v in arr))

    print("\nExamples (routing sequence of the original / after changing one digit):")
    for en_t, zh_t, r0, r1 in examples:
        print(f"  EN: {en_t[:60]}")
        print(f"  ZH: {zh_t[:24]}")
        print(f"  orig: {r0}")
        print(f"  one digit changed: {r1}   {'same' if r0 == r1 else 'different'}")


if __name__ == "__main__":
    main()
