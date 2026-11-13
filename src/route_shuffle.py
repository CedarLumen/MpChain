"""Routing shuffle control: does the routing carry "content information" or a "marginal distribution"?

Procedure: decode the same batch of sentences twice.
  own       use the routing the model computes itself (equal to normal greedy decoding, used as the baseline)
  shuffled  randomly permute each sentence's routing plan within the same sentence-length bucket and force it
            (expert usage frequency stays the same; only the "which sentence gets which routing" correspondence is broken)
  random    pick an expert independently and uniformly at each step (even the marginal distribution is destroyed)

If shuffled falls clearly below own, the routing carries content-bound information rather than marginal
statistics such as "expert usage frequency".
"""
import argparse

import numpy as np
import torch

import eval_bleu_any as E
from mpchain_tr import MpChainTr

LEN, LZH = 100, 32
dev = "cuda"
BUCKETS = [(2, 6), (7, 12), (13, 20), (21, 32)]


def build_net(ck, ve=153, vz=2048):
    ck = torch.load(ck, map_location=dev, weights_only=False)
    ar = ck["args"]
    net = MpChainTr(ar["N"], ve, vz, ar["d"], ar.get("dmp") or ar["d"], ar["T"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


def pick_out(outs, stops, T, VZ):
    p = torch.stack(stops, 1)
    pick = (p > 0).float().argmax(1)
    pick[(p <= 0).all(1)] = T - 1
    return torch.stack(outs, 1).gather(
        1, pick[:, None].unsqueeze(-1).expand(-1, 1, VZ)).squeeze(1)


@torch.no_grad()
def decode(net, rows, mode, plans=None, maxlen=32, ban=(0, 1), T=6):
    """mode: own / forced. Returns (generation, lengths, the routing plan each sample used at the first step)."""
    B = len(rows)
    en = np.zeros((B, LEN), np.int64)
    env = np.zeros((B, LEN, 1), bool)
    for i, j in enumerate(rows):
        n = int(E.EL[j])
        en[i, LEN - n:] = E.EN[j, :n]
        env[i, LEN - n:] = True
    en_t = torch.from_numpy(en).cuda()
    env_t = torch.from_numpy(env).cuda()
    h0 = net.prefill(en_t, env_t)

    pre = np.zeros((B, maxlen), np.int64)
    lens = np.ones(B, np.int64)          # same as eval_bleu_any: the prefix starts with 1 pad
    gen = np.zeros((B, maxlen), np.int64)
    first_plan = None
    for s in range(maxlen):
        zh = np.zeros((B, LZH), np.int64)
        zhv = np.zeros((B, LZH, 1), bool)
        for i in range(B):
            L = int(lens[i])
            zh[i, LZH - L:] = pre[i, :L]
            zhv[i, LZH - L:] = True
        rp = None
        if mode == "forced":
            rp = torch.from_numpy(np.ascontiguousarray(plans)).cuda()
        outs, stops, _, idxs = net(en_t, env_t, torch.from_numpy(zh).cuda(),
                                   torch.from_numpy(zhv).cuda(), pre=h0, route_plan=rp)
        if first_plan is None:
            first_plan = torch.stack(idxs, 1).cpu().numpy()
        lg = pick_out(outs, stops, T, VZ=E.VZ)
        for b in ban:
            lg[:, b] = -1e9
        nxt = lg.argmax(-1).cpu().numpy()
        gen[:, s] = nxt                      # outputs are stored by step index
        for i in range(B):
            if lens[i] < maxlen:
                pre[i, lens[i]] = nxt[i]     # the prefix is stored by length (position 0 is the start pad)
                lens[i] += 1
    return gen, lens, first_plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    net = build_net(a.ckpt)
    rng = np.random.default_rng(11)
    rows = rng.choice(E.VA, a.n, replace=False)
    refs = [E.to_str(E.ZH[j, :int(E.ZL[j])]) for j in rows]
    zl = [int(E.ZL[j]) for j in rows]      # same protocol as eval_bleu_any: hyp is truncated to the reference length

    g0, l0, plans = decode(net, rows, "own", T=net.T)
    hyps0 = [E.to_str(g0[i][:zl[i]]) for i in range(len(rows))]
    b_own, _ = E.corpus_bleu(hyps0, refs, 2)
    acc_own = sum(1 for i in range(len(rows)) for k in range(zl[i])
                  if g0[i][k] == E.ZH[rows[i], k]) / sum(zl)

    # permute the routing plans within the same sentence-length bucket
    lens = np.array([int(E.ZL[j]) for j in rows])
    perm = np.arange(len(rows))
    for lo, hi in BUCKETS:
        idx = np.where((lens >= lo) & (lens <= hi))[0]
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    shuffled = plans[perm]
    changed = float((shuffled != plans).any(1).mean())

    g1, l1, _ = decode(net, rows, "forced", plans=shuffled, T=net.T)
    hyps1 = [E.to_str(g1[i][:zl[i]]) for i in range(len(rows))]
    b_sh, _ = E.corpus_bleu(hyps1, refs, 2)
    acc_sh = sum(1 for i in range(len(rows)) for k in range(zl[i])
                 if g1[i][k] == E.ZH[rows[i], k]) / sum(zl)

    rand_plans = rng.integers(0, net.N, size=plans.shape)
    g2, l2, _ = decode(net, rows, "forced", plans=rand_plans, T=net.T)
    hyps2 = [E.to_str(g2[i][:zl[i]]) for i in range(len(rows))]
    b_rd, _ = E.corpus_bleu(hyps2, refs, 2)
    acc_rd = sum(1 for i in range(len(rows)) for k in range(zl[i])
                 if g2[i][k] == E.ZH[rows[i], k]) / sum(zl)

    print(f"ckpt {a.ckpt}  sentences {len(rows)}  "
          f"({changed:.0%} of sentences changed the routing on at least one step in the permutation)")
    print(f"{'condition':<26}{'BLEU-2':>9}{'tok acc':>10}{'vs own':>10}")
    for tag, b, ac in (("own (model-chosen)", b_own, acc_own),
                       ("shuffled (permuted within length bucket)", b_sh, acc_sh),
                       ("random (uniform random per step)", b_rd, acc_rd)):
        print(f"{tag:<26}{b:>9.4f}{ac:>10.4f}{b/b_own:>10.3f}")


if __name__ == "__main__":
    main()
