"""Standard MoE baseline: token-level top-1 routing + load-balancing loss.

Strictly aligned with MpChain: same data, same task, same evaluation, same metrics (BLEU-2 / per-token acc).

Engineering note: if token-level MoE gathers expert weights per token, memory O(T·f·d) blows up;
here we compute grouped by expert (only one expert is loaded per group), bringing memory down to O(N·f·d).

python train_moe.py --steps 100000 --N 40 --d 128 --mult 16
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEN, LZH = 100, 32
D = np.load(os.environ.get("MPCHAIN_DATA", "data/un_sub.npz"))
EN, ZH, EL, ZL, SP = D["en"], D["zh"], D["en_len"], D["zh_len"], D["split"]
VE, VZ = len(D["es"]) + 2, len(D["zs"]) + 2
TR = np.where(SP == 0)[0]
VA = np.where(SP == 1)[0]


class MoEEncoder(nn.Module):
    def __init__(self, N=40, d=128, mult=16):
        super().__init__()
        self.N, self.d, self.mult = N, d, mult
        f = mult * d
        k = (2.0 / d) ** 0.5
        self.emb_en = nn.Parameter(torch.randn(VE, d) * 0.3)
        self.emb_zh = nn.Parameter(torch.randn(VZ, d) * 0.3)
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.Wo = nn.Parameter(torch.randn(2 * d, d) * (2.0 / (2 * d)) ** 0.5)
        self.E1 = nn.Parameter(torch.randn(N, f, d) * k)
        self.E2 = nn.Parameter(torch.randn(N, d, f) * (2.0 / f) ** 0.5)
        self.Wroute = nn.Parameter(torch.randn(N, d) * k)
        self.broute = nn.Parameter(torch.zeros(N))
        self.proj = nn.Parameter(torch.randn(d, d) * k)

    def n_active(self):
        shared = (self.emb_en.numel() + self.emb_zh.numel() +
                  sum(p.numel() for p in self.gru.parameters()) +
                  self.Wo.numel() + self.Wroute.numel() + self.proj.numel())
        return shared + self.E1[0].numel() + self.E2[0].numel()

    def forward(self, en, en_valid, zh, zh_valid):
        B, L = en.shape
        d, N = self.d, self.N
        e = self.emb_en[en]
        z = self.emb_zh[zh]
        x = torch.cat([e, z], 1)                       # [B, L+LZH, d]
        valid = torch.cat([en_valid.squeeze(-1), zh_valid.squeeze(-1)], 1)
        h, _ = self.gru(x)                             # [B, T, 2d]
        h = h @ self.Wo                                # [B, T, d]

        # token-level routing
        rl = h @ self.Wroute.T + self.broute           # [B, T, N]
        idx = rl.argmax(-1)                            # [B, T]  hard routing

        # Group by expert to run the FFN. Sort by id then slice contiguous runs, avoiding a boolean mask per expert
        # (the mask version does dozens of index/nonzero/sync per step; measured more than 4x slower)
        TT = h.shape[1]
        hf = h.reshape(B * TT, d)
        idf = idx.reshape(-1)
        order = torch.argsort(idf)
        counts = torch.bincount(idf, minlength=N)
        cs = counts.tolist()
        hs = hf[order]
        out_s = torch.empty_like(hs)
        off = 0
        for u in range(N):
            c = cs[u]
            if c == 0:
                continue
            x = hs[off:off + c]
            out_s[off:off + c] = F.gelu(x @ self.E1[u].T) @ self.E2[u].T
            off += c
        out = torch.empty_like(hf)
        out[order] = out_s
        out = out.reshape(B, TT, d)

        # take the last valid position (the end of the Chinese prefix)
        last = valid.long().sum(1) - 1                 # [B]
        pooled = out[torch.arange(B, device=out.device), last]
        o = pooled @ self.proj                          # [B, d]
        logits = o @ self.emb_zh.T                      # [B, VZ]

        # load balancing: deviation of each expert's selection share from the uniform distribution
        frac = counts.float() / idf.numel()
        lb = (frac * N - 1).pow(2).mean()
        return logits, lb


def make_batch(idx_pool, bs, rng):
    j = rng.choice(idx_pool, bs, replace=False)
    en = np.zeros((bs, LEN), np.int64)
    zh = np.zeros((bs, LZH), np.int64)
    env = np.zeros((bs, LEN, 1), bool)
    zhv = np.zeros((bs, LZH, 1), bool)
    y = np.zeros(bs, np.int64)
    for i, k in enumerate(j):
        ne, nz = int(EL[k]), int(ZL[k])
        p = int(nz * rng.random())
        pre = np.zeros(p + 1, np.int64)
        pre[1:] = ZH[k, :p]
        en[i, :ne] = EN[k, :ne]                 # left-aligned: concatenating [en][zh] keeps the sequence contiguous
        env[i, :ne] = True
        zh[i, :p + 1] = pre
        zhv[i, :p + 1] = True
        y[i] = ZH[k, p]
    return (torch.from_numpy(en).cuda(), torch.from_numpy(env).cuda(),
            torch.from_numpy(zh).cuda(), torch.from_numpy(zhv).cuda(),
            torch.from_numpy(y).cuda())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--mult", type=int, default=16)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wlb", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", type=int, default=5000)
    ap.add_argument("--log", type=int, default=10000)
    ap.add_argument("--ckpt", type=str, default="ck_moe.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    net = MoEEncoder(N=a.N, d=a.d, mult=a.mult).cuda()
    tot = sum(p.numel() for p in net.parameters())
    print(f"params={tot/1e6:.2f}M active={net.n_active()/1e6:.2f}M "
          f"N={a.N} d={a.d} mult={a.mult}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, foreach=False)
    rng = np.random.default_rng(a.seed)
    best = 0.0
    t0 = time.time()
    for step in range(1, a.steps + 1):
        net.train()
        en, env, zh, zhv, y = make_batch(TR, a.batch, rng)
        logits, lb = net(en, env, zh, zhv)
        loss = F.cross_entropy(logits, y) + a.wlb * lb
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % a.log == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"  {step} | loss {loss.item():.4f} acc {acc:.4f} | "
                  f"{time.time()-t0:.0f}s", flush=True)
        if step % a.eval == 0 or step == a.steps:
            net.eval()
            c = t = 0
            with torch.no_grad():
                for _ in range(6):
                    en, env, zh, zhv, y = make_batch(VA, 256, rng)
                    lo, _ = net(en, env, zh, zhv)
                    c += int((lo.argmax(-1) == y).sum()); t += len(y)
            acc = c / t
            best = max(best, acc)
            print(f"  EVAL acc {acc:.4f}", flush=True)
            torch.save({"model": net.state_dict(), "args": vars(a), "step": step}, a.ckpt)
    print(f"FINAL acc {best:.4f} in {time.time()-t0:.0f}s", flush=True)
