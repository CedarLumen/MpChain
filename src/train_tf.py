"""Pure Transformer baseline: same total parameters, fully activated (contrast with MpChain's "large total, small active").

Input = [en chars][SEP][BOS][Chinese prefix], take the last position to predict the next token.
Shared en-zh vocabulary: en ids offset by VZ, SEP = VZ+VE.

python train_tf.py --steps 15000
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LEN, LZH = 100, 32
SLEN = LEN + 1 + LZH + 1
D = np.load(os.environ.get("MPCHAIN_DATA", "data/un_sub.npz"))
EN, ZH, EL, ZL, SP = D["en"], D["zh"], D["en_len"], D["zh_len"], D["split"]
VE, VZ = len(D["es"]) + 2, len(D["zs"]) + 2
SEP = VZ + VE
VT = VZ + VE + 1
TR = np.where(SP == 0)[0]
VA = np.where(SP == 1)[0]


class TF(nn.Module):
    def __init__(self, V=VT, d=384, layers=12, nhead=6):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, SLEN, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VZ)

    def forward(self, x):
        mask = x == 0
        h = self.emb(x) + self.pos[:, :x.shape[1]]
        h = self.norm(self.enc(h, src_key_padding_mask=mask))
        return self.head(h[:, -1])


def make_batch(rows, bs, rng):
    x = np.zeros((bs, SLEN), np.int64)
    y = np.zeros(bs, np.int64)
    for i, j in enumerate(rng.choice(rows, bs, replace=False)):
        ne, nz = int(EL[j]), int(ZL[j])
        k = rng.integers(0, nz)
        s = np.zeros(ne + 2 + k, np.int64)
        s[:ne] = EN[j, :ne] + VZ
        s[ne] = SEP
        s[ne + 2:] = ZH[j, :k]
        x[i, SLEN - len(s):] = s
        y[i] = ZH[j, k]
    return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()


@torch.no_grad()
def evaluate(net, batches=6, bs=256, seed=7):
    rng = np.random.default_rng(seed)
    net.eval()
    c = n = 0
    for _ in range(batches):
        x, y = make_batch(VA, bs, rng)
        c += int((net(x).argmax(-1) == y).sum())
        n += bs
    net.train()
    return c / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log", type=int, default=2500)
    ap.add_argument("--eval", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="ckpt_tf.pt")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(a.seed)
    net = TF(d=a.d, layers=a.layers).cuda()
    model = torch.compile(net, mode="default") if a.compile else net
    print(f"params={sum(p.numel() for p in net.parameters())/1e6:.2f}M d={a.d} "
          f"layers={a.layers} slen={SLEN}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    rng = np.random.default_rng(a.seed)
    start = 1
    if a.resume and os.path.exists(a.ckpt):
        sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)
        net.load_state_dict(sd["model"])
        if "opt" in sd:
            opt.load_state_dict(sd["opt"])
        start = int(sd.get("step", 0)) + 1
        if "rng" in sd:
            rng.bit_generator.state = sd["rng"]
        print(f"resumed from step {start-1}", flush=True)
    t0 = time.time()
    for step in range(start, a.steps + 1):
        x, y = make_batch(TR, a.batch, rng)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % a.log == 0:
            print(f"{step:5d} | loss {loss.item():.4f} | {time.time()-t0:.0f}s", flush=True)
        if a.eval and step % a.eval == 0:
            print(f"  EVAL acc {evaluate(net):.4f}", flush=True)
            torch.save({"model": net.state_dict(), "args": vars(a), "opt": opt.state_dict(),
                        "step": step, "rng": rng.bit_generator.state}, a.ckpt)

    print(f"FINAL eval acc {evaluate(net):.4f} in {time.time()-t0:.0f}s", flush=True)
    torch.save({"model": net.state_dict(), "args": vars(a), "opt": opt.state_dict(),
                "step": a.steps, "rng": rng.bit_generator.state}, a.ckpt)


if __name__ == "__main__":
    main()
