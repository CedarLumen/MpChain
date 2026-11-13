"""Compute BLEU-2 under the same protocol for dense TF ckpts (the same 500 sentences as eval_bleu_any)."""
import argparse
import os

import numpy as np
import torch

import eval_bleu_any as E
from train_tf import SEP, SLEN, VZ, TF

dev = "cuda"


@torch.no_grad()
def generate_tf(net, rows, maxlen=32, ban=(0, 1)):
    B = len(rows)
    ne = [int(E.EL[j]) for j in rows]
    pre = np.zeros((B, maxlen), np.int64)
    lens = np.zeros(B, np.int64)
    gen = np.zeros((B, maxlen), np.int64)
    for s in range(maxlen):
        x = np.zeros((B, SLEN), np.int64)
        for i, j in enumerate(rows):
            n = ne[i]
            L = int(lens[i])
            seg = np.zeros(n + 2 + L, np.int64)
            seg[:n] = E.EN[j, :n] + VZ
            seg[n] = SEP
            seg[n + 2:] = pre[i, :L]
            x[i, SLEN - len(seg):] = seg
        lg = net(torch.from_numpy(x).cuda())
        for b in ban:
            lg[:, b] = -1e9
        nxt = lg.argmax(-1).cpu().numpy()
        for i in range(B):
            if lens[i] < maxlen:
                gen[i, lens[i]] = nxt[i]
                pre[i, lens[i]] = nxt[i]
                lens[i] += 1
    return gen, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ar = ck.get("args", {})
    net = TF(d=ar.get("d", 384), layers=ar.get("layers", 12)).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()

    rng = np.random.default_rng(11)          # exactly the same batch of sentences as eval_bleu_any
    rows = rng.choice(E.VA, 500, replace=False)
    gen, tl = generate_tf(net, rows)
    hyps = [E.to_str(gen[i][:tl[i]]) for i in range(len(rows))]
    refs = [E.to_str(E.ZH[j, :int(E.ZL[j])]) for j in rows]
    b, ps = E.corpus_bleu(hyps, refs, 2)
    hit = sum(1 for i in range(len(rows)) for k in range(int(tl[i]))
              if gen[i][k] == E.ZH[rows[i], k])
    nt = int(sum(tl))
    print(f"{os.path.basename(a.ckpt)}  arch=tf  step={ck.get('step')}  "
          f"BLEU-2 {b:.4f}  p1 {ps[0]:.3f}  p2 {ps[1]:.3f}  tok_acc {hit/nt:.4f}")


if __name__ == "__main__":
    main()
