"""Verify on the main configuration (N=40) that "the parameter shapes do not contain T": the same weights are instantiated directly with a larger T.

Why this script is needed: Section 3.5 of the main text says "weights trained with T = 6 can be instantiated directly with a larger T",
but reports/ only had the conclusion for the early configuration (N=100), with no artifact. Here, for the three seeds of the main configuration and
T ∈ {4, 6, 8, 10}, each runs the same 500 sentences (seed=11, the same batch of sentences for every T), writing
BLEU-2 / p1 / p2 / token accuracy / mean steps / per-sentence agreement with T=6 into reports/EXTEND_T.txt.

T=4 is an intentional negative control: truncation has to lose points, otherwise "a larger T changes nothing" would be a vacuous claim.

python eval_extend_t.py
"""
import os

import numpy as np
import torch

from eval_bleu_any import VA, VZ, VE, ZL, ZH, corpus_bleu, generate_mp, to_str
from mpchain_tr import MpChainTr

CKPTS = [
    ("seed0", "ck_seed0.pt"),
    ("seed1", "ck_seed1.pt"),
    ("seed2", "ck_seed2.pt"),
]
TS = [4, 6, 8, 10]
N, SEED = 500, 11
OUT = "results/EXTEND_T.txt"


@torch.no_grad()
def run(path, T, rows):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    a = ck["args"]
    net = MpChainTr(N=a["N"], Ve=VE, Vz=VZ, d=a["d"], dmp=a.get("dmp", 0) or a["d"], T=T).cuda()
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, "shape mismatch at T=%d: %s %s" % (T, missing[:5], unexpected[:5])
    net.eval()
    gen, tl, steps = generate_mp(net, rows)
    hyps = [to_str(gen[i][:tl[i]]) for i in range(N)]
    refs = [to_str(ZH[j, :int(ZL[j])]) for j in rows]
    b, ps = corpus_bleu(hyps, refs, 2)
    nt = sum(int(tl[i]) for i in range(N))
    hit = sum(1 for i in range(N) for k in range(int(tl[i])) if gen[i][k] == ZH[rows[i], k])
    return dict(bleu=b, p1=ps[0], p2=ps[1], acc=hit / max(nt, 1),
                steps=float(steps.mean()), hyps=hyps)


def main():
    rows = np.random.default_rng(SEED).choice(VA, N, replace=False)
    lines = [
        "# main configuration (N=40) re-instantiated with a different T: the same weights, no retraining, no re-initialisation",
        "# %d-sentence test subset (seed=%d, the same batch of sentences for every T); T=4 is the truncation control" % (N, SEED),
        "",
        "ckpt     T   BLEU-2      p1      p2   tok_acc   mean steps   same as T=6",
    ]
    print(lines[0])
    for name, path in CKPTS:
        cur = {T: run(path, T, rows) for T in TS}
        for T in TS:
            m = cur[T]
            same = (sum(1 for x, y in zip(m["hyps"], cur[6]["hyps"]) if x == y) / N) if T != 6 else 1.0
            lines.append("%-8s %2d   %.4f  %.3f  %.3f   %.4f    %.2f      %5.1f%%"
                         % (name, T, m["bleu"], m["p1"], m["p2"], m["acc"], m["steps"], 100 * same))
            print("%-8s %2d   %.4f  %.4f  steps %.2f"
                  % (name, T, m["bleu"], m["acc"], m["steps"]), flush=True)
        base = cur[6]
        lines.append("         ΔBLEU-2 vs T=6: T=4 %+.4f / T=8 %+.4f / T=10 %+.4f"
                     % (cur[4]["bleu"] - base["bleu"], cur[8]["bleu"] - base["bleu"],
                        cur[10]["bleu"] - base["bleu"]))
        lines.append("")
    for T in TS:
        vals = []
        for name, path in CKPTS:
            vals.append(None)
        pass
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
