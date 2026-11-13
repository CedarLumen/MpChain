"""给两个模型加同一条序列停止规则（首个句号/分号截断）后重算 BLEU。

动机：两个模型都没有 EOS，按固定 32 字符生成，句子结束后继续吐高频字符，
这会主导 BLEU 的精确率。加同一个后处理截断，才是可比的"翻译质量"。
"""
import numpy as np
import torch

import eval_bleu_any as E
from eval_tf_bleu import generate_tf
from mpchain_tr import MpChainTr
from train_tf import TF

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

STOP = set("。！？；")


def cut(s):
    for i, ch in enumerate(s):
        if ch in STOP:
            return s[:i + 1]
    return s


def score(hyps, refs, tag):
    b_raw, _ = E.corpus_bleu(hyps, refs, 2)
    b_cut, _ = E.corpus_bleu([cut(h) for h in hyps], [cut(r) for r in refs], 2)
    print(f"{tag:<28} BLEU-2 原样 {b_raw:.4f}   截断后 {b_cut:.4f}")


rng = np.random.default_rng(11)
rows = rng.choice(E.VA, 500, replace=False)
refs = [E.to_str(E.ZH[j, :int(E.ZL[j])]) for j in rows]

ck = torch.load(r"E:\MpChain_ckpt\20261113_tf\ck_tf100k.pt", map_location=dev, weights_only=False)
tf = TF(d=384, layers=12).to(dev); tf.load_state_dict(ck["model"]); tf.eval()
g, tl = generate_tf(tf, rows)
score([E.to_str(g[i][:tl[i]]) for i in range(len(rows))], refs, "稠密 TF 100k")
del tf; torch.cuda.empty_cache()

for name, path in (("seed0", r"E:\MpChain_ckpt\20261112_100k\ck_seed0.pt"),
                   ("seed1", r"E:\MpChain_ckpt\20261112_100k\ck_seed1.pt"),
                   ("seed2", r"E:\MpChain_ckpt\20261112_100k\ck_seed2.pt")):
    ck2 = torch.load(path, map_location=dev, weights_only=False)
    ar = ck2["args"]
    mp = MpChainTr(ar["N"], 153, 2048, ar["d"], ar.get("dmp") or ar["d"], ar["T"]).to(dev)
    mp.load_state_dict(ck2["model"]); mp.eval()
    gm, tlm, _ = E.generate_mp(mp, rows)
    score([E.to_str(gm[i][:tlm[i]]) for i in range(len(rows))], refs, f"本文 {name}")
    del mp; torch.cuda.empty_cache()
