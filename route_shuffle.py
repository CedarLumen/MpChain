"""路由打乱对照：路由携带的是"内容信息"还是"边际分布"？

做法：同一批句子解码两次。
  own       用模型自己算出的路由（等同于常规贪心解码，作为基线）
  shuffled  把每条句子的路由计划在同句长桶内随机置换后强制使用
            （保持专家使用频次不变，只打断"哪条句子配哪条路由"的对应）
  random    每步独立均匀随机挑专家（连边际分布都破坏）

若 shuffled 明显低于 own，则路由携带的是与内容绑定的信息，而不是"专家使用
频率"这类边际统计。
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
    """mode: own / forced。返回 (生成, 长度, 每个 sample 在第一步用的路由计划)。"""
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
    lens = np.ones(B, np.int64)          # 与 eval_bleu_any 一致：前缀从 1 个 pad 开始
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
        gen[:, s] = nxt                      # 输出按步号存
        for i in range(B):
            if lens[i] < maxlen:
                pre[i, lens[i]] = nxt[i]     # 前缀按长度存（第 0 位是起始 pad）
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
    zl = [int(E.ZL[j]) for j in rows]      # 与 eval_bleu_any 同一口径：hyp 截断到参考长度

    g0, l0, plans = decode(net, rows, "own", T=net.T)
    hyps0 = [E.to_str(g0[i][:zl[i]]) for i in range(len(rows))]
    b_own, _ = E.corpus_bleu(hyps0, refs, 2)
    acc_own = sum(1 for i in range(len(rows)) for k in range(zl[i])
                  if g0[i][k] == E.ZH[rows[i], k]) / sum(zl)

    # 同句长桶内置换路由计划
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

    print(f"ckpt {a.ckpt}  句数 {len(rows)}  "
          f"（置换中 {changed:.0%} 的句子换了至少一步路由）")
    print(f"{'条件':<26}{'BLEU-2':>9}{'tok acc':>10}{'相对 own':>10}")
    for tag, b, ac in (("own（模型自选）", b_own, acc_own),
                       ("shuffled（同长度桶内置换）", b_sh, acc_sh),
                       ("random（每步均匀随机）", b_rd, acc_rd)):
        print(f"{tag:<26}{b:>9.4f}{ac:>10.4f}{b/b_own:>10.3f}")


if __name__ == "__main__":
    main()
