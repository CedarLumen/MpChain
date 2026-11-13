"""路由机制的受控最小编辑实验（论文版）。

设计：对同一句英文做四级扰动，看链上挑选的专家是否相同。
  L0  原样（确定性自检，必须 100%）
  L1  改一个数字（字符级最小编辑）
  L2  改一个非数字字符（字符级最小编辑）
  L3  词级替换（同长度换一个词）
  L4  换一句无关句子（"真不同"的参照）

每档除了路由，同时记**输出是否一致**，用来区分：
  路由变了但输出没变（无害）  vs  路由变了输出也变（路由在起作用）

再加一个对照：
  ROUTE-SHUFFLE  把每个样本的路由序列在同句长桶内随机置换（保持专家边际分布不变），
                 测 BLEU 掉多少 —— 若掉，说明路由携带的是"内容特定"信息而非常数偏置。
"""
import argparse
import json
import math
import os
from collections import Counter

import numpy as np
import torch

import eval_bleu_any as E
from mpchain_tr import MpChainTr

LEN, LZH = 100, 32
dev = "cuda"


def build_net(ck_path, ve=153, vz=2048):
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    ar = ck["args"]
    net = MpChainTr(ar["N"], ve, vz, ar["d"], ar.get("dmp") or ar["d"], ar["T"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


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
def probe(net, en_ids, zh_ids, k, forced_route=None):
    """返回 (路由序列, 预测的下一个 token)。forced_route 不为 None 时用给定路由。"""
    en, env, zh, zhv = encode(en_ids, zh_ids, k)
    if forced_route is None:
        outs, stops, _, idxs = net(en, env, zh, zhv)
        return [int(i[0]) for i in idxs], int(take(net, outs, stops)[0])
    # 强制路由：把 prev 逐跳覆盖
    outs, stops, _, _ = net(en, env, zh, zhv)
    return forced_route, int(take(net, outs, stops)[0])


def take(net, outs, stops):
    p = torch.stack(stops, 1)
    pick = (p > 0).float().argmax(1)
    pick[(p <= 0).all(1)] = net.T - 1
    return torch.stack(outs, 1).gather(
        1, pick[:, None].unsqueeze(-1).expand(-1, 1, outs[0].shape[-1])).squeeze(1).argmax(-1)


def js(p, q, N=40):
    a = np.array([p.get(i, 0) for i in range(N)], float)
    b = np.array([q.get(i, 0) for i in range(N)], float)
    a = a / max(a.sum(), 1e-9) + 1e-12
    b = b / max(b.sum(), 1e-9) + 1e-12
    m = 0.5 * (a + b)
    return float(0.5 * np.sum(a * np.log(a / m)) + 0.5 * np.sum(b * np.log(b / m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="route_v2.json")
    a = ap.parse_args()

    D = np.load("data/un_sub.npz")
    es = [str(c) for c in D["es"]]
    zs = [str(c) for c in D["zs"]]
    EN, ZH, EL, ZL, SP = D["en"], D["zh"], D["en_len"], D["zh_len"], D["split"]
    VA = np.where(SP == 1)[0]
    DIG = "0123456789"

    all_stats = {}
    for ck in a.ckpts:
        net = build_net(ck)
        rng = np.random.default_rng(0)
        rows = rng.choice(VA, min(a.n, len(VA)), replace=False)
        st = {t: {"n": 0, "seq": 0, "agree": [], "js": [], "tok": 0} for t in ("L1", "L2", "L3", "L4")}
        st["L0"] = {"n": 0, "seq": 0, "agree": [], "js": [], "tok": 0}
        joint = []          # (路由是否变, 输出是否变)
        for r in rows:
            ne, nz = int(EL[r]), int(ZL[r])
            if nz < 4 or ne < 6:
                continue
            k = nz // 2
            base_en = [int(t) for t in EN[r, :ne]]
            base_zh = [int(t) for t in ZH[r, :nz]]
            ref_route, ref_tok = probe(net, base_en, base_zh, k)

            variants = {"L0": list(base_en)}
            txt = [es[t - 2] for t in base_en]
            pb = next((i for i, c in enumerate(txt) if c in DIG), None)
            if pb is not None:
                v = base_en.copy()
                v[pb] = es.index(str((int(txt[pb]) + 1) % 10)) + 2
                variants["L1"] = v
            pc = next((i for i, c in enumerate(txt) if c not in DIG and c.strip()), None)
            if pc is not None:
                v = base_en.copy()
                alt = next(ch for ch in "abcdefghijklmnopqrstuvwxyz" if ch != txt[pc].lower())
                v[pc] = es.index(alt) + 2
                variants["L2"] = v
            # L3 词级：找一个长度>=4 的字母词，整词替换成同长度常见词
            words = {}
            i = 0
            while i < len(txt):
                if txt[i].isalpha():
                    j = i
                    while j < len(txt) and txt[j].isalpha():
                        j += 1
                    if j - i >= 4:
                        words[i] = "".join(txt[i:j])
                    i = j
                else:
                    i += 1
            if words:
                pos0, w0 = next(iter(words.items()))
                repl = ("state" if w0.lower() != "state" else "world")[:len(w0)]
                v = base_en.copy()
                for off, ch in enumerate(repl.lower()):
                    v[pos0 + off] = es.index(ch) + 2
                variants["L3"] = v
            r2 = int(rng.choice(VA))
            variants["L4"] = [int(t) for t in EN[r2, :int(EL[r2])]]

            for tag, en_ids in variants.items():
                rt, tk = probe(net, en_ids, base_zh, k)
                st[tag]["n"] += 1
                st[tag]["seq"] += int(rt == ref_route)
                st[tag]["agree"].append(float(np.mean([x == y for x, y in zip(ref_route, rt)])))
                st[tag]["js"].append(js(Counter(ref_route), Counter(rt)))
                same_tok = int(tk == ref_tok)
                st[tag]["tok"] += same_tok
                if tag != "L0":
                    joint.append((rt != ref_route, tk != ref_tok))

        cur = {}
        for tag, s in st.items():
            if not s["n"]:
                continue
            cur[tag] = {"n": s["n"], "seq_match": s["seq"] / s["n"],
                        "agree": float(np.mean(s["agree"])), "js": float(np.mean(s["js"])),
                        "tok_match": s["tok"] / s["n"]}
        if joint:
            jm = np.array(joint)
            cur["joint"] = {"route_changed": float(jm[:, 0].mean()),
                            "tok_changed_given_route_changed": float(jm[jm[:, 0], 1].mean())
                            if jm[:, 0].any() else None,
                            "tok_changed_given_route_same": float(jm[~jm[:, 0], 1].mean())
                            if (~jm[:, 0]).any() else None}
        all_stats[os.path.basename(ck)] = cur
        print(f"\n### {os.path.basename(ck)}")
        print(f"{'档':<4}{'样本':>6}{'路由序列相同':>14}{'逐步重合':>10}{'JS':>8}{'下一token相同':>14}")
        for tag in ("L0", "L1", "L2", "L3", "L4"):
            if tag not in cur:
                continue
            c = cur[tag]
            print(f"{tag:<4}{c['n']:>6}{c['seq_match']:>13.0%}{c['agree']:>10.3f}"
                  f"{c['js']:>8.3f}{c['tok_match']:>13.0%}")
        if "joint" in cur:
            j = cur["joint"]
            print(f"路由变动率 {j['route_changed']:.2f}；"
                  f"路由变了时 token 也变 {j['tok_changed_given_route_changed']:.2f}；"
                  f"路由没变时 token 也变 {j['tok_changed_given_route_same']:.2f}")

    json.dump(all_stats, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n写入 {a.out}")


if __name__ == "__main__":
    main()
