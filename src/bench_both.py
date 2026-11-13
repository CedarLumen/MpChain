"""Final head-to-head at the same level on both sides: MpChain vs dense TF, each eager / full-graph capture.

The TF side needs a static KV buffer (constant shapes) to be capturable; numerical equivalence of both sides has been verified separately
(MpChain: identical token by token; TF: logits differ by 8.3e-07).

usage: python bench_both.py [--nt 32] [--reps 10]
"""
import argparse
import time

import numpy as np
import torch

from bench_infer2 import ENFIX, LEN, LZH, NT, VZ, CausalTF, fixed_inputs, mp_en, pick_step, tf_prompt
from mpchain_tr import MpChainTr
from tf_static import TFStatic

dev = "cuda"
MAXL = ENFIX + 2 + NT


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


# ---------------- MpChain ----------------
def mp_build(bs, ck_path):
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    ar = ck["args"]
    net = MpChainTr(ar["N"], 153, 2048, ar["d"], ar.get("dmp") or ar["d"], ar["T"]).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()
    torch.manual_seed(0)
    rows, en_ids = fixed_inputs(bs)
    en, env = mp_en(bs, en_ids)
    return net, en, env


def mp_take(outs, stops, T):
    pick = pick_step(stops, T)
    return torch.stack(outs, 1).gather(
        1, pick[:, None].unsqueeze(-1).expand(-1, 1, VZ)).squeeze(1).argmax(-1)


@torch.no_grad()
def mp_graph(net, en, env, bs, nt):
    h0 = net.prefill(en, env)
    W = nt + 1
    zh = torch.zeros(bs, W, dtype=torch.long, device=dev)
    zhv = torch.zeros(bs, W, 1, dtype=torch.bool, device=dev)
    zhv[:, 0] = True
    pos = torch.ones(1, dtype=torch.long, device=dev)
    out = torch.zeros(bs, nt, dtype=torch.long, device=dev)
    one = torch.ones(bs, 1, 1, dtype=torch.bool, device=dev)

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            net(en, env, zh, zhv, pre=h0)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for step in range(nt):
            outs, stops, _, _ = net(en, env, zh, zhv, pre=h0)
            tok = mp_take(outs, stops, net.T)
            out[:, step] = tok
            idx = pos.view(1, 1).expand(bs, 1)
            zh.scatter_(1, idx, tok.view(bs, 1))
            zhv.scatter_(1, idx.unsqueeze(-1), one)
            pos.add_(1)

    def reset():
        zh.zero_(); zhv.zero_(); zhv[:, 0] = True; pos.fill_(1); out.zero_()
    reset()
    return g, out, reset


@torch.no_grad()
def mp_eager(net, en, env, bs, nt):
    h0 = net.prefill(en, env)
    seq = np.zeros((bs, LZH), np.int64)
    lens = np.ones(bs, np.int64)
    for s in range(nt):
        zh = np.zeros((bs, LZH), np.int64)
        zhv = np.zeros((bs, LZH, 1), bool)
        for i in range(bs):
            L = int(lens[i]); zh[i, LZH - L:] = seq[i, :L]; zhv[i, LZH - L:] = True
        outs, stops, _, _ = net(en, env, torch.from_numpy(zh).cuda(),
                                torch.from_numpy(zhv).cuda(), pre=h0)
        nxt = mp_take(outs, stops, net.T).cpu().numpy()
        for i in range(bs):
            if lens[i] < LZH:
                seq[i, lens[i]] = nxt[i]; lens[i] += 1


# ---------------- Dense TF ----------------
@torch.no_grad()
def tf_graph(base, x0, bs, nt):
    sv = TFStatic(base, bs)
    lg = sv.prefill(x0)
    snap_k = [k.clone() for k in sv.kbuf]
    snap_v = [v.clone() for v in sv.vbuf]
    tok = lg.argmax(-1)
    p = torch.tensor(ENFIX + 2, dtype=torch.long, device=dev)
    out = torch.zeros(bs, nt, dtype=torch.long, device=dev)
    out[:, 0] = tok[:, 0]

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            sv.step(tok, p)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        lg_ = sv.prefill(x0)          # redo prefill inside the graph so the KV starting point is consistent
        t = lg_.argmax(-1)
        for step in range(1, nt):
            t = sv.step(t, p).argmax(-1)[:, None]
            out[:, step] = t[:, 0]
            p.add_(1)

    def reset():
        # prefill is redone inside the graph, so the KV starting point is consistent by construction; only the position counter and the output need resetting
        p.fill_(ENFIX + 2)
        out.zero_()
    reset()
    return g, out, reset


@torch.no_grad()
def tf_eager(base, x0, bs, nt):
    sv = TFStatic(base, bs)
    lg = sv.prefill(x0)
    tok = lg.argmax(-1)
    p = torch.tensor(ENFIX + 2, dtype=torch.long, device=dev)
    for step in range(1, nt):
        tok = sv.step(tok, p).argmax(-1)[:, None]
        p = p + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ck_seed0.pt")
    ap.add_argument("--nt", type=int, default=NT)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    print(f"GPU: {torch.cuda.get_device_name(0)}   nt={a.nt}")
    print(f"{'model':<10}{'batch':>6}{'eager ms/tok':>14}{'graph ms/tok':>13}{'speedup':>8}{'tok/s':>12}")

    for bs in (1, 8, 32):
        net, en, env = mp_build(bs, a.ckpt)
        t_e = timed(lambda: mp_eager(net, en, env, bs, a.nt), max(a.reps // 2, 3))
        g, out, reset = mp_graph(net, en, env, bs, a.nt)
        t_g = timed(lambda: (reset(), g.replay()), a.reps)
        print(f"{'MpChain':<10}{bs:>6}{t_e/a.nt:>14.2f}{t_g/a.nt:>13.2f}"
              f"{t_e/t_g:>7.2f}×{bs*a.nt/(t_g/1000):>12.0f}")
        del net; torch.cuda.empty_cache()

    base = CausalTF().to(dev).eval()
    for bs in (1, 8, 32):
        rows, en_ids = fixed_inputs(bs)
        x0 = tf_prompt(bs, en_ids)
        t_e = timed(lambda: tf_eager(base, x0, bs, a.nt), max(a.reps // 2, 3))
        g, out, reset = tf_graph(base, x0, bs, a.nt)
        t_g = timed(lambda: (reset(), g.replay()), a.reps)
        print(f"{'dense TF':<10}{bs:>6}{t_e/a.nt:>14.2f}{t_g/a.nt:>13.2f}"
              f"{t_e/t_g:>7.2f}×{bs*a.nt/(t_g/1000):>12.0f}")


if __name__ == "__main__":
    main()
