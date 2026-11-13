"""推理效率 benchmark（严格版）：MpChain vs 稠密 Transformer。

修掉了 bench_infer.py 的四个问题：
  1) 原来 dt=(总时间)/reps 但只跑了一遍 ⇒ 数字虚小 3 倍；
  2) 没有 warmup，首次编译/分配全算进去；
  3) Transformer 基线没有 KV cache，等于拿"每次都整段重算"去比；
  4) 显存只测了 reset 之后的峰值，不含权重本身。

口径（全部固定形状，避免 padding 干扰计时）：
  en 固定 30 字符；解码 32 个 token；batch ∈ {1,8,32}；fp32。
  首 token 延迟 = prefill（编码英文 + 生成第 1 个 token）
  每 token 延迟 = 后续每个 token 的墙钟（含 Python 调度，故另给纯 GPU 时间）
  吞吐 = batch × 32 / 总时间

模型：
  mpchain        总参 ~22.9M / 激活 ~0.56M（N40 d128 T6）
  causal_tf_kv   同规模因果 Transformer（12 层 d384 ffn1536）+ KV cache（标准高效部署）
  causal_tf_nokv 同架构但每步整段重算（对照"不用 cache 会多贵"）
  tf_enc        仓库原有 encoder 式基线 ckpt_tf15k.pt（只能整段重算）

python bench_infer2.py [--reps 20] [--quick]
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpchain_tr import MpChainTr

LEN, LZH = 100, 32
NT = 32
ENFIX = 30
BATCHES = [1, 8, 32]
D = np.load("data/un_sub.npz")
EN, ZH, EL, ZL, SP = D["en"], D["zh"], D["en_len"], D["zh_len"], D["split"]
VE, VZ = len(D["es"]) + 2, len(D["zs"]) + 2
SEP = VZ + VE
VT = VZ + VE + 1
VA = np.where(SP == 1)[0]


# ---------------- 因果 Transformer（可 KV cache） ----------------
class Block(nn.Module):
    def __init__(self, d, nhead, ffn):
        super().__init__()
        self.d, self.nhead = d, nhead
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, ffn), nn.Linear(ffn, d)

    def forward(self, x, cache=None, causal=False):
        B, L, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(d, dim=-1)
        if cache is not None:
            k = torch.cat([cache[0], k], 1)
            v = torch.cat([cache[1], v], 1)
        hd = d // self.nhead
        q = q.view(B, -1, self.nhead, hd).transpose(1, 2)
        kk = k.view(B, -1, self.nhead, hd).transpose(1, 2)
        vv = v.view(B, -1, self.nhead, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, kk, vv, is_causal=causal)
        a = a.transpose(1, 2).reshape(B, L, d)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x, (k, v)


class CausalTF(nn.Module):
    def __init__(self, d=384, layers=12, nhead=6, ffn=1536):
        super().__init__()
        self.emb = nn.Embedding(VT, d, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, LEN + LZH + 4, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, nhead, ffn) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VZ)

    def forward(self, x, caches=None, causal=False):
        h = self.emb(x) + self.pos[:, : x.shape[1]]
        new = []
        for i, blk in enumerate(self.blocks):
            h, c = blk(h, None if caches is None else caches[i], causal=causal)
            new.append(c)
        return self.head(self.norm(h)[:, -1]), new


# ---------------- 输入构造（固定形状） ----------------
def fixed_inputs(bs):
    rng = np.random.default_rng(0)
    rows = rng.choice(VA, bs, replace=False)
    en_ids = np.zeros((bs, ENFIX), np.int64)
    for i, j in enumerate(rows):
        ne = int(EL[j])
        seg = EN[j, :ne][:ENFIX]
        en_ids[i, :len(seg)] = seg           # 右侧补 0，长度固定
    return rows, en_ids


def mp_en(bs, en_ids):
    """MpChain 侧：en 右对齐放进 LEN=100 的窗口（与训练一致，只是截短到 30）。"""
    en = np.zeros((bs, LEN), np.int64)
    env = np.zeros((bs, LEN, 1), bool)
    n = min(ENFIX, LEN)
    en[:, LEN - n:] = en_ids[:, :n]
    env[:, LEN - n:] = True
    return torch.from_numpy(en).cuda(), torch.from_numpy(env).cuda()


def tf_prompt(bs, en_ids):
    x = np.zeros((bs, ENFIX + 2), np.int64)
    x[:, :ENFIX] = en_ids + VZ
    x[:, ENFIX] = SEP
    x[:, ENFIX + 1] = 0
    return torch.from_numpy(x).cuda()


def pick_step(stops, T):
    p = torch.stack(stops, 1)
    pick = (p > 0).float().argmax(1)
    pick[(p <= 0).all(1)] = T - 1
    return pick


def params_of(net):
    tot = sum(p.numel() for p in net.parameters())
    wb = sum(p.numel() * p.element_size() for p in net.parameters()) / 1024 ** 2
    return tot, wb


def timed(fn, reps, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return float(np.median(ts)), float(ts.min())


# ---------------- 三个 bench ----------------
@torch.no_grad()
def bench_mp(net, bs, reps, use_pre=False):
    rows, en_ids = fixed_inputs(bs)
    en, env = mp_en(bs, en_ids)
    T = net.T
    h0 = net.prefill(en, env) if use_pre else None

    def one_step(seq, lens):
        zh = np.zeros((bs, LZH), np.int64)
        zhv = np.zeros((bs, LZH, 1), bool)
        for i in range(bs):
            L = int(lens[i])
            zh[i, LZH - L:] = seq[i, :L]
            zhv[i, LZH - L:] = True
        outs, stops, _, _ = net(en, env, torch.from_numpy(zh).cuda(),
                                torch.from_numpy(zhv).cuda(), pre=h0)
        pick = pick_step(stops, T)
        lg = torch.stack(outs, 1).gather(
            1, pick[:, None].unsqueeze(-1).expand(-1, 1, VZ)).squeeze(1)
        return lg.argmax(-1)

    def fresh():
        return np.zeros((bs, LZH), np.int64), np.ones(bs, np.int64)

    torch.cuda.reset_peak_memory_stats()
    s0, l0 = fresh()
    first, _ = timed(lambda: one_step(*[s0, l0]), reps)

    def decode_all():
        seq, lens = fresh()
        for s in range(NT):
            nxt = one_step(seq, lens)
            for i in range(bs):
                if lens[i] < LZH:
                    seq[i, lens[i]] = nxt[i].item()
                    lens[i] += 1

    tot, _ = timed(decode_all, max(reps // 2, 3))
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    return first, (tot - first) / (NT - 1), bs * NT / (tot / 1000), peak


@torch.no_grad()
def bench_causal(net, bs, reps, use_kv=True):
    rows, en_ids = fixed_inputs(bs)
    x0 = tf_prompt(bs, en_ids)
    out = np.zeros((bs, NT), np.int64)

    def mark():
        # CUDA graph 下必须声明一步开始，否则缓存张量会被下一次运行覆写
        torch.compiler.cudagraph_mark_step_begin()

    def first_step():
        lg, caches = net(x0, None, causal=True)
        return lg.argmax(-1), caches

    torch.cuda.reset_peak_memory_stats()
    lg1, caches = None, None

    def warm():
        nonlocal lg1, caches
        lg1, caches = first_step()

    first, _ = timed(warm, reps)

    if use_kv:
        def decode_all():
            mark()
            lg, cs = net(x0, None, causal=True)
            nxt = lg.argmax(-1)[:, None]
            for s in range(1, NT):
                mark()
                lg2, cs = net(nxt, cs, causal=False)
                cs = tuple((k.clone(), v.clone()) for k, v in cs)
                nxt = lg2.argmax(-1)[:, None]
        reps_d = max(reps // 2, 3)
    else:
        def decode_all():
            x = x0.clone()
            for s in range(NT):
                mark()
                lg, _ = net(x, None, causal=True)
                nxt = lg.argmax(-1)[:, None]
                x = torch.cat([x, nxt], 1)
        reps_d = max(reps // 3, 2)
    tot, _ = timed(decode_all, reps_d)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    # 总时长里扣掉一次 prefill 的估计（用 first）
    return first, max(tot - first, 1e-6) / (NT - 1), bs * NT / (tot / 1000), peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="ck_best.pt")
    ap.add_argument("--batches", type=str, default="")
    ap.add_argument("--opt", type=int, default=0,
                    help="1 = 用 torch.compile(mode=reduce-overhead) 做推理优化（两边同等对待）")
    ap.add_argument("--pre", type=int, default=0,
                    help="1 = 英文编码只算一次（prefill 缓存），仅对 MpChain 有效")
    ap.add_argument("--fast", type=int, default=0,
                    help="1 = 推理期只跑 batch 内真实前缀长度（数值等价，仅改计算量）")
    a = ap.parse_args()
    reps = 4 if a.quick else a.reps
    batches = [1] if a.quick else ([int(x) for x in a.batches.split(",")] if a.batches else BATCHES)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    def maybe_compile(net):
        if not a.opt:
            return net
        mode = "default" if a.opt == 1 else "reduce-overhead"
        return torch.compile(net, mode=mode, fullgraph=False)

    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"总显存 {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB  "
          f"opt={a.opt}"
          f"{'' if not a.opt else ' (compile default)' if a.opt == 1 else ' (compile reduce-overhead)'}",
          flush=True)
    print(f"{'模型':<14}{'batch':>6}{'首token ms':>12}{'每token ms':>12}"
          f"{'吞吐 tok/s':>13}{'权重MB':>9}{'峰值MB':>9}", flush=True)

    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    ar = ck["args"]
    mp = MpChainTr(N=ar["N"], Ve=VE, Vz=VZ, d=ar["d"], dmp=ar.get("dmp", 0) or ar["d"],
                   T=ar["T"]).cuda()
    mp.load_state_dict(ck["model"]); mp.eval()
    if a.fast:
        mp.opt_prefix = 1
    mp = maybe_compile(mp)
    tot, wb = params_of(mp)
    for bs in batches:
        f, per, tps, pk = bench_mp(mp, bs, reps, use_pre=bool(a.pre))
        tag = ('MpChain+pre' if a.pre else 'MpChain') + ('+fast' if a.fast else '')
        print(f"{tag:<14}{bs:>6}{f:>12.2f}{per:>12.2f}{tps:>13.1f}{wb:>9.0f}{pk:>9.0f}", flush=True)
    print(f"   (MpChain 总参 {tot/1e6:.2f}M)", flush=True)
    del mp; torch.cuda.empty_cache()

    tf = maybe_compile(CausalTF().cuda().eval())
    tot, wb = params_of(tf)
    print(f"   (CausalTF 总参 {tot/1e6:.2f}M  d=384 L=12 ffn=1536)", flush=True)
    for bs in batches:
        f, per, tps, pk = bench_causal(tf, bs, reps, use_kv=True)
        print(f"{'causal+KV':<14}{bs:>6}{f:>12.2f}{per:>12.2f}{tps:>13.1f}{wb:>9.0f}{pk:>9.0f}", flush=True)
    for bs in batches:
        f, per, tps, pk = bench_causal(tf, bs, reps, use_kv=False)
        print(f"{'causal noKV':<14}{bs:>6}{f:>12.2f}{per:>12.2f}{tps:>13.1f}{wb:>9.0f}{pk:>9.0f}", flush=True)


if __name__ == "__main__":
    main()
