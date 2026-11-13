"""Single-sequence inference timing on a Raspberry Pi (ARM SBC): the same weights, this method vs the dense baseline, pure CPU.

Usage (on the Pi, run from the same directory as this file):
    python3 pi_bench.py                     # 32 tokens, all cores, 5 runs
    python3 pi_bench.py --threads 1         # single thread (to see whether operator scheduling is the bottleneck)
    python3 pi_bench.py --nt 16 --reps 7

Output: reports/PI_LOCAL.txt (UTF-8, can be copied straight back)

Protocol matches Section 6.4 of the paper: batch=1, fp32, greedy decoding, per-token latency taken as the median;
the dense side uses a static KV buffer with the same architecture and shapes (12 layers d=384, ffn 1536), weights do not affect timing cost
(consistent with the timing protocol of Section 6 of the paper).
"""
import argparse
import os
import platform
import time

try:                                  # resource only exists on Unix-like systems; running on Windows is for local self-checks
    import resource
except ImportError:
    resource = None

HERE = os.path.dirname(os.path.abspath(__file__))
# the dataset must be pointed at before importing bench_infer2: it reads the npz at module load time
os.environ.setdefault("MPCHAIN_DATA", os.path.join(HERE, "data", "un_sub.npz"))

import numpy as np                     # noqa: E402
import torch                           # noqa: E402

from bench_infer2 import (CausalTF, ENFIX, LEN, NT, SEP, VZ,  # noqa: E402
                          fixed_inputs)
from mpchain_tr import MpChainTr       # noqa: E402
from tf_static import TFStatic         # noqa: E402

OUT = os.path.join(HERE, "reports", "PI_LOCAL.txt")
_LINES = []


def say(s=""):
    print(s, flush=True)
    _LINES.append(s)


def pick_step(stops, T):
    """The step at which the first stop fires (logit > 0); if none fires, take T-1. Same logic as eval_bleu_any.py."""
    p = torch.stack(stops, 1)
    pick = (p > 0).float().argmax(1)
    pick[(p <= 0).all(1)] = T - 1
    return pick


def mp_take(outs, stops, T):
    pick = pick_step(stops, T)
    return torch.stack(outs, 1).gather(
        1, pick[:, None].unsqueeze(-1).expand(-1, 1, VZ)).squeeze(1).argmax(-1)


def _t(x):
    """Kept for compatibility: move tensors to CPU, convert numpy arrays straight into tensors."""
    t = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    return t.cpu() if torch.is_tensor(t) else t


def build_mp_inputs(en_ids):
    """Same protocol as bench_infer2.mp_en: right-align en into the LEN=100 window (that one hard-codes .cuda())."""
    en = np.zeros((1, LEN), np.int64)
    env = np.zeros((1, LEN, 1), bool)
    n = min(ENFIX, LEN)
    en[:, LEN - n:] = en_ids[:, :n]
    env[:, LEN - n:] = True
    return torch.from_numpy(en), torch.from_numpy(env)


def build_tf_prompt(en_ids):
    """Same protocol as bench_infer2.tf_prompt."""
    x = np.zeros((1, ENFIX + 2), np.int64)
    x[:, :ENFIX] = en_ids + VZ
    x[:, ENFIX] = SEP
    x[:, ENFIX + 1] = 0
    return torch.from_numpy(x)


@torch.no_grad()
def decode_ours(net, en, env, nt):
    en, env = _t(en), _t(env)
    h0 = net.prefill(en, env)
    zh = torch.zeros(1, nt + 1, dtype=torch.long)
    zhv = torch.zeros(1, nt + 1, 1, dtype=torch.bool)
    zhv[:, 0] = True
    toks = []
    for step in range(nt):
        outs, stops, _, _ = net(en, env, zh, zhv, pre=h0)
        tok = mp_take(outs, stops, net.T)
        toks.append(int(tok))
        zh[:, step + 1] = tok
        zhv[:, step + 1] = True
    return toks


@torch.no_grad()
def decode_dense(base, x0, nt):
    sv = TFStatic(base, 1, device=torch.device("cpu"))
    lg = sv.prefill(x0)
    tok = lg.argmax(-1)
    toks = [int(tok)]
    for s in range(1, nt):
        tok = sv.step(tok, torch.tensor(ENFIX + 1 + s)).argmax(-1)[:, None]
        toks.append(int(tok))
    return toks


def timed(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), min(ts), max(ts)


def meminfo():
    try:
        with open("/proc/meminfo") as f:
            d = {k.strip(): v for k, v in (l.split(":", 1) for l in f if ":" in l)}
        tot = int(d["MemTotal"].split()[0]) / 1024 ** 2
        avail = int(d["MemAvailable"].split()[0]) / 1024 ** 2
        return tot, avail
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ck_seed0_slim.pt"))
    ap.add_argument("--nt", type=int, default=NT)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="0 = use all cores")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.set_float32_matmul_precision("highest")

    tot, avail = meminfo()
    say("# Raspberry Pi on-device timing (pure CPU, batch=1)")
    say("# time %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    say("# machine %s / %s / %s" % (platform.machine(), platform.system(), platform.platform()))
    say("# Python %s / torch %s / numpy %s" % (platform.python_version(), torch.__version__, np.__version__))
    say("# memory %.2f GiB total / %.2f GiB available; torch thread count %d" % (tot, avail, torch.get_num_threads()))
    say("# protocol: decode %d tokens, fp32, greedy, median of %d runs; CUDA available = %s"
        % (a.nt, a.reps, torch.cuda.is_available()))
    say("")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ar = ck["args"]
    net = MpChainTr(ar["N"], 153, 2048, ar["d"], ar.get("dmp") or ar["d"], ar["T"]).eval()
    net.load_state_dict(ck["model"])
    w_ours = sum(p.numel() for p in net.parameters()) * 4 / 2 ** 20

    rows, en_ids = fixed_inputs(1)
    en, env = build_mp_inputs(en_ids)
    toks = decode_ours(net, en, env, a.nt)
    say("[check] first 8 tokens of this method: %s" % toks[:8])

    t_ours, lo1, hi1 = timed(lambda: decode_ours(net, en, env, a.nt), a.reps)
    say("[A] this method   %.1f ms/token (%.1f–%.1f)  ≈ %.1f token/s"
        % (t_ours / a.nt, lo1 / a.nt, hi1 / a.nt, 1000.0 * a.nt / t_ours))

    base = CausalTF().eval()
    x0 = build_tf_prompt(en_ids)
    toks2 = decode_dense(base, x0, a.nt)
    say("[check] first 8 tokens of the dense baseline: %s" % toks2[:8])

    t_dense, lo2, hi2 = timed(lambda: decode_dense(base, x0, a.nt), a.reps)
    say("[B] dense baseline %.1f ms/token (%.1f–%.1f)  ≈ %.1f token/s"
        % (t_dense / a.nt, lo2 / a.nt, hi2 / a.nt, 1000.0 * a.nt / t_dense))
    say("")
    say("[summary] this method/dense = %.2f× (>1 means this method is slower)" % (t_ours / t_dense))
    say("[summary] weights fp32: this method %.0f MB; dense %.0f MB"
        % (w_ours, sum(p.numel() for p in base.parameters()) * 4 / 2 ** 20))
    rss = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0) if resource else float("nan")
    say("[summary] peak RSS %.0f MB (resource.ru_maxrss, unit KB)" % rss)
    say("")
    say("# note: the dense side uses randomly initialised weights (timing cost is independent of the weights, same protocol as Section 6 of the paper),")
    say("#       so the tokens it generates are meaningless; this method loads a real ckpt, so its tokens are meaningful.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES) + "\n")
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
