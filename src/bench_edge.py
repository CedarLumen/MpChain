"""Real-device (small device) experiment: laptop GPU latency + integrated board power + pure-CPU inference.

Why this exists: reviewers will ask "what does that edge advantage of yours look like on real hardware". Only three measurable things are measured here:

  A. per-token latency on a laptop GPU (RTX 5060 Laptop 8 GB), both sides at the same level (full-graph capture);
  B. board power sampled with nvidia-smi during one continuous decoding loop, integrated to give joules per token;
     idle power is recorded too, to state the protocol (not subtracted, readers can judge for themselves);
  C. per-token latency on pure CPU (4 threads, no GPU) -- edge devices usually have no dedicated accelerator.

Protocol matches BENCH_BOTH in docs: batch=1, decode 32 tokens, fp32, median; energy integrated over a continuous loop,
not from a single 2 ms instantaneous sample (a 100 ms sampling period cannot capture that).

usage: python bench_edge.py [--ckpt ck_seed0.pt]
output: reports/EDGE_LOCAL.txt (UTF-8, written by the script itself)
"""
import argparse
import os
import statistics
import subprocess
import threading
import time

import numpy as np
import torch

from bench_both import mp_build, mp_graph, mp_take, tf_graph, timed
from bench_infer2 import ENFIX, NT, VZ, CausalTF, fixed_inputs, tf_prompt
from tf_static import TFStatic

DEV = "cuda"
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "EDGE_LOCAL.txt")
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


class PowerSampler(threading.Thread):
    """Sample board power (W) every 100 ms."""

    def __init__(self, period_ms=100):
        super().__init__(daemon=True)
        self.period_ms = period_ms
        self.samples = []
        self.stop = False
        self.err = None
        self._proc = None

    def run(self):
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                 "-lms", str(self.period_ms)],
                stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in self._proc.stdout:
                if self.stop:
                    break
                try:
                    self.samples.append(float(line.strip().split()[0]))
                except (ValueError, IndexError):
                    continue
        except Exception as ex:                                   # noqa: BLE001
            self.err = str(ex)
        finally:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:                                  # noqa: BLE001
                    pass


def sample_idle(seconds=5.0):
    s = PowerSampler()
    s.start()
    time.sleep(seconds)
    s.stop = True
    time.sleep(0.3)
    return s.samples


def integrate(run_one_replay, tokens_per_replay, seconds=12.0):
    """Run continuously for `seconds` seconds; return (J/token, mean board power W, token count, wall clock s)."""
    s = PowerSampler()
    s.start()
    time.sleep(0.5)
    n = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        run_one_replay()
        n += tokens_per_replay
    torch.cuda.synchronize()
    dur = time.perf_counter() - t0
    s.stop = True
    time.sleep(0.3)
    pw = [x for x in s.samples if 5.0 <= x <= 300.0]
    if not pw:
        raise RuntimeError("no power samples collected (is nvidia-smi unavailable?): %s" % s.err)
    mean_w = statistics.fmean(pw)
    return mean_w * dur / n, mean_w, n, dur, len(pw)


# ---------------- CPU side ----------------
@torch.no_grad()
def mp_eager_cpu(net, en, env, nt):
    h0 = net.prefill(en, env)
    zh = torch.zeros(1, nt + 1, dtype=torch.long)
    zhv = torch.zeros(1, nt + 1, 1, dtype=torch.bool)
    zhv[:, 0] = True
    for step in range(nt):
        outs, stops, _, _ = net(en, env, zh, zhv, pre=h0)
        tok = mp_take(outs, stops, net.T)
        zh[:, step + 1] = tok
        zhv[:, step + 1] = True


@torch.no_grad()
def tf_static_cpu(base, x0, nt):
    sv = TFStatic(base, 1, device=torch.device("cpu"))
    lg = sv.prefill(x0)
    tok = lg.argmax(-1)
    for s in range(1, nt):
        tok = sv.step(tok, torch.tensor(ENFIX + 1 + s)).argmax(-1)[:, None]


def timed_cpu(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ck_seed0.pt")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--runs", type=int, default=3, help="how many rounds to repeat the A/B sections (to gauge noise)")
    ap.add_argument("--nt", type=int, default=NT)
    ap.add_argument("--cpu-threads", type=int, default=4)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    say("# Real-device experiment (small device)")
    say("# time %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    say("# device %s / %.1f GiB / torch %s" % (
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / 2**30, torch.__version__))
    say("# protocol batch=1, decode %d tokens, fp32, median; energy = integrated board power / token count" % a.nt)
    say("")

    # ---------- A/B. GPU latency + integrated power (several rounds, report the range) ----------
    net, en, env = mp_build(1, a.ckpt)
    g_mp, out_mp, reset_mp = mp_graph(net, en, env, 1, a.nt)
    # the dense side matches Section 6 of the paper: same architecture / same shapes (12 layers d384 ffn1536), weights do not affect timing cost
    base = CausalTF().to(DEV).eval()
    rows, en_ids = fixed_inputs(1)
    x0 = tf_prompt(1, en_ids)
    g_tf, out_tf, reset_tf = tf_graph(base, x0, 1, a.nt)

    lat_mp, lat_tf, e_mp, e_tf, pw_mp, pw_tf = [], [], [], [], [], []
    idle = sample_idle(5.0)
    say("[B] idle board power %.1f W (%d samples, 5 s)" % (statistics.fmean(idle), len(idle)))
    say("")
    say("run  MpChain ms/tok   dense ms/tok   ratio   MpChain W   dense W   MpChain J/tok  dense J/tok  energy ratio")
    for r in range(a.runs):
        t1 = timed(lambda: (reset_mp(), g_mp.replay()), a.reps)
        t2 = timed(lambda: (reset_tf(), g_tf.replay()), a.reps)
        j1, w1, n1, d1, k1 = integrate(lambda: (reset_mp(), g_mp.replay()), a.nt)
        j2, w2, n2, d2, k2 = integrate(lambda: (reset_tf(), g_tf.replay()), a.nt)
        lat_mp.append(t1 / a.nt); lat_tf.append(t2 / a.nt)
        e_mp.append(j1); e_tf.append(j2); pw_mp.append(w1); pw_tf.append(w2)
        say("%2d %14.2f %14.2f %7.2f× %10.1f %9.1f %13.3f %12.3f %8.2f×"
            % (r + 1, t1 / a.nt, t2 / a.nt, t1 / t2, w1, w2, j1, j2, j1 / j2))
    say("")
    say("[summary] latency MpChain %.2f ms/token (%.2f–%.2f), dense %.2f (%.2f–%.2f), ratio %.2f–%.2f"
        % (statistics.median(lat_mp), min(lat_mp), max(lat_mp),
           statistics.median(lat_tf), min(lat_tf), max(lat_tf),
           min(m1 / m2 for m1, m2 in zip(lat_mp, lat_tf)),
           max(m1 / m2 for m1, m2 in zip(lat_mp, lat_tf))))
    say("[summary] energy MpChain %.3f J/token (%.3f–%.3f), dense %.3f (%.3f–%.3f); energy ratio MpChain/dense %.2f–%.2f"
        % (statistics.median(e_mp), min(e_mp), max(e_mp),
           statistics.median(e_tf), min(e_tf), max(e_tf),
           min(m1 / m2 for m1, m2 in zip(e_mp, e_tf)),
           max(m1 / m2 for m1, m2 in zip(e_mp, e_tf))))
    say("[summary] board power MpChain %.1f W, dense %.1f W (sampled in the same rounds)"
        % (statistics.fmean(pw_mp), statistics.fmean(pw_tf)))
    say("")
    t_mp, t_tf = statistics.median(lat_mp) * a.nt, statistics.median(lat_tf) * a.nt

    # ---------- C. Pure CPU ----------
    torch.set_num_threads(a.cpu_threads)
    say("[C] torch thread count %d, no GPU" % torch.get_num_threads())

    net_cpu = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ar = net_cpu["args"]
    from mpchain_tr import MpChainTr
    m_cpu = MpChainTr(ar["N"], 153, 2048, ar["d"], ar.get("dmp") or ar["d"], ar["T"]).eval()
    m_cpu.load_state_dict(net_cpu["model"])
    en_c, env_c = en.cpu(), env.cpu()
    t_cpu_mp = timed_cpu(lambda: mp_eager_cpu(m_cpu, en_c, env_c, a.nt), reps=5)
    say("[C] MpChain  %.1f ms/token (%.2f s / %d tokens, median of 5)"
        % (t_cpu_mp / a.nt, t_cpu_mp / 1000, a.nt))

    base_cpu = CausalTF().eval()
    base_cpu.load_state_dict(base.state_dict())
    x0_cpu = x0.cpu()
    t_cpu_tf = timed_cpu(lambda: tf_static_cpu(base_cpu, x0_cpu, a.nt), reps=5)
    say("[C] dense TF %.1f ms/token (%.2f s / %d tokens, median of 5)"
        % (t_cpu_tf / a.nt, t_cpu_tf / 1000, a.nt))
    say("[C] ratio %.2f×" % (t_cpu_mp / t_cpu_tf))
    say("")

    # ---------- Weights ----------
    w_ours = sum(p.numel() for p in m_cpu.parameters()) * 4 / 2**20
    w_dense = sum(p.numel() for p in base_cpu.parameters()) * 4 / 2**20
    say("[D] weights fp32: MpChain %.0f MB / dense %.0f MB" % (w_ours, w_dense))
    say("[D] note: the dense side is timed with randomly initialised weights of the same architecture and shapes (cost is independent of the weights), consistent with Section 6 of the paper")

    open(REPORT, "w", encoding="utf-8").write("\n".join(LINES) + "\n")
    print("\nwrote %s" % REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
