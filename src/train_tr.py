"""Real translation: UN en-zh, character level, predict the next token from the English plus the Chinese prefix, with early stopping.

python train_tr.py --steps 30000 --T 10
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from mpchain_tr import MpChainTr, stop_loss
from win import Win

LEN, LZH = 100, 32
D = np.load(os.environ.get("MPCHAIN_DATA", "data/un_sub.npz"))
EN, ZH, EL, ZL, SP = D["en"], D["zh"], D["en_len"], D["zh_len"], D["split"]
VE, VZ = len(D["es"]) + 2, len(D["zs"]) + 2
TR = np.where(SP == 0)[0]
VA = np.where(SP == 1)[0]


def make_batch(rows, bs, rng):
    idx = rng.choice(rows, bs, replace=False)
    en = np.zeros((bs, LEN), np.int64)
    zh = np.zeros((bs, LZH), np.int64)
    env = np.zeros((bs, LEN, 1), bool)
    zhv = np.zeros((bs, LZH, 1), bool)
    y = np.zeros(bs, np.int64)
    for i, j in enumerate(idx):
        ne, nz = int(EL[j]), int(ZL[j])
        k = rng.integers(0, nz)
        pre = np.zeros(k + 1, np.int64)
        pre[1:] = ZH[j, :k]
        en[i, LEN - ne:] = EN[j, :ne]
        env[i, LEN - ne:] = True
        zh[i, LZH - k - 1:] = pre
        zhv[i, LZH - k - 1:] = True
        y[i] = ZH[j, k]
    return (torch.from_numpy(en).cuda(), torch.from_numpy(env).cuda(),
            torch.from_numpy(zh).cuda(), torch.from_numpy(zhv).cuda(),
            torch.from_numpy(y).cuda())


def pick_step(stops, T):
    p = torch.stack(stops, 1)
    pick = (p > 0).float().argmax(1)
    pick[(p <= 0).all(1)] = T - 1
    return pick


@torch.no_grad()
def evaluate(net, batches=6, bs=256, seed=7):
    rng = np.random.default_rng(seed)
    net.eval()
    c = f = n = st = 0
    T = net.T
    for _ in range(batches):
        en, env, zh, zhv, y = make_batch(VA, bs, rng)
        outs, stops, _, _ = net(en, env, zh, zhv)
        pick = pick_step(stops, T)
        pred = torch.stack([o.argmax(-1) for o in outs], 1).gather(1, pick[:, None]).squeeze(1)
        c += int((pred == y).sum())
        f += int((outs[-1].argmax(-1) == y).sum())
        st += int((pick + 1).sum())
        n += bs
    net.train()
    return c / n, f / n, st / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--dmp", type=int, default=0)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--W", type=int, default=8192)
    ap.add_argument("--c0mul", type=float, default=1.2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--log", type=int, default=2500)
    ap.add_argument("--eval", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="ckpt_tr.pt")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--cmode", type=str, default="default")
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--eta_min", type=float, default=1e-4)
    ap.add_argument("--cos", type=int, default=1)
    ap.add_argument("--no_mp", type=int, default=0)
    ap.add_argument("--rand_route", type=int, default=0)
    ap.add_argument("--last_only", type=int, default=0)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(a.seed)
    dmp = a.dmp or a.d
    net = MpChainTr(N=a.N, Ve=VE, Vz=VZ, d=a.d, dmp=dmp, T=a.T).cuda()
    model = torch.compile(net, mode=a.cmode) if a.compile else net
    print(f"params={sum(p.numel() for p in net.parameters())/1e6:.2f}M N={a.N} d={a.d} dmp={dmp} T={a.T} "
          f"lam={a.lam} margin={a.margin} compile={a.compile}/{a.cmode}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr, foreach=False)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.eta_min)
             if a.cos else None)
    start = 1
    if a.resume and os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
        net.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            for g in opt.param_groups:
                g["lr"] = a.lr
        start = int(ck.get("step", 0)) + 1
        print(f"resumed from step {start}", flush=True)
    if sched is not None and start > 1:
        for g in opt.param_groups:
            g["initial_lr"] = a.lr
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=a.steps, eta_min=a.eta_min, last_epoch=start - 1)
    rng = np.random.default_rng(a.seed)
    win = Win(a.N, a.W)
    c0 = a.c0mul * a.W / a.N
    al = torch.tensor([0.02 + 0.28 * t / max(a.T - 2, 1) for t in range(a.T - 1)], device="cuda")
    t0 = time.time()

    for step in range(start, a.steps + 1):
        en, env, zh, zhv, y = make_batch(TR, a.batch, rng)
        pen = (torch.as_tensor(win.cnt, device="cuda", dtype=torch.float32),
               torch.as_tensor(win.r, device="cuda", dtype=torch.float32), al, c0)
        outs, stops, rls, idxs = model(en, env, zh, zhv, pen=pen,
                                       no_mp=bool(a.no_mp), rand_route=bool(a.rand_route))
        loss, ls = stop_loss(outs, stops, y, lam=a.lam, margin=a.margin)
        if a.last_only:
            loss = ls[-1].mean() + a.lam * (loss - torch.stack(ls, 1).mean(1).mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()

        ti = torch.tensor([t for _, _, t in rls], device="cuda")
        xi = torch.cat(idxs)
        win.add(ti.repeat_interleave(a.batch), xi, ls[-1].detach().repeat(len(rls)))

        if step % a.log == 0:
            pick = pick_step(stops, a.T)
            pred = torch.stack([o.argmax(-1) for o in outs], 1).gather(1, pick[:, None]).squeeze(1)
            print(f"{step:5d} | loss {loss.item():.4f} acc {float((pred==y).float().mean()):.4f} "
                  f"full {float((outs[-1].argmax(-1)==y).float().mean()):.4f} "
                  f"steps {float((pick+1).float().mean()):.2f} | {time.time()-t0:.0f}s", flush=True)
        if a.eval and step % a.eval == 0:
            acc, facc, stp = evaluate(net)
            print(f"  EVAL acc {acc:.4f} (full {facc:.4f}, avg steps {stp:.2f})", flush=True)
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step,
                        "args": vars(a)}, a.ckpt)

    acc, facc, stp = evaluate(net)
    print(f"FINAL acc {acc:.4f} (full {facc:.4f}, avg steps {stp:.2f}) in {time.time()-t0:.0f}s", flush=True)
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": a.steps,
                "args": vars(a)}, a.ckpt)


if __name__ == "__main__":
    main()
