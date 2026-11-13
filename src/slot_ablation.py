"""Slot ablation: mask the k-th Mp slot one at a time and see how much accuracy drops.

With T=6, step t reads slots 0..t-1, so slot 5 is never read; only 0..4 are tested.
"""
import numpy as np
import torch

from mpchain_tr import MpChainTr
from train_tr import EL, EN, VA, VE, VZ, ZH, ZL, pick_step

LEN, LZH = 100, 32
CK = "ck_seed0.pt"
ck = torch.load(CK, map_location="cuda", weights_only=False)
a = ck["args"]
net = MpChainTr(N=a["N"], Ve=VE, Vz=VZ, d=a["d"], dmp=a["d"], T=a["T"]).cuda()
net.load_state_dict(ck["model"])
net.eval()
T = net.T
print(f"ckpt step={ck.get('step')} T={T} N={a['N']} d={a['d']}", flush=True)


def batch(bs, rng):
    idx = rng.choice(VA, bs, replace=False)
    en = np.zeros((bs, LEN), np.int64)
    zh = np.zeros((bs, LZH), np.int64)
    env = np.zeros((bs, LEN, 1), bool)
    zhv = np.zeros((bs, LZH, 1), bool)
    y = np.zeros(bs, np.int64)
    for i, j in enumerate(idx):
        ne, nz = int(EL[j]), int(ZL[j])
        k = int(nz * 0.5)
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


@torch.no_grad()
def run(mask, rounds=4, bs=256, seed=31):
    rng = np.random.default_rng(seed)
    c = t = st = 0
    for _ in range(rounds):
        en, env, zh, zhv, y = batch(bs, rng)
        outs, stops, _, _ = net(en, env, zh, zhv, mask_slots=mask)
        pick = pick_step(stops, T)
        pred = torch.stack([o.argmax(-1) for o in outs], 1).gather(1, pick[:, None]).squeeze(1)
        c += int((pred == y).sum())
        st += int((pick + 1).sum())
        t += bs
    return c / t, st / t


base, bstep = run(())
print(f"\nbaseline          acc {base:.4f}  steps {bstep:.2f}", flush=True)
print(f"{'mask slot':<10}{'acc':>8}{'Δ':>9}{'steps':>8}")
for k in range(T - 1):
    acc, st = run((k,))
    print(f"  slot {k+1} masked   {acc:>8.4f}{(acc-base)*100:>+8.1f}{st:>8.2f}", flush=True)
acc, st = run(tuple(range(T - 1)))
print(f"  all masked      {acc:>8.4f}{(acc-base)*100:>+8.1f}{st:>8.2f}", flush=True)
