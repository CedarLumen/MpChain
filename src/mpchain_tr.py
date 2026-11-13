"""MpChain applied to real translation, with early stopping (EOS).

Call 1: the executive model reads the English source and writes the representation into Mp slot 1.
Calls 2..T: read the Chinese prefix (the translated part) and take all earlier slots from Mp.
Every step emits: vocabulary distribution, stop logit, next-step routing score.

During training we run all T steps to get each step's loss, then compute the expected loss of "stopping early";
the stop decision is audited by the counterfactual behavior of step t+1 (if the next step is more accurate, lower the stop probability of step t).
At inference we can end early according to the stop probability, saving the compute of the later steps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RouteOut(torch.autograd.Function):
    """Straight-through + count penalty: forward is always 0, backward writes only the idx dimension of logits."""

    @staticmethod
    def forward(ctx, sel, logits, idx, wgt):
        ctx.save_for_backward(idx, wgt)
        ctx.n = logits.shape[1]
        return torch.zeros_like(sel)

    @staticmethod
    def backward(ctx, g):
        idx, wgt = ctx.saved_tensors
        gb = g.abs().mean()
        gl = torch.zeros(g.shape[0], ctx.n, dtype=g.dtype, device=g.device)
        gl.scatter_(1, idx[:, None], (g + gb * wgt)[:, None])
        return None, gl, None, None


def _gru(x, Wx, Wh, bx, bh, valid):
    """x [B,L,d] -> last-step hidden state [B,d]. Positions where valid [B,L,1] is False do not update the state."""
    B, L, d = x.shape
    gx = torch.baddbmm(bx.unsqueeze(1), x, Wx)
    hh = torch.zeros(B, d, device=x.device, dtype=x.dtype)
    for s in range(L):
        g = gx[:, s]
        gh = torch.baddbmm(bh.unsqueeze(1), hh.unsqueeze(1), Wh).squeeze(1)
        rz = torch.sigmoid(g[:, :2 * d] + gh[:, :2 * d])
        n = torch.tanh(g[:, 2 * d:] + rz[:, :d] * gh[:, 2 * d:])
        hh = torch.where(valid[:, s], torch.lerp(n, hh, rz[:, d:]), hh)
    return hh


class MpChainTr(nn.Module):
    def __init__(self, N=100, Ve=160, Vz=2050, d=64, dmp=64, T=10):
        super().__init__()
        self.N, self.Ve, self.Vz, self.d, self.dmp, self.T = N, Ve, Vz, d, dmp, T
        self.opt_prefix = 0      # 1 = at inference run only the true prefix length (no weight or numeric change)
        d2 = d + dmp
        scale = (128.0 / d) ** 0.5          # init scale shrinks with d; keeps the original value at d=128
        rn = lambda *s, k=0.08: torch.randn(*s) * (k * scale)
        self.emb_en = nn.Parameter(rn(N, Ve, d, k=0.3))
        self.emb_zh = nn.Parameter(rn(N, Vz, d, k=0.3))
        self.Wx1 = nn.Parameter(rn(N, 3 * d, d))
        self.Wh1 = nn.Parameter(rn(N, 3 * d, d))
        self.bx1 = nn.Parameter(torch.zeros(N, 3 * d))
        self.bh1 = nn.Parameter(torch.zeros(N, 3 * d))
        self.Wx2 = nn.Parameter(rn(N, 3 * d, d))
        self.Wh2 = nn.Parameter(rn(N, 3 * d, d))
        self.bx2 = nn.Parameter(torch.zeros(N, 3 * d))
        self.bh2 = nn.Parameter(torch.zeros(N, 3 * d))
        self.Wq = nn.Parameter(rn(N, dmp, d) / (dmp ** 0.5))
        self.Wroute = nn.Parameter(rn(N, N, d2))
        self.broute = nn.Parameter(rn(N, N, k=0.15))
        self.Wwrite = nn.Parameter(rn(N, dmp, d2))
        self.Wproj = nn.Parameter(rn(N, d, d2))
        self.Wstop = nn.Parameter(rn(N, 1, d2, k=0.03))

    def prefill(self, en, en_valid):
        """Compute only the English encoding of step 1. It does not change while generating each token, so it can be cached once.

        Note that routing at t=0 is always expert 0 (prev is initialized to 0), so it is independent of the chain.
        """
        B = en.shape[0]
        idx = torch.zeros(B, dtype=torch.long, device=en.device)
        x = self.emb_en[idx[:, None], en]
        return _gru(x, self.Wx1[idx].transpose(1, 2), self.Wh1[idx].transpose(1, 2),
                    self.bx1[idx], self.bh1[idx], en_valid)

    def forward(self, en, en_valid, zh, zh_valid, pen=None, stop_bias=0.0,
                no_mp=False, rand_route=False, mask_slots=(), pre=None, zh_len=None,
                route_plan=None):
        B = en.shape[0]
        T = self.T
        dev, d, dmp = en.device, self.d, self.dmp
        slots, rls, idxs, outs, stops = [], [], [], [], []
        prev = torch.zeros(B, dtype=torch.long, device=dev)
        carry = None
        for t in range(T):
            idx = prev
            if t == 0:
                if pre is not None:
                    h = pre                      # reuse the prefill result (numerically identical)
                else:
                    x = self.emb_en[idx[:, None], en]
                    h = _gru(x, self.Wx1[idx].transpose(1, 2), self.Wh1[idx].transpose(1, 2),
                             self.bx1[idx], self.bh1[idx], en_valid)
            else:
                # run only the prefix length actually present in the batch (inference-time optimization; training prefixes are random, so no gain)
                if self.opt_prefix and not self.training:
                    K = zh_len
                    if K is None:                      # fallback: costs one sync, do not take this path with CUDA graph
                        vl0 = zh_valid.squeeze(-1)
                        K = int(vl0.sum(1).max().item()) if vl0.numel() else 0
                    if K > 0:
                        LZ = zh.shape[1]
                        x = self.emb_zh[idx[:, None], zh[:, LZ - K:]]
                        h = _gru(x, self.Wx2[idx].transpose(1, 2), self.Wh2[idx].transpose(1, 2),
                                 self.bx2[idx], self.bh2[idx], zh_valid[:, LZ - K:])
                    else:
                        h = torch.zeros(B, d, device=dev, dtype=self.emb_zh.dtype)
                else:
                    x = self.emb_zh[idx[:, None], zh]
                    h = _gru(x, self.Wx2[idx].transpose(1, 2), self.Wh2[idx].transpose(1, 2),
                             self.bx2[idx], self.bh2[idx], zh_valid)
            if carry is not None:
                h = h + carry[:, None]
            if slots and not no_mp:
                MP = torch.stack(slots, 1)
                q = torch.bmm(h.unsqueeze(1), self.Wq[idx].transpose(1, 2)).squeeze(1)
                att = torch.bmm(q.unsqueeze(1), MP.transpose(1, 2)).squeeze(1)
                if mask_slots:
                    ms = [x for x in mask_slots if x < MP.shape[1]]
                    if ms:
                        att[:, ms] = -1e9
                zmp = torch.bmm(att.softmax(-1).unsqueeze(1), MP).squeeze(1)
            else:
                zmp = torch.zeros(B, dmp, device=dev, dtype=h.dtype)
            h2 = torch.cat([h, zmp], -1)
            o = torch.bmm(h2.unsqueeze(1), self.Wproj[idx].transpose(1, 2)).squeeze(1)
            outs.append(torch.bmm(o.unsqueeze(1), self.emb_zh[idx].transpose(1, 2)).squeeze(1))
            st = torch.bmm(h2.unsqueeze(1), self.Wstop[idx].transpose(1, 2)).squeeze(1).squeeze(1)
            stops.append(st + stop_bias)
            rl = torch.baddbmm(self.broute[idx].unsqueeze(1), h2.unsqueeze(1),
                               self.Wroute[idx].transpose(1, 2)).squeeze(1)
            slots.append(torch.bmm(h2.unsqueeze(1), self.Wwrite[idx].transpose(1, 2)).squeeze(1))
            if t < T - 1:
                if route_plan is not None:
                    nxt = route_plan[:, t]        # force the routing (for mechanism ablation)
                else:
                    nxt = (torch.randint(0, self.N, (B,), device=dev) if rand_route
                           else rl.argmax(-1))
                if pen is None:
                    wgt = torch.zeros_like(rl[:, 0])
                else:
                    pc, pr, pal, pc0 = pen
                    c = pc[nxt]
                    wgt = pal[t] * ((c - pc0).clamp(min=0) / pc0).clamp(max=1.0) * (1.0 - pr[nxt])
                rls.append((rl, nxt, t))
                idxs.append(nxt)
                s_sel = rl.gather(1, nxt[:, None]).squeeze(1)
                carry = _RouteOut.apply(s_sel, rl, nxt, wgt) / (d + dmp)
                prev = nxt
        self.last_slots = [s.detach() for s in slots]
        return outs, stops, rls, idxs


def stop_loss(outs, stops, y, lam=1.0, margin=0.0):
    """Optimize the output at every step; the stop head is supervised with "will the next step be better".

    label_t = 1 means step t+1 is not better (so step t should stop),
    label_t = 0 means waiting still pays off. margin makes "about as good" count as should-stop.
    """
    T = len(outs)
    ls = [F.cross_entropy(o, y, reduction="none") for o in outs]
    main = torch.stack(ls, 1).mean(1).mean()
    bc = 0.0
    for t in range(T - 1):
        with torch.no_grad():
            should = (ls[t + 1].detach() >= ls[t].detach() - margin).float()
        bc = bc + F.binary_cross_entropy_with_logits(stops[t], should)
    return main + lam * bc / max(T - 1, 1), ls
