"""MpChain 用于真实翻译，带提前终止（EOS）。

第 1 次调用：执行模型读英文原文，把表示写进 Mp 槽位 1。
第 2..T 次调用：读中文前缀（已译部分），并从 Mp 取前面所有槽位。
每一步都输出：词表分布、终止 logit、下一步路由分数。

训练时跑满 T 步拿到每一部的 loss，再算"提前停"的期望损失；
终止决策由第 t+1 步的反事实表现审计（下一步更准就压低第 t 步的终止概率）。
推理时可以按终止概率提前结束，省掉后面几步的算力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RouteOut(torch.autograd.Function):
    """直通 + 计数惩罚：前向恒 0，反向只写 logits 的 idx 那一维。"""

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
    """x [B,L,d] -> 末步隐状态 [B,d]。valid [B,L,1] 为 False 的位置不更新状态。"""
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
        self.opt_prefix = 0      # 1 = 推理期只跑真实前缀长度（不改权重、不改数值）
        d2 = d + dmp
        scale = (128.0 / d) ** 0.5          # 初始化尺度随 d 收缩，d=128 时保持原值
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
        """只算第 1 步的英文编码。生成每个 token 时它都不变，可缓存一次。

        注意 t=0 的路由恒为专家 0（prev 初始化为 0），所以与链无关。
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
                    h = pre                      # 复用 prefill 的结果（数值完全相同）
                else:
                    x = self.emb_en[idx[:, None], en]
                    h = _gru(x, self.Wx1[idx].transpose(1, 2), self.Wh1[idx].transpose(1, 2),
                             self.bx1[idx], self.bh1[idx], en_valid)
            else:
                # 只跑 batch 内真实存在的前缀长度（推理期优化；训练期前缀长度随机，无收益）
                if self.opt_prefix and not self.training:
                    K = zh_len
                    if K is None:                      # 兜底：会同步一次，CUDA graph 里不要走这条
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
                    nxt = route_plan[:, t]        # 强制指定路由（机制对照用）
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
    """每步都优化输出；终止头用"下一步会不会更好"做监督标签。

    label_t = 1 表示第 t+1 步并没有更好（所以第 t 步就该停），
    label_t = 0 表示等一下还有收益。margin 让"差不多好"也算该停。
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
