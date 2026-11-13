"""稠密 TF 的静态 KV buffer 版：形状全程恒定，可整段捕获进一张 CUDA graph。

与 bench_graphfull.py 的 MpChain 版对应，两边爬同一架梯子才能公平比较。

关键点：
  ① K/V 预分配成 (B, MAXL, d)，每步用设备端下标 scatter 写入；
  ② 注意力始终在 MAXL 个位置上算，靠位置相关的加性掩码屏蔽未来；
  ③ 位置编码按绝对位置取，不能用 pos[:, :1]（那正是 bench_infer2 缓存路径的隐性 bug）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_infer2 import Block, CausalTF, ENFIX, NT, SEP, VZ, fixed_inputs, tf_prompt

dev = "cuda"
MAXL = ENFIX + 2 + NT


class TFStatic(nn.Module):
    def __init__(self, base: CausalTF, B, maxlen=MAXL):
        super().__init__()
        self.base, self.B, self.maxlen = base, B, maxlen
        d = base.emb.embedding_dim
        self.kbuf = [torch.zeros(B, maxlen, d, device=dev) for _ in base.blocks]
        self.vbuf = [torch.zeros(B, maxlen, d, device=dev) for _ in base.blocks]
        # 加性因果掩码表：mask[pos] 是长度 maxlen 的行，j<=pos 为 0，否则 -inf
        idx = torch.arange(maxlen, device=dev)
        m = torch.where(idx[None, :] <= idx[:, None], 0.0, float("-inf"))
        self.mask = m                                   # (maxlen, maxlen)
        self.pos = torch.arange(maxlen, device=dev)

    def _attn(self, l, q, k_new, v_new, p):
        B, d = q.shape[0], q.shape[-1]
        kb, vb = self.kbuf[l], self.vbuf[l]
        kb.scatter_(1, p.view(1, 1, 1).expand(B, -1, kb.shape[-1]), k_new.unsqueeze(1))
        vb.scatter_(1, p.view(1, 1, 1).expand(B, -1, vb.shape[-1]), v_new.unsqueeze(1))
        nhead = self.base.blocks[l].nhead
        hd = d // nhead
        qq = q.view(B, 1, nhead, hd).transpose(1, 2)
        kk = kb.view(B, self.maxlen, nhead, hd).transpose(1, 2)
        vv = vb.view(B, self.maxlen, nhead, hd).transpose(1, 2)
        am = self.mask.gather(0, p.view(1, 1).expand(1, self.maxlen)).view(1, 1, self.maxlen)
        a = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=am)
        return a.transpose(1, 2).reshape(B, 1, d)

    @torch.no_grad()
    def prefill(self, x):
        """整段 prompt：算 logits 并把 K/V 填进前 L 个位置。"""
        b0 = self.base
        L = x.shape[1]
        h = b0.emb(x) + b0.pos[:, :L]
        for l, blk in enumerate(b0.blocks):
            hh = blk.ln1(h)
            q, k, v = blk.qkv(hh).split(h.shape[-1], dim=-1)
            self.kbuf[l][:, :L] = k
            self.vbuf[l][:, :L] = v
            nhead, hd = blk.nhead, h.shape[-1] // blk.nhead
            B = h.shape[0]
            qq = q.view(B, L, nhead, hd).transpose(1, 2)
            kk = k.view(B, L, nhead, hd).transpose(1, 2)
            vv = v.view(B, L, nhead, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
            a = a.transpose(1, 2).reshape(B, L, h.shape[-1])
            h = h + blk.proj(a)
            h = h + blk.fc2(F.gelu(blk.fc1(blk.ln2(h))))
        return b0.head(b0.norm(h)[:, -1:])

    @torch.no_grad()
    def step(self, tok, p):
        """单 token 解码步：tok (B,1)，p 0-d 张量=该 token 的绝对位置。"""
        b0 = self.base
        d = b0.emb.embedding_dim
        h = b0.emb(tok) + b0.pos[0].index_select(0, p.view(1)).view(1, 1, -1)
        for l, blk in enumerate(b0.blocks):
            hh = blk.ln1(h)
            q, k, v = blk.qkv(hh).split(h.shape[-1], dim=-1)
            a = self._attn(l, q[:, 0], k[:, 0], v[:, 0], p)
            h = h + blk.proj(a)
            h = h + blk.fc2(F.gelu(blk.fc1(blk.ln2(h))))
        return b0.head(b0.norm(h)[:, -1])


@torch.no_grad()
def decode_ref(base, x0, nt):
    """参考实现：bench_infer2 的 KV cache 路径（每步新建张量，位置编码有偏差）。"""
    lg, caches = base(x0, None, causal=True)
    out = [int(lg.argmax(-1)[0])]
    for s in range(nt - 1):
        nxt = torch.tensor([[out[-1]]], device=dev)
        lg, caches = base(nxt, caches, causal=False)
        out.append(int(lg.argmax(-1)[0]))
    return out


@torch.no_grad()
def decode_static(base, x0, nt, bs=1):
    """静态 buffer 版：先 prefill，再逐步 step（绝对位置正确）。"""
    sv = TFStatic(base, x0.shape[0])
    lg = sv.prefill(x0)
    out = [int(lg[0].argmax(-1))]
    for s in range(1, nt):
        nxt = torch.tensor([[out[-1]]], device=dev)
        lg = sv.step(nxt, torch.tensor(ENFIX + 1 + s, device=dev))
        out.append(int(lg[0].argmax(-1)))
    return out


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    base = CausalTF().to(dev).eval()
    bs = 1
    rows, en_ids = fixed_inputs(bs)
    x0 = tf_prompt(bs, en_ids)

    ref = decode_ref(base, x0, NT)
    got = decode_static(base, x0, NT, bs)
    same = ref[:8] == got[:8]
    print(f"参考(现有 KV 路径) 前 8: {ref[:8]}")
    print(f"静态 buffer 前 8:     {got[:8]}")
    print(f"前 8 个 token 是否一致: {same}（不一致是预期的：现有 KV 路径位置编码取错）")


if __name__ == "__main__":
    main()
