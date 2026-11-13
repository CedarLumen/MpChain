"""Static KV-buffer version of the dense TF: shapes stay constant throughout, so it can be captured into one CUDA graph as a whole.

Corresponds to the MpChain version in bench_graphfull.py; both sides have to climb the same ladder for a fair comparison.

Key points:
  ① K/V are preallocated as (B, MAXL, d) and written each step by a device-side index scatter;
  ② attention is always computed over MAXL positions, with a position-dependent additive mask hiding the future;
  ③ positional encoding is taken by absolute position; pos[:, :1] cannot be used (that is exactly the hidden bug of the bench_infer2 cache path).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_infer2 import Block, CausalTF, ENFIX, NT, SEP, VZ, fixed_inputs, tf_prompt

dev = "cuda"
MAXL = ENFIX + 2 + NT


class TFStatic(nn.Module):
    def __init__(self, base: CausalTF, B, maxlen=MAXL, device=None):
        super().__init__()
        self.base, self.B, self.maxlen = base, B, maxlen
        d = base.emb.embedding_dim
        dev = device if device is not None else base.emb.weight.device
        self.kbuf = [torch.zeros(B, maxlen, d, device=dev) for _ in base.blocks]
        self.vbuf = [torch.zeros(B, maxlen, d, device=dev) for _ in base.blocks]
        # additive causal mask table: mask[pos] is a row of length maxlen, 0 for j<=pos and -inf otherwise
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
        """Whole prompt: compute logits and fill K/V into the first L positions."""
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
        """Single-token decode step: tok (B,1), p is a 0-d tensor = the absolute position of that token."""
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
    """Reference implementation: the KV cache path of bench_infer2 (new tensors every step, positional encoding is off)."""
    lg, caches = base(x0, None, causal=True)
    out = [int(lg.argmax(-1)[0])]
    for s in range(nt - 1):
        nxt = torch.tensor([[out[-1]]], device=dev)
        lg, caches = base(nxt, caches, causal=False)
        out.append(int(lg.argmax(-1)[0]))
    return out


@torch.no_grad()
def decode_static(base, x0, nt, bs=1):
    """Static-buffer version: prefill first, then step by step (absolute positions are correct)."""
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
    print(f"reference (existing KV path) first 8: {ref[:8]}")
    print(f"static buffer first 8:              {got[:8]}")
    print(f"first 8 tokens identical: {same} (a mismatch is expected: the existing KV path picks the wrong positional encoding)")


if __name__ == "__main__":
    main()
