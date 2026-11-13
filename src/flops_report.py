"""FLOP accounting for the three (an architectural property, independent of the implementation).

Protocol: FlopCounterMode counts the multiply-adds of aten operators, where 1 MAC counts as 2 FLOPs (PyTorch's default convention).
If the paper reports MACs, just divide the numbers in this table by 2; the ratios are unaffected.

python flops_report.py
"""
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from bench_infer2 import ENFIX, LEN, LZH, NT, VZ, CausalTF, fixed_inputs, mp_en, tf_prompt
from mpchain_tr import MpChainTr


def count(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def main():
    dev = "cuda"
    torch.manual_seed(0)

    # ---- MpChain: source-language encoding (prefill) and one chain step ----
    ve, vz = 153, 2048
    net = MpChainTr(40, ve, vz, 128, 128, 6).to(dev).eval()
    rows, en_ids = fixed_inputs(8)
    en, env = mp_en(8, en_ids)

    pre = net.prefill(en, env)
    zh = torch.zeros(8, LZH, dtype=torch.long, device=dev)
    zhv = torch.zeros(8, LZH, 1, dtype=torch.bool, device=dev)

    f_pre = count(lambda: net.prefill(en, env)) / 8
    f_chain = count(lambda: net(en, env, zh, zhv, pre=pre)) / 8
    # one step: 1/6 of a full chain-to-solution pass (every step in the chain has the same structure; the first step additionally encodes the source language, so this is a lower-bound estimate)
    print(f"MpChain source encoding (per sentence)   {f_pre/1e6:9.1f} MFLOP")
    print(f"MpChain one full 6-step chain (per token) {f_chain/1e6:9.1f} MFLOP")
    print(f"  → average per chain step               {f_chain/6/1e6:9.1f} MFLOP")
    print(f"  → early stop at 3.4 steps on average   {f_chain*3.4/6/1e6:9.1f} MFLOP")

    # ---- Standard MoE: one forward pass (batch 8) ----
    from train_moe import MoEEncoder
    moe = MoEEncoder(40, 128, 16).to(dev).eval()
    f_moe = count(lambda: moe(en, env, zh, zhv)) / 8
    print(f"MoE per token (batch 8)                {f_moe/1e6:9.1f} MFLOP")

    # ---- Dense TF: one full forward pass over 134 positions (predicting 1 token) ----
    tf = CausalTF().to(dev).eval()
    x0 = tf_prompt(8, en_ids)
    f_tf_nokv = count(lambda: tf(x0, None, causal=True)) / 8
    print(f"Dense TF per token (no KV, 134 pos)    {f_tf_nokv/1e6:9.1f} MFLOP")

    # decode step with KV cache: only 1 new position is computed
    _, caches = tf(x0[:, :-1], None, causal=True)
    q1 = x0[:, -1:]
    f_tf_kv = count(lambda: tf(q1, caches, causal=False)) / 8
    print(f"Dense TF per token (with KV cache)     {f_tf_kv/1e6:9.1f} MFLOP")

    print()
    print(f"ratio (no KV / MpChain@3.4 steps)      {f_tf_nokv/(f_chain*3.4/6):6.1f}×")
    print(f"ratio (KV  / MpChain@3.4 steps)        {f_tf_kv/(f_chain*3.4/6):6.1f}×")
    print(f"ratio (MoE / MpChain@3.4 steps)        {f_moe/(f_chain*3.4/6):6.1f}×")


if __name__ == "__main__":
    main()
