"""三者的 FLOPs 计账（与实现无关的架构属性）。

口径：FlopCounterMode 统计 aten 算子的乘加量，其中 1 MAC 记为 2 FLOPs（PyTorch 的默认口径）。
论文里若按 MACs 报，把本表数字除以 2 即可；比值不受影响。

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

    # ---- MpChain：源语言编码（prefill）与一次链步 ----
    ve, vz = 153, 2048
    net = MpChainTr(40, ve, vz, 128, 128, 6).to(dev).eval()
    rows, en_ids = fixed_inputs(8)
    en, env = mp_en(8, en_ids)

    pre = net.prefill(en, env)
    zh = torch.zeros(8, LZH, dtype=torch.long, device=dev)
    zhv = torch.zeros(8, LZH, 1, dtype=torch.bool, device=dev)

    f_pre = count(lambda: net.prefill(en, env)) / 8
    f_chain = count(lambda: net(en, env, zh, zhv, pre=pre)) / 8
    # 单步：完整到解一步的 1/6（链内每步结构相同，第一步多编码源语言，这里给下界估计）
    print(f"MpChain 源语言编码（每句一次）      {f_pre/1e6:9.1f} MFLOP")
    print(f"MpChain 一次完整 6 步链（每 token） {f_chain/1e6:9.1f} MFLOP")
    print(f"  → 单链步平均                       {f_chain/6/1e6:9.1f} MFLOP")
    print(f"  → 按平均 3.4 步早停                {f_chain*3.4/6/1e6:9.1f} MFLOP")

    # ---- 标准 MoE：一次前向（batch 8）----
    from train_moe import MoEEncoder
    moe = MoEEncoder(40, 128, 16).to(dev).eval()
    f_moe = count(lambda: moe(en, env, zh, zhv)) / 8
    print(f"MoE 每 token（batch 8）             {f_moe/1e6:9.1f} MFLOP")

    # ---- 稠密 TF：一次 134 位置完整前向（预测 1 个 token）----
    tf = CausalTF().to(dev).eval()
    x0 = tf_prompt(8, en_ids)
    f_tf_nokv = count(lambda: tf(x0, None, causal=True)) / 8
    print(f"稠密 TF 每 token（无 KV、134 位置）  {f_tf_nokv/1e6:9.1f} MFLOP")

    # 有 KV cache 的解码步：只算 1 个新位置
    _, caches = tf(x0[:, :-1], None, causal=True)
    q1 = x0[:, -1:]
    f_tf_kv = count(lambda: tf(q1, caches, causal=False)) / 8
    print(f"稠密 TF 每 token（有 KV cache）      {f_tf_kv/1e6:9.1f} MFLOP")

    print()
    print(f"比值（无 KV / MpChain@3.4 步）      {f_tf_nokv/(f_chain*3.4/6):6.1f}×")
    print(f"比值（KV  / MpChain@3.4 步）        {f_tf_kv/(f_chain*3.4/6):6.1f}×")
    print(f"比值（MoE / MpChain@3.4 步）        {f_moe/(f_chain*3.4/6):6.1f}×")


if __name__ == "__main__":
    main()
