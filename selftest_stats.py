"""本地自检：用合成 evaldump 验证 stats_ablation.py 的统计逻辑（不需要 GPU）。"""
import json
import os
import numpy as np

D = os.path.join(os.path.dirname(__file__), 'evaldump_test')
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)

# 造 4 个模型：ref 质量固定，arm_a 明显更差，arm_b 与 ref 相同（应 p≈1）
def make(quality, n=500):
    recs = []
    for i in range(n):
        L = int(rng.integers(5, 20))
        hits = int(round(L * quality))
        recs.append({"idx": i, "hyp": "x", "ref": "y", "bleu2": quality + float(rng.normal(0, .05)),
                     "p1": quality, "p2": quality, "m1": hits, "t1": L,
                     "m2": max(hits - 1, 0), "t2": L, "rlen": L,
                     "hit": hits, "len": L, "steps": 3})
    return recs

for name, q in [('ck_seed0.pt', .30), ('ck_seed1.pt', .305), ('ck_seed2.pt', .295),
                ('ck_ab_nomp.pt', .12), ('ck_ab_randroute.pt', .15), ('ck_ab_lastonly.pt', .05),
                ('ck_moe.pt', .30)]:
    with open(os.path.join(D, f'{name}__synthetic.jsonl'), 'w', encoding='utf-8') as f:
        for r in make(q):
            f.write(json.dumps(r) + '\n')
print('wrote', len(os.listdir(D)), 'files ->', D)
