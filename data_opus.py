"""OPUS-100 en-zh train -> 字级 ids（与 data.py 同口径：en<=100, zh<=32，2% 哈希验证集）。

与 data.py 唯一区别：语料换成 OPUS-100 train(1M)，词表按本语料前 10 万对重算。
用法: python data_opus.py
产物: data/opus_train.npz
"""
import hashlib
import os
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

SRC = os.environ.get("MPCHAIN_OPUS_SRC",
                     "/root/autodl-tmp/opus100/train-00000-of-00001.parquet")
OUT = os.environ.get("MPCHAIN_OPUS_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "opus_train.npz"))
LEN, LZH = 100, 32
VZ, VE = 2048, 200

if not os.path.exists(SRC):
    raise SystemExit(f"找不到 OPUS-100 源文件 {SRC}\n请先设置环境变量 MPCHAIN_OPUS_SRC")

rows = pq.ParquetFile(SRC).read(columns=["translation"]).column("translation").to_pylist()
pairs = [(r["en"], r["zh"]) for r in rows if r and r.get("en") and r.get("zh")]
pairs = [(a, b) for a, b in pairs if 2 <= len(a) <= LEN and 2 <= len(b) <= LZH]
print(f"pairs {len(pairs)}")

cz, ce = Counter(), Counter()
for a, b in pairs[:100000]:
    cz.update(b)
    ce.update(a)
zs = [c for c, _ in cz.most_common(VZ - 2)]
es = [c for c, _ in ce.most_common(VE - 2)]
zi = {c: i + 2 for i, c in enumerate(zs)}
ei = {c: i + 2 for i, c in enumerate(es)}
print(f"vocab zh {len(zi)+2} en {len(ei)+2}")

n = len(pairs)
EN = np.zeros((n, LEN), dtype=np.int16)
ZH = np.zeros((n, LZH), dtype=np.int16)
EL = np.zeros(n, dtype=np.int8)
ZL = np.zeros(n, dtype=np.int8)
SP = np.zeros(n, dtype=np.int8)
oov_z = tot_z = 0
for i, (a, b) in enumerate(pairs):
    ea = [ei.get(c, 1) for c in a]
    zb = [zi.get(c, 1) for c in b]
    oov_z += sum(1 for c in b if c not in zi)
    tot_z += len(b)
    EN[i, :len(ea)] = ea
    ZH[i, :len(zb)] = zb
    EL[i] = len(ea)
    ZL[i] = len(zb)
    h = int.from_bytes(hashlib.md5(b.encode()).digest()[:4], "little")
    SP[i] = 1 if h % 100 < 2 else 0

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, en=EN, zh=ZH, en_len=EL, zh_len=ZL, split=SP,
                    zs=np.array(zs), es=np.array(es))
print(f"saved {OUT} train {int((SP==0).sum())} valid {int((SP==1).sum())}")
print(f"zh OOV(全量) {oov_z/tot_z:.2%} | zh_len mean {ZL.mean():.1f} | en_len mean {EL.mean():.1f}")
