"""滑动窗口统计：路由负载处罚要用的频次与相对表现。

从早期的 train.py 抽出（那个文件是另一套任务，不进复现包）。
"""
from collections import deque

import numpy as np


class Win:
    """滑动窗口：记 (t, model_idx, sample_loss)。只收 t>=1 的决策性选择。"""

    def __init__(self, N, W):
        self.N, self.W = N, W
        self.dq = deque()
        self.cnt = np.zeros(N)
        self.r = np.full(N, 0.5)
        self.tm = np.zeros((16, N), dtype=np.int64)
        self.k = 0

    def add(self, t, idx, loss):
        self.dq.extend(zip(t.tolist(), idx.tolist(), loss.tolist()))
        while len(self.dq) > self.W:
            self.dq.popleft()
        self.k += 1
        self.refresh(full=self.k % 8 == 1)

    def refresh(self, full=True):
        if not self.dq:
            return
        n = len(self.dq)
        tt = np.fromiter((a for a, _, _ in self.dq), dtype=np.int64, count=n)
        ii = np.fromiter((b for _, b, _ in self.dq), dtype=np.int64, count=n)
        self.cnt = np.bincount(ii, minlength=self.N).astype(np.float64)
        self.tm = np.bincount(tt * self.N + ii, minlength=16 * self.N).reshape(16, self.N)
        if not full:
            return
        ll = np.fromiter((c for _, _, c in self.dq), dtype=np.float64, count=n)
        m = ll.mean()
        r = np.full(self.N, 0.5)
        for k in range(self.N):
            mk = ii == k
            s = mk.sum()
            if s >= 8:
                r[k] = float((ll[mk] < m).mean())
        self.r = r
