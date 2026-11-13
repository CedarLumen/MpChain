"""Sliding-window statistics for the routing load penalty: frequencies and relative performance.

Extracted from an earlier training script of a different task, which is not part of this release.
"""
from collections import deque

import numpy as np


class Win:
    """Sliding window over (call index, model index, sample loss); only decision-making calls with t >= 1.

    `cnt` holds how often each expert was called inside the window, `tm` the same counts per call
    index, and `r` the fraction of calls in which an expert did better than the window mean. The
    router uses these three to penalise experts that are called far more often than the others
    (paper section 3.4).
    """

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
