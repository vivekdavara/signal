"""The two baselines every later model must beat (DESIGN.md, "Baselines").

Both take the binary user × item history matrix. Popularity scores a movie by how many users liked it. Item-kNN
scores movie i for a user by summing ``sim(j, i)`` over the user's history movies j, where each j keeps only its k most
similar movies:

    sim(j, i) = co(j, i) / (sqrt(n_j * n_i) + shrink),   sim(j, j) = 0

``co`` counts users who liked both, ``n`` users who liked each. The similarity matrix is built in column blocks
(``X^T X[:, block]``), so memory is O(items × block) instead of O(items²).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

IntArray = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]


class Popularity:
    name = "popularity"

    def __init__(self) -> None:
        self.counts: Float32Array | None = None

    def fit(self, history: sp.csr_matrix) -> None:
        self.counts = np.asarray(history.sum(axis=0), dtype=np.float32).ravel()

    def score(self, users: IntArray) -> Float32Array:
        if self.counts is None:
            raise RuntimeError("fit first")
        return np.broadcast_to(self.counts, (len(users), len(self.counts)))


def neighbours(history: sp.csr_matrix, k: int, shrink: float, block: int = 512) -> sp.csr_matrix:
    """The pruned similarity matrix S: row j holds sim(j, i) for j's k most similar movies i, ties to smaller i.

    Movies with zero similarity are never neighbours, so a row can hold fewer than k entries.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if shrink < 0:
        raise ValueError("shrink must be non-negative")
    x = sp.csc_matrix(history, dtype=np.float32)
    n_items = x.shape[1]
    counts = np.asarray(x.sum(axis=0), dtype=np.float64).ravel()
    xt = sp.csr_matrix(x.T)
    rows: list[IntArray] = []
    cols: list[IntArray] = []
    vals: list[Float32Array] = []
    keep = min(k, n_items)
    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        # co[i, c] = users who liked item i and item start + c.
        co = np.asarray((xt @ x[:, start:stop]).toarray(), dtype=np.float64)
        denom = np.sqrt(np.outer(counts, counts[start:stop])) + shrink
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0, co / denom, 0.0)
        block_items = np.arange(start, stop)
        sim[block_items, block_items - start] = 0.0
        # Column c is sim(start + c, ·) by symmetry; a stable sort on -sim keeps equal values in index order.
        top = np.argsort(-sim, axis=0, kind="stable")[:keep]
        top_vals = np.take_along_axis(sim, top, axis=0)
        for c in range(stop - start):
            nonzero = top_vals[:, c] > 0
            rows.append(np.full(int(nonzero.sum()), start + c, dtype=np.int64))
            cols.append(top[nonzero, c].astype(np.int64))
            vals.append(top_vals[nonzero, c].astype(np.float32))
    data = np.concatenate(vals) if vals else np.empty(0, dtype=np.float32)
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    return sp.csr_matrix((data, (row, col)), shape=(n_items, n_items), dtype=np.float32)


class ItemKNN:
    name = "item-knn"

    def __init__(self, k: int = 100, shrink: float = 0.0, block: int = 512) -> None:
        self.k = k
        self.shrink = shrink
        self.block = block
        self.history: sp.csr_matrix | None = None
        self.similarity: sp.csr_matrix | None = None

    @property
    def params(self) -> dict[str, float | int]:
        return {"k": self.k, "shrink": self.shrink}

    def fit(self, history: sp.csr_matrix) -> None:
        self.history = sp.csr_matrix(history, dtype=np.float32)
        self.similarity = neighbours(self.history, self.k, self.shrink, self.block)

    def score(self, users: IntArray) -> Float32Array:
        if self.history is None or self.similarity is None:
            raise RuntimeError("fit first")
        scores: Float32Array = np.asarray((self.history[users] @ self.similarity).toarray(), dtype=np.float32)
        return scores
