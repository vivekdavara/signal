"""The two baselines every later model must beat (DESIGN.md, "Baselines").

Both take the binary user × item history matrix. Popularity scores a movie by how many users liked it. Item-kNN
scores movie i for a user by summing ``sim(j, i)`` over the user's history movies j, where each j keeps only its k most
similar movies:

    sim(j, i) = co(j, i) / (sqrt(n_j * n_i) + shrink),   sim(j, j) = 0

``co`` counts users who liked both, ``n`` users who liked each. The similarity matrix is built in column blocks
(``X^T X[:, block]``), so memory is O(items × block) instead of O(items²).

**Exact ties.** "Ties go to the smaller index" must not be decided by rounding. Two things would decide it otherwise:
equal similarities computed by different expressions (3/sqrt(27) and 1/sqrt(3) differ in the last bit of a double),
and equal scores summed in different orders. So, with no shrink, a similarity is computed as sqrt(co² / (n_j n_i)):
equal rationals divide to the same correctly rounded double, and so take the same square root. Every similarity is
then rounded to a multiple of 2^-32 and kept as an integer, and a score is an exact integer sum. Similarities closer
than 2^-32 (about 2.3e-10) count as tied; that is the resolution of the "similarity" this baseline means.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

IntArray = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]

# Similarities are integers in units of 2^-32. A score sums at most one similarity (<= 1) per history movie, so it
# stays an exact float64 (below 2^53) for histories of up to 2^21 movies.
SCALE = 2**32
MAX_HISTORY = 2**21


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
    """The pruned similarity matrix S in units of 2^-32 (int64): row j holds sim(j, i) for j's k most similar movies
    i, ties to the smaller i.

    Movies whose similarity rounds to zero are never neighbours, so a row can hold fewer than k entries.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not math.isfinite(shrink) or shrink < 0:
        raise ValueError(f"shrink must be a non-negative number, got {shrink}")
    if block < 1:
        raise ValueError("block must be at least 1")
    x = sp.csc_matrix(history, dtype=np.float32)
    n_items = x.shape[1]
    counts = np.asarray(x.sum(axis=0), dtype=np.float64).ravel()
    xt = sp.csr_matrix(x.T)
    rows: list[IntArray] = []
    cols: list[IntArray] = []
    vals: list[IntArray] = []
    keep = min(k, n_items)
    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        # co[i, c] = users who liked item i and item start + c.
        co = np.asarray((xt @ x[:, start:stop]).toarray(), dtype=np.float64)
        # Counts and co-counts are integers far below 2^26, so these products are exact in float64.
        product = np.outer(counts, counts[start:stop])
        with np.errstate(divide="ignore", invalid="ignore"):
            if shrink == 0:
                sim = np.where(product > 0, np.sqrt(co * co / product), 0.0)
            else:
                sim = co / (np.sqrt(product) + shrink)
        fixed = np.rint(sim * SCALE).astype(np.int64)
        block_items = np.arange(start, stop)
        fixed[block_items, block_items - start] = 0
        # Column c is sim(start + c, ·) by symmetry; a stable sort on the negated integers keeps ties in index order.
        top = np.argsort(-fixed, axis=0, kind="stable")[:keep]
        top_vals = np.take_along_axis(fixed, top, axis=0)
        for c in range(stop - start):
            nonzero = top_vals[:, c] > 0
            rows.append(np.full(int(nonzero.sum()), start + c, dtype=np.int64))
            cols.append(top[nonzero, c].astype(np.int64))
            vals.append(top_vals[nonzero, c])
    data = np.concatenate(vals) if vals else np.empty(0, dtype=np.int64)
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    return sp.csr_matrix((data, (row, col)), shape=(n_items, n_items), dtype=np.int64)


class ItemKNN:
    name = "item-knn"

    def __init__(self, k: int = 100, shrink: float = 0.0, block: int = 512) -> None:
        # k and block slice arrays, so they must be integers (review round 2 of day 2: k=5.0 failed only at fit).
        integers = all(isinstance(v, int | np.integer) and not isinstance(v, bool) for v in (k, block))
        if not integers or k < 1 or block < 1 or not math.isfinite(shrink) or shrink < 0:
            raise ValueError(f"item-kNN needs k >= 1, block >= 1 and a finite shrink >= 0 (k={k}, shrink={shrink})")
        self.k = k
        self.shrink = shrink
        self.block = block
        self.history: sp.csr_matrix | None = None
        self.similarity: sp.csr_matrix | None = None

    @property
    def params(self) -> dict[str, float | int]:
        return {"k": self.k, "shrink": self.shrink}

    def fit(self, history: sp.csr_matrix) -> None:
        longest = int(np.diff(history.indptr).max()) if history.shape[0] else 0
        if longest > MAX_HISTORY:
            raise ValueError(f"a history of {longest} movies would overflow exact scores (limit {MAX_HISTORY})")
        self.history = sp.csr_matrix(history, dtype=np.int64)
        self.similarity = neighbours(history, self.k, self.shrink, self.block)

    def score(self, users: IntArray) -> Float64Array:
        """Exact sums in units of 2^-32, returned as float64 (exact: below 2^53, see MAX_HISTORY)."""
        if self.history is None or self.similarity is None:
            raise RuntimeError("fit first")
        total = np.asarray((self.history[users] @ self.similarity).toarray(), dtype=np.int64)
        scores: Float64Array = total.astype(np.float64)
        return scores
