"""Full-ranking evaluation with one held-out item per user (DESIGN.md, "Evaluation protocol").

Every model scores every movie. A user's history is excluded, and the held-out item's **rank** is computed exactly:
1 + the number of non-history movies scored higher, + the number scored equal with a smaller index. That is the rank
it would get from a sort by (score descending, index ascending), so ties never depend on a sort algorithm. Recall@K
and NDCG@K follow from the rank; the top-K lists (same order) give catalogue coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
Scores = npt.NDArray[np.float32] | npt.NDArray[np.float64]

KS = (10, 20, 50)
BATCH = 512


class Scorer(Protocol):
    """Anything evaluation can rank with: a fitted baseline, or a trained model's exported embeddings."""

    @property
    def name(self) -> str: ...

    def score(self, users: IntArray) -> Scores:
        """Scores for every item, one row per user."""


class Recommender(Scorer, Protocol):
    def fit(self, history: sp.csr_matrix) -> None:
        """Learn from the binary user × item history matrix."""


def ranks(scores: Scores, history: sp.csr_matrix, targets: IntArray) -> IntArray:
    """The 1-based rank of each row's target among the row's non-history items, ties broken by item index."""
    n_rows, n_items = scores.shape
    rows = np.arange(n_rows)
    if history[rows, targets].any():
        raise ValueError("a target item is in its user's history")
    masked = np.asarray(scores, dtype=np.float64).copy()
    masked[history.nonzero()] = -np.inf
    target_scores = masked[rows, targets][:, None]
    if not np.isfinite(target_scores).all():
        raise ValueError("a target item has a non-finite score")
    higher = (masked > target_scores).sum(axis=1)
    smaller_index = np.arange(n_items)[None, :] < targets[:, None]
    tied_before = ((masked == target_scores) & smaller_index).sum(axis=1)
    result: IntArray = (1 + higher + tied_before).astype(np.int64)
    return result


def top_k(scores: Scores, history: sp.csr_matrix, k: int) -> IntArray:
    """Each row's k best non-history items, by (score descending, index ascending); -1 pads a row with fewer."""
    masked = np.asarray(scores, dtype=np.float64).copy()
    masked[history.nonzero()] = -np.inf
    # A stable sort keeps equal scores in index order.
    order: IntArray = np.argsort(-masked, axis=1, kind="stable")[:, :k].astype(np.int64)
    order[np.take_along_axis(masked, order, axis=1) == -np.inf] = -1
    return order


def recall_at(rank: IntArray, k: int) -> FloatArray:
    out: FloatArray = (rank <= k).astype(np.float64)
    return out


def ndcg_at(rank: IntArray, k: int) -> FloatArray:
    # With one relevant item the ideal DCG is 1, so NDCG = 1 / log2(rank + 1) inside the cut-off.
    out: FloatArray = np.where(rank <= k, 1.0 / np.log2(rank + 1.0), 0.0)
    return out


@dataclass
class Evaluation:
    model: str
    split: str
    ranks: IntArray
    n_items: int
    top: IntArray  # each user's top max(KS) items
    ks: tuple[int, ...] = KS
    params: dict[str, float | int] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in self.ks:
            out[f"recall@{k}"] = float(recall_at(self.ranks, k).mean())
            out[f"ndcg@{k}"] = float(ndcg_at(self.ranks, k).mean())
            shown = self.top[:, :k]
            out[f"coverage@{k}"] = len(np.unique(shown[shown >= 0])) / self.n_items
        return out


def evaluate(
    model: Scorer,
    history: sp.csr_matrix,
    targets: IntArray,
    split: str,
    ks: Sequence[int] = KS,
    batch: int = BATCH,
) -> Evaluation:
    """Rank every user's target with ``model`` (already fit on ``history``), in batches of users."""
    n_users, n_items = history.shape
    if len(targets) != n_users:
        raise ValueError(f"{len(targets)} targets for {n_users} users")
    k_max = max(ks)
    all_ranks = np.empty(n_users, dtype=np.int64)
    all_top = np.empty((n_users, min(k_max, n_items)), dtype=np.int64)
    for start in range(0, n_users, batch):
        users = np.arange(start, min(start + batch, n_users), dtype=np.int64)
        scores = model.score(users)
        rows = history[users]
        all_ranks[users] = ranks(scores, rows, targets[users])
        all_top[users] = top_k(scores, rows, k_max)
    return Evaluation(model.name, split, all_ranks, n_items, all_top, tuple(ks))


def bootstrap_ci(
    values: FloatArray, n_resamples: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """A percentile bootstrap interval for the mean over users."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1 - level) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(low), float(high)


def paired_bootstrap_ci(
    a: FloatArray, b: FloatArray, n_resamples: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """An interval for mean(a) - mean(b) when a and b are two models' values on the same users."""
    if a.shape != b.shape:
        raise ValueError("paired values need the same users")
    return bootstrap_ci(a - b, n_resamples, seed, level)
