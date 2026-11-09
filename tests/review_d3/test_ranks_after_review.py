"""Review D3: ``ranks_after`` against ``rerank`` when a score isn't a number."""

from __future__ import annotations

import numpy as np

from signalrec import ranking as rk


class _FirstColumn(rk.RankerWeights):
    """A ranker whose score is feature 0, as in tests/test_ranking.py."""

    def __init__(self) -> None:
        super().__init__(np.zeros(len(rk.FEATURES)), np.ones(len(rk.FEATURES)), ())

    def score(self, features: rk.Float64Array) -> rk.Float64Array:
        out: rk.Float64Array = features[:, 0]
        return out


def test_a_nan_score_ranks_where_rerank_puts_it() -> None:
    """Defect: ``ranks_after`` counts higher and earlier-equal scores; every comparison with NaN is False, so a
    held-out movie scored NaN gets rank 1 (the best), while ``rerank``'s argsort puts NaN last. NaN reaches the scores
    from a non-finite feature, e.g. ``log1p`` of the negative popularity ``FeatureStore.popularity`` accepts.

    Input: one list of 4, scores [0.5, 0.9, NaN, 0.1], the held-out movie at position 2.
    Expected: the rank equals its 1-based position in ``rerank`` (4), or an error. Actual: rank 1.
    """
    features = np.zeros((1, 4, len(rk.FEATURES)))
    features[0, :, 0] = [0.5, 0.9, np.nan, 0.1]
    lists = rk.Lists(features, np.ones((1, 4), dtype=bool), np.array([2], dtype=np.int64))
    weights = _FirstColumn()
    try:
        rank = int(rk.ranks_after(weights, lists)[0])
    except ValueError:
        return
    order = weights.rerank(np.arange(4, dtype=np.int64), features[0])
    assert rank == int(np.flatnonzero(order == 2)[0]) + 1
