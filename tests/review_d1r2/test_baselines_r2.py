"""Review round 2 (day 1): item-kNN's parameter checks after the fixed-point rewrite."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from signalrec.baselines import ItemKNN, neighbours


def test_a_nan_shrink_is_refused() -> None:
    """Defect: baselines.neighbours (line 59) checks ``shrink < 0``, which is False for NaN, so a NaN shrink passes.
    co / (sqrt(n_j n_i) + nan) is NaN, np.rint(NaN * 2^32).astype(int64) is INT64_MIN (a cast warning), and every
    similarity is then dropped as "not > 0": an item-kNN with no neighbours at all, whose scores are all 0, so its
    "recommendations" are the catalogue in index order. ``signalrec recommend --shrink nan`` reaches it (only --grid
    refuses non-finite shrinks).

    Input: 3 users × 3 items with co-likes everywhere, k=2, shrink=NaN.
    Expected: ValueError (like a negative shrink). Actual: an all-zero similarity matrix, no error.
    """
    x = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="shrink"):
            neighbours(x, 2, float("nan"))
        with pytest.raises(ValueError, match="shrink"):
            ItemKNN(k=2, shrink=float("nan")).fit(x)
