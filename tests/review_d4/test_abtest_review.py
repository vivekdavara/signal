"""Review D4: abtest.py's click model."""

from __future__ import annotations

import numpy as np

from signalrec import abtest


def _brute_force(model: abtest.ClickModel, rank: int) -> float:
    """P(at least one click), enumerating which slots are examined and clicked."""
    exam = model.examined()
    p_none = 1.0
    for r in range(model.k):
        p_click = model.p_target if r + 1 == rank else model.p_other
        p_none *= 1.0 - exam[r] * p_click
    return 1.0 - p_none


def test_conversion_is_finite_when_p_other_is_one() -> None:
    """Defect (abtest.py:94, ClickModel.conversion): the probability with the target shown divides the all-other
    product by the target slot's no-click factor. With p_other = 1 (accepted by __post_init__ as a probability) slot 1
    is examined with probability 1, its factor is 0, and 0/0 gives NaN for every user whose next movie is ranked first.

    Input: ClickModel(p_target=0.9, p_other=1.0, k=10).conversion([1]).
    Expected: 1 - 0.1 * prod_{r>=2}(1 - 1/log2(r+1)) (finite, by direct product). Actual: nan.
    """
    model = abtest.ClickModel(0.9, 1.0, 10)
    for rank in (1, 2, 0):
        got = float(model.conversion(np.array([rank]))[0])
        want = _brute_force(model, rank)  # rank 0 matches no slot: not shown
        assert np.isfinite(got), f"rank {rank}: {got}"
        assert abs(got - want) < 1e-12, (rank, got, want)
