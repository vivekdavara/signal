"""Ranks, metrics and intervals against brute force and hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from signalrec import evaluate as ev


def csr(dense: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(dense.astype(np.float32))


@st.composite
def ranking_problems(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_users = draw(st.integers(1, 6))
    n_items = draw(st.integers(2, 12))
    # Small integer scores make ties common.
    scores = draw(hnp.arrays(np.float64, (n_users, n_items), elements=st.integers(-3, 3).map(float)))
    history = draw(hnp.arrays(np.bool_, (n_users, n_items)))
    targets = []
    for u in range(n_users):
        free = np.flatnonzero(~history[u])
        if len(free) == 0:
            history[u, 0] = False
            free = np.array([0])
        targets.append(int(draw(st.sampled_from(free.tolist()))))
    return scores, history, np.array(targets, dtype=np.int64)


def brute_rank(scores: np.ndarray, history: np.ndarray, target: int) -> int:
    order = sorted((i for i in range(len(scores)) if not history[i]), key=lambda i: (-scores[i], i))
    return order.index(target) + 1


@settings(max_examples=400, deadline=None)
@given(ranking_problems())
def test_ranks_match_a_sort_by_score_then_index(problem: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    scores, history, targets = problem
    got = ev.ranks(scores, csr(history), targets)
    want = [brute_rank(scores[u], history[u], int(targets[u])) for u in range(len(targets))]
    assert got.tolist() == want


@settings(max_examples=400, deadline=None)
@given(ranking_problems(), st.integers(1, 12))
def test_top_k_agrees_with_ranks(problem: tuple[np.ndarray, np.ndarray, np.ndarray], k: int) -> None:
    scores, history, targets = problem
    top = ev.top_k(scores, csr(history), k)
    rank = ev.ranks(scores, csr(history), targets)
    for u in range(len(targets)):
        listed = [i for i in top[u].tolist() if i >= 0]
        free = [i for i in range(scores.shape[1]) if not history[u, i]]
        assert listed == sorted(free, key=lambda i: (-scores[u, i], i))[:k]
        assert (targets[u] in listed) == (rank[u] <= k)


def test_recall_and_ndcg_by_hand() -> None:
    rank = np.array([1, 3, 10, 11], dtype=np.int64)
    assert ev.recall_at(rank, 10).tolist() == [1, 1, 1, 0]
    ndcg = ev.ndcg_at(rank, 10)
    assert ndcg[0] == 1.0
    assert ndcg[1] == 0.5
    assert ndcg[2] == pytest.approx(1 / math.log2(11))
    assert ndcg[3] == 0.0


def test_a_target_in_the_history_is_refused() -> None:
    with pytest.raises(ValueError, match="in its user's history"):
        ev.ranks(np.zeros((1, 3)), csr(np.array([[0, 1, 0]])), np.array([1]))


def test_a_non_finite_target_score_is_refused() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        ev.ranks(np.array([[0.0, np.nan, 1.0]]), csr(np.zeros((1, 3))), np.array([1]))


class Fixed:
    name = "fixed"

    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores

    def fit(self, history: sp.csr_matrix) -> None:
        pass

    def score(self, users: np.ndarray) -> np.ndarray:
        out: np.ndarray = self.scores[users]
        return out


def test_batching_does_not_change_the_result() -> None:
    rng = np.random.default_rng(3)
    scores = rng.integers(0, 4, size=(37, 60)).astype(np.float32)
    history = rng.random((37, 60)) < 0.3
    targets = np.array([int(np.flatnonzero(~row)[0]) for row in history], dtype=np.int64)
    a = ev.evaluate(Fixed(scores), csr(history), targets, "val", batch=5)
    b = ev.evaluate(Fixed(scores), csr(history), targets, "val", batch=1000)
    np.testing.assert_array_equal(a.ranks, b.ranks)
    np.testing.assert_array_equal(a.top, b.top)
    assert a.summary() == b.summary()


def test_summary_by_hand() -> None:
    # Two users, four items, no history. User 0's target ranks 1st, user 1's 3rd.
    scores = np.array([[4, 3, 2, 1], [4, 3, 2, 1]], dtype=np.float32)
    e = ev.evaluate(Fixed(scores), csr(np.zeros((2, 4))), np.array([0, 2]), "test", ks=(1, 2))
    assert e.ranks.tolist() == [1, 3]
    assert e.summary() == {
        "recall@1": 0.5,
        "ndcg@1": 0.5,
        "coverage@1": 0.25,
        "recall@2": 0.5,
        "ndcg@2": 0.5,
        "coverage@2": 0.5,
    }


def test_coverage_ignores_padding_when_a_user_has_seen_almost_everything() -> None:
    scores = np.array([[1, 2, 3]], dtype=np.float32)
    e = ev.evaluate(Fixed(scores), csr(np.array([[1, 1, 0]])), np.array([2]), "val", ks=(2,))
    assert e.top.tolist() == [[2, -1]]
    assert e.summary()["coverage@2"] == pytest.approx(1 / 3)


def test_bootstrap_intervals() -> None:
    rng = np.random.default_rng(0)
    values = (rng.random(2000) < 0.3).astype(np.float64)
    low, high = ev.bootstrap_ci(values)
    assert low < values.mean() < high
    # The normal approximation's 95% width is 2 * 1.96 * sqrt(p (1 - p) / n) = 0.0405 for this sample; a 90% interval
    # (a wrong alpha, as in a review-round-1 survivor) would be 0.034.
    expected = 2 * 1.96 * math.sqrt(values.mean() * (1 - values.mean()) / len(values))
    assert abs((high - low) - expected) < 0.08 * expected
    assert ev.bootstrap_ci(values) == (low, high)
    lo_d, hi_d = ev.paired_bootstrap_ci(values, values)
    assert lo_d == hi_d == 0.0
    with pytest.raises(ValueError, match="same users"):
        ev.paired_bootstrap_ci(values, values[:10])


def test_bootstrap_resamples_every_user() -> None:
    # Only the last user has a 1: an interval that never draws the last index would be [0, 0].
    values = np.zeros(100)
    values[-1] = 1.0
    assert ev.bootstrap_ci(values)[1] > 0


def test_bootstrap_defaults_are_the_documented_ones() -> None:
    values = np.random.default_rng(1).random(300)
    assert ev.bootstrap_ci(values) == ev.bootstrap_ci(values, n_resamples=1000, seed=0, level=0.95)


def test_paired_intervals_are_for_a_minus_b() -> None:
    a = np.full(50, 0.5)
    b = np.zeros(50)
    b[:10] = 1.0
    low, high = ev.paired_bootstrap_ci(a, b)
    assert 0 < low <= 0.3 <= high


def test_top_k_breaks_ties_by_index_past_insertion_sort_sizes() -> None:
    # 40 equal scores: NumPy's default sort is only stable by accident below 16 elements (a review-round-1 survivor).
    scores = np.zeros((1, 40))
    assert ev.top_k(scores, csr(np.zeros((1, 40))), 30)[0].tolist() == list(range(30))


def test_evaluate_checks_its_inputs() -> None:
    with pytest.raises(ValueError, match="2 targets for 3 users"):
        ev.evaluate(Fixed(np.zeros((3, 4))), csr(np.zeros((3, 4))), np.array([0, 1]), "val")
    with pytest.raises(ValueError, match="non-finite"):
        ev.ranks(np.array([[0.0, np.inf, 1.0]]), csr(np.zeros((1, 3))), np.array([1]))
