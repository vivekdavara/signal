"""Review findings (day 1): floating-point rounding decides item-kNN "ties" that the spec breaks by index."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from signalrec import evaluate as ev
from signalrec.baselines import ItemKNN, neighbours


def test_mathematically_tied_neighbours_go_to_the_smaller_index() -> None:
    """Defect: baselines.neighbours (line 61) compares similarities as rounded float64 quotients, so two exactly
    equal similarities can differ in the last bit and the "ties to smaller i" rule is decided by rounding.

    Input: 9 users, 3 items. Item 0 is liked by all 9, item 2 by users 0-2, item 1 by user 0 only. So
    sim(2, 0) = 3 / sqrt(9 * 3) and sim(2, 1) = 1 / sqrt(1 * 3), both exactly 1 / sqrt(3).
    Expected (k=1): row 2 keeps item 0, the smaller index of the tie.
    Actual: row 2 keeps item 1, because 3 / fl(sqrt(27)) = 0.5773502691896257 < 1 / fl(sqrt(3)) = 0.5773502691896258.
    """
    x = np.zeros((9, 3), dtype=np.float32)
    x[:, 0] = 1
    x[:3, 2] = 1
    x[0, 1] = 1
    s = neighbours(sp.csr_matrix(x), k=1, shrink=0.0)
    assert s[2].indices.tolist() == [0]


def test_mathematically_equal_scores_from_different_terms_differ_by_at_most_their_rounding() -> None:
    """Review finding, kept as a bound rather than fixed: two items whose scores are equal as real numbers but are
    sums of *different* similarities can differ after rounding, and then the order between them is decided by it.

    Input: the 8 × 7 matrix below, k=5, shrink=0; user 7 (history 0, 1, 2, 4, 5) has items 3 and 6 left.
    score(3) = 1/sqrt(5) + 1/sqrt(8) + 1/sqrt(12) + 1/sqrt(12) and score(6) = 1/sqrt(5) + 1/sqrt(3) + 1/sqrt(8):
    equal, since 2/sqrt(12) = 1/sqrt(3). The reviewer found item 3 ranked 2nd (float32 sums). With similarities in
    units of 2^-32, 2 * round(2^32/sqrt(12)) = 2479700524 but round(2^32/sqrt(3)) = 2479700525, so item 3 is still
    one unit behind. No finite representation fixes this in general (deciding whether two sums of square roots are
    equal needs exact algebraic arithmetic). What baselines guarantees instead: equal similarities are equal (the
    test above), scores are exact sums of the rounded similarities (test_baselines), and a score is within half a
    unit per summed term of the real one, so this pair differs by at most 4 units.
    """
    x = np.array(
        [
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    history = sp.csr_matrix(x)
    model = ItemKNN(k=5, shrink=0.0)
    model.fit(history)
    user = np.array([7], dtype=np.int64)
    scores = model.score(user)[0]
    assert abs(scores[3] - scores[6]) <= 4  # 4 summed terms on the larger side, half a unit each, twice over
    assert scores[6] - scores[3] == 1  # the case as it stands: one unit, so item 3 ranks 2nd
    rank = ev.ranks(model.score(user), history[user], np.array([3], dtype=np.int64))
    assert rank.tolist() == [2]
