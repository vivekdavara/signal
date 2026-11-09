"""Review D3: ranking metrics at cut-offs above the candidate count ``n``.

``ranks_after`` gives a held-out movie that wasn't retrieved the rank ``n + 1`` ("outside every cut-off up to the list
length"), but ``experiment_ranking.metrics`` always reports Recall/NDCG at 10, 20 and 50 too. For any K > n, rank
``n + 1 <= K`` counts the miss as a hit, although the module docstring promises that a movie that wasn't retrieved
"counts as a miss at every cut-off".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from signalrec import experiment_ranking as er
from signalrec import movielens, prepare, synthetic
from signalrec import ranking as rk
from signalrec import twotower as tt
from signalrec.prepare import Prepared

TT = tt.Config(dim=16, hidden=16, batch_size=64, lr=1e-2, epochs=3, max_history=5)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=150, n_items=160)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


def test_a_movie_that_was_not_retrieved_is_a_miss_at_every_reported_cut_off() -> None:
    """Defect: with ``n`` = 5 candidates, a user whose held-out movie wasn't retrieved gets rank 6, and ``metrics``
    reports recall@10/20/50 = 1 and NDCG@10/20/50 = 1/log2(7) for that user.

    Input: two lists of 5 candidates, user 0's held-out movie absent (label -1), user 1's at position 2.
    Expected: every metric of user 0 is 0 (or the cut-offs above n aren't reported). Actual: recall@10 is 1.0.
    """
    features = np.zeros((2, 5, len(rk.FEATURES)))
    mask = np.ones((2, 5), dtype=bool)
    lists = rk.Lists(features, mask, np.array([-1, 2], dtype=np.int64))
    per_user = er.metrics(rk.ranks_after(None, lists), 5)
    missed = {name: float(values[0]) for name, values in per_user.items()}
    assert missed == dict.fromkeys(per_user, 0.0)


def test_the_reported_test_recall_at_50_never_exceeds_the_share_retrieved(prepared: Prepared) -> None:
    """Defect, end to end: ``run`` with ``n`` = 20 (as ``signalrec ranking --candidates 20`` in the CLI test) reports
    test recall@50 = 1.0 for both orders, though only about a quarter of the held-out movies are in the top 20.

    Input: the synthetic dataset (150 users, 160 movies), one linear ranker for one epoch, ``n`` = 20.
    Expected: recall@50 <= the share of users whose held-out movie was retrieved (a movie outside the top 20 can't be
    in the top 50 of a 20-movie list). Actual: recall@50 = 1.0 while the share retrieved is ~0.24.
    """
    result: dict[str, Any] = er.run(prepared, TT, (0,), 1, 20)
    share = result["test"]["held-out movie retrieved (share of users)"]
    assert share < 1.0  # some held-out movies were not retrieved
    # Adapted by the fix (fc5c122), which stopped reporting cut-offs above n: the assertion now covers every recall
    # that is reported, and Recall@50 must not be.
    for order in ("retrieved order", "re-ranked"):
        assert "recall@50" not in result["test"][order]
        recalls = {name: cell["value"] for name, cell in result["test"][order].items() if name.startswith("recall@")}
        assert recalls and all(value <= share for value in recalls.values()), (order, recalls)
