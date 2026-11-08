"""scripts/recency_ablation.py: recent-history matrices and item-kNN scored from them."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from signalrec import evaluate as ev
from signalrec import movielens, prepare, synthetic
from signalrec.baselines import ItemKNN
from signalrec.prepare import Prepared

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recency_ablation.py"
sys.path.insert(0, str(SCRIPT.parent))
import recency_ablation  # noqa: E402


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=80, n_items=120)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


@pytest.mark.parametrize("n", [1, 3, 0])
def test_last_n_keeps_each_users_most_recent_training_positives(prepared: Prepared, n: int) -> None:
    matrix = recency_ablation.last_n_matrix(prepared, n)
    for u in range(prepared.n_users):
        mine = prepared.train_items[prepared.train_users == u]  # already in (time, item) order
        want = mine if n == 0 else mine[-n:]
        assert sorted(matrix[u].indices.tolist()) == sorted(want.tolist())


def test_recent_knn_from_every_positive_is_item_knn(prepared: Prepared) -> None:
    history = prepared.history(include_val=False)
    knn = ItemKNN(k=5, shrink=0.0)
    knn.fit(history)
    scorer = recency_ablation.RecentKNN(knn, recency_ablation.last_n_matrix(prepared, 0))
    users = np.arange(prepared.n_users, dtype=np.int64)
    assert np.array_equal(scorer.score(users), knn.score(users))
    a = ev.evaluate(scorer, history, prepared.val_items, "val").summary()
    b = ev.evaluate(knn, history, prepared.val_items, "val").summary()
    assert a == b


def test_recent_knn_scores_only_from_the_recent_positives(prepared: Prepared) -> None:
    history = prepared.history(include_val=False)
    knn = ItemKNN(k=5, shrink=0.0)
    knn.fit(history)
    recent = recency_ablation.last_n_matrix(prepared, 1)
    scorer = recency_ablation.RecentKNN(knn, recent)
    assert knn.similarity is not None
    for u in range(prepared.n_users):
        (last,) = recent[u].indices.tolist()
        row = knn.similarity[[last]].toarray()[0].astype(np.float64)
        assert np.array_equal(scorer.score(np.array([u]))[0], row)
