"""The baselines against a direct dense computation of their definitions."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from signalrec import evaluate as ev
from signalrec import movielens, prepare, synthetic
from signalrec.baselines import ItemKNN, Popularity, neighbours


def dense_neighbours(x: np.ndarray, k: int, shrink: float) -> np.ndarray:
    """DESIGN.md's definition, written out item by item."""
    n_items = x.shape[1]
    counts = x.sum(axis=0)
    s = np.zeros((n_items, n_items))
    for j in range(n_items):
        sims = []
        for i in range(n_items):
            if i == j:
                continue
            co = float((x[:, j] * x[:, i]).sum())
            denom = np.sqrt(counts[j] * counts[i]) + shrink
            sim = co / denom if denom > 0 else 0.0
            if sim > 0:
                sims.append((-sim, i))
        for neg, i in sorted(sims)[:k]:
            s[j, i] = -neg
    return s


matrices = hnp.arrays(np.float32, st.tuples(st.integers(1, 9), st.integers(1, 9)), elements=st.sampled_from([0.0, 1.0]))


@settings(max_examples=300, deadline=None)
@given(matrices, st.integers(1, 10), st.sampled_from([0.0, 0.5, 3.0]), st.integers(1, 4))
def test_neighbours_match_the_definition(x: np.ndarray, k: int, shrink: float, block: int) -> None:
    got = neighbours(sp.csr_matrix(x), k, shrink, block).toarray()
    np.testing.assert_allclose(got, dense_neighbours(x, k, shrink), rtol=1e-6, atol=0)


def test_ties_keep_the_smaller_item() -> None:
    # Items 1, 2 and 3 are each liked by the same single user as item 0: equal similarity to item 0.
    x = np.array([[1, 1, 1, 1]], dtype=np.float32)
    s = neighbours(sp.csr_matrix(x), k=2, shrink=0.0).toarray()
    assert s[0].tolist() == [0, 1, 1, 0]
    assert s[3].tolist() == [1, 1, 0, 0]


@settings(max_examples=150, deadline=None)
@given(matrices, st.integers(1, 10))
def test_item_knn_scores_sum_neighbour_similarities_over_the_history(x: np.ndarray, k: int) -> None:
    model = ItemKNN(k=k, shrink=1.0, block=2)
    model.fit(sp.csr_matrix(x))
    users = np.arange(x.shape[0], dtype=np.int64)
    np.testing.assert_allclose(model.score(users), x @ dense_neighbours(x, k, 1.0), rtol=1e-5, atol=1e-6)


def test_popularity_counts_likes() -> None:
    x = sp.csr_matrix(np.array([[1, 0, 1], [1, 0, 0]], dtype=np.float32))
    model = Popularity()
    model.fit(x)
    assert model.score(np.array([0, 1], dtype=np.int64)).tolist() == [[2, 0, 1], [2, 0, 1]]


def test_unfitted_models_refuse_to_score() -> None:
    with pytest.raises(RuntimeError):
        Popularity().score(np.array([0], dtype=np.int64))
    with pytest.raises(RuntimeError):
        ItemKNN().score(np.array([0], dtype=np.int64))


def test_bad_parameters_are_refused() -> None:
    x = sp.csr_matrix(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="k must"):
        neighbours(x, 0, 0.0)
    with pytest.raises(ValueError, match="shrink"):
        neighbours(x, 1, -1.0)


def test_item_knn_beats_popularity_on_the_synthetic_data(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The generator has per-user structure (latent factors), so neighbours should find more than popularity does.

    A sanity check of the whole path on synthetic data, not a claim about real users.
    """
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic")
    p = prepare.prepare(movielens.load("synthetic", raw), prepare.Params())
    history = p.history(include_val=False)
    results = {}
    for model in (Popularity(), ItemKNN(k=50)):
        model.fit(history)
        results[model.name] = ev.evaluate(model, history, p.val_items, "val").summary()["ndcg@10"]
    assert results["item-knn"] > results["popularity"]


def test_ties_keep_the_smaller_items_beyond_insertion_sort_sizes() -> None:
    # 64 movies liked by the same 3 users: every similarity is equal, so each movie's 10 neighbours must be the 10
    # smallest other indices. (NumPy sorts fewer than 16 elements by insertion sort, which is stable anyway.)
    x = np.ones((3, 64), dtype=np.float32)
    s = neighbours(sp.csr_matrix(x), k=10, shrink=0.0, block=7)
    for j in range(64):
        assert s[j].indices.tolist() == sorted(i for i in range(11) if i != j)[:10]
