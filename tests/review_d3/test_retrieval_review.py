"""Review D3: retrieval.py's HNSW search-list claim and the retrieved order of tied scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from signalrec import evaluate as ev
from signalrec import experiment_ranking as er
from signalrec import movielens, prepare, synthetic
from signalrec import ranking as rk
from signalrec import retrieval as rt
from signalrec import twotower as tt
from signalrec.prepare import Prepared

TT = tt.Config(dim=16, hidden=16, batch_size=64, lr=1e-2, epochs=3, max_history=5)
N = 30


def _unit(rng: np.random.Generator, n: int, d: int) -> rt.Float32Array:
    x = rng.normal(size=(n, d)).astype(np.float32)
    out: rt.Float32Array = x / np.linalg.norm(x, axis=1, keepdims=True)
    return out


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=150, n_items=160)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


def test_hnsw_searches_with_at_least_n_candidates_as_documented() -> None:
    """Defect: the module docstring and ``IndexSpec.ef_search``'s comment say "FAISS searches with max(ef_search, n)",
    so asking for 200 candidates searches with at least 200. FAISS 1.15 doesn't: its stopping rule uses ``efSearch``
    itself, so ``ef_search`` below ``n`` returns a much worse list (recall ~0.65 of the exact top 200 at 16).

    Input: 2,000 random unit vectors in 32 dimensions, HNSW (M=32, efConstruction=200) with ef_search 16 and with
    ef_search 200, 10 queries for n = 200.
    Expected (by the documented rule both search with 200): the same ids. Actual: they differ.
    """
    rng = np.random.default_rng(0)
    items = _unit(rng, 2000, 32)
    low = rt.Retriever(items, rt.IndexSpec("hnsw", ef_search=16))
    at_n = rt.Retriever(items, rt.IndexSpec("hnsw", ef_search=200))
    for user in _unit(rng, 10, 32):
        assert low.search(user, 200)[0].tolist() == at_n.search(user, 200)[0].tolist()


def test_flat_search_orders_tied_movies_like_the_protocol() -> None:
    """Defect: the flat index is "the exact answer, the reference", and the protocol (``evaluate.top_k`` and
    ``evaluate.ranks``) breaks score ties by the smaller movie index. FAISS's flat search returns tied movies in
    descending index order, so the retrieved order is not the protocol's order whenever scores tie.

    Input: 10 identical item vectors, a query along them, n = 5.
    Expected: [0, 1, 2, 3, 4] (``evaluate.top_k``'s answer). Actual: [4, 3, 2, 1, 0].
    """
    items = np.zeros((10, 4), dtype=np.float32)
    items[:, 0] = 1.0
    user = np.array([1, 0, 0, 0], dtype=np.float32)
    ids, _ = rt.Retriever(items).search(user, 5)
    no_history = prepare.interaction_matrix(np.empty(0, np.int64), np.empty(0, np.int64), 1, 10)
    want = ev.top_k((items.astype(np.float64) @ user.astype(np.float64))[None, :], no_history, 5)[0]
    assert ids.tolist() == want.tolist()


def test_the_retrieved_order_is_the_full_ranking_when_scores_tie(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect, through ``build_lists``: experiment_ranking's docstring says label + 1 is the held-out movie's rank in
    the full ranking (so "Recall@K and NDCG@K are the full-ranking metrics for K up to n"); the existing test checks it
    on vectors without ties. With a tie the FAISS order puts the larger index first, so label + 1 is off by one from
    ``evaluate.ranks`` (and ``rerank``'s "ties keep the retrieved order" inherits the reversed order).

    Input: the synthetic dataset with the two-tower model replaced by fixed embeddings in which movie 2k + 1 has the
    same vector as movie 2k; validation lists of 30.
    Expected: ``ranks_after(None, lists)`` == min(evaluate's rank, n + 1) for every user. Actual: ~20 of 149 differ.
    """
    rng = np.random.default_rng(0)
    items = _unit(rng, prepared.n_items, 8)
    items[1::2] = items[0 : len(items[1::2]) * 2 : 2]
    embeddings = tt.Embeddings(_unit(rng, prepared.n_users, 8), items)

    def fake_train(*args: Any, **kwargs: Any) -> tuple[None, list[Any]]:
        return None, []

    def fake_embed(*args: Any, **kwargs: Any) -> tt.Embeddings:
        return embeddings

    monkeypatch.setattr(tt, "train", fake_train)
    monkeypatch.setattr(tt, "embed", fake_embed)
    lists = er.build_lists(prepared, TT, include_val=False, n=N).lists
    history = prepared.history(include_val=False)
    full = ev.evaluate(embeddings, history, prepared.val_items, "val").ranks
    # The precondition: some held-out movie in the top n has an unseen twin (a tie the order must break).
    twins = prepared.val_items ^ 1
    tied = [t < prepared.n_items and not history[u, t] for u, t in enumerate(twins.tolist())]
    assert (np.array(tied) & (full <= N)).any()
    got = rk.ranks_after(None, lists)
    want = np.where(full <= N, full, N + 1)
    assert got.tolist() == want.tolist(), f"{int((got != want).sum())} of {len(got)} users differ"
