"""retrieval.py: FAISS indexes against a brute-force answer computed in NumPy."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from signalrec import retrieval as rt

N, D = 400, 16


def _unit(rng: np.random.Generator, n: int, d: int = D) -> rt.Float32Array:
    x = rng.normal(size=(n, d)).astype(np.float32)
    out: rt.Float32Array = x / np.linalg.norm(x, axis=1, keepdims=True)
    return out


def _clustered(rng: np.random.Generator, n: int, clusters: int = 8) -> rt.Float32Array:
    centres = _unit(rng, clusters)
    x = centres[rng.integers(0, clusters, size=n)] + 0.15 * rng.normal(size=(n, D)).astype(np.float32)
    out: rt.Float32Array = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return out


def _brute_force(items: rt.Float32Array, user: rt.Float32Array, n: int, exclude: rt.IntArray) -> rt.IntArray:
    """Every movie scored in float64, best first, ties to the smaller id, the excluded ones dropped."""
    scores = items.astype(np.float64) @ user.astype(np.float64)
    order = np.argsort(-scores, kind="stable")
    out: rt.IntArray = order[~np.isin(order, exclude)][:n].astype(np.int64)
    return out


@pytest.fixture(scope="module")
def items() -> rt.Float32Array:
    return _clustered(np.random.default_rng(1), N)


@pytest.mark.parametrize(
    "spec",
    [
        rt.IndexSpec("flat"),
        # Exhaustive settings make the approximate indexes exact: every cell probed, a search list as long as the set.
        rt.IndexSpec("ivf", nlist=16, nprobe=16),
        rt.IndexSpec("hnsw", m=16, ef_construction=400, ef_search=N),
    ],
    ids=["flat", "ivf-all-cells", "hnsw-ef-n"],
)
def test_exact_settings_give_the_brute_force_answer(items: rt.Float32Array, spec: rt.IndexSpec) -> None:
    retriever = rt.Retriever(items, spec)
    rng = np.random.default_rng(2)
    for _ in range(20):
        user = _unit(rng, 1)[0]
        exclude = rng.choice(N, size=int(rng.integers(0, 60)), replace=False).astype(np.int64)
        ids, scores = retriever.search(user, 50, exclude)
        want = _brute_force(items, user, 50, exclude)
        assert ids.tolist() == want.tolist()
        np.testing.assert_allclose(scores, items[want].astype(np.float64) @ user.astype(np.float64), atol=1e-5)
        assert (np.diff(scores) <= 0).all()  # best first


@pytest.mark.parametrize("kind", rt.KINDS)
def test_excluded_movies_never_come_back(items: rt.Float32Array, kind: str) -> None:
    retriever = rt.Retriever(items, rt.IndexSpec(kind, nlist=16, nprobe=16))
    user = _unit(np.random.default_rng(3), 1)[0]
    best, _ = retriever.search(user, 10)
    ids, _ = retriever.search(user, 10, best)  # exclude exactly the movies it would have returned
    assert not set(ids.tolist()) & set(best.tolist())
    assert len(ids) == 10


def test_a_heavy_user_still_gets_n_candidates(items: rt.Float32Array) -> None:
    # The reason for excluding inside the search: filtering the top n afterwards leaves fewer than n.
    retriever = rt.Retriever(items)
    user = _unit(np.random.default_rng(4), 1)[0]
    top_300, _ = retriever.search(user, 300)
    ids, _ = retriever.search(user, 50, top_300)
    assert len(ids) == 50
    assert ids.tolist() == _brute_force(items, user, 50, top_300).tolist()


def test_fewer_movies_left_than_asked_for(items: rt.Float32Array) -> None:
    retriever = rt.Retriever(items)
    user = _unit(np.random.default_rng(5), 1)[0]
    exclude = np.arange(N - 7, dtype=np.int64)
    ids, scores = retriever.search(user, 50, exclude)
    assert sorted(ids.tolist()) == list(range(N - 7, N))
    assert len(scores) == 7
    ids, _ = retriever.search(user, 10, np.arange(N, dtype=np.int64))
    assert len(ids) == 0
    ids, _ = retriever.search(user, 10 * N)  # n above the catalogue: every movie once
    assert sorted(ids.tolist()) == list(range(N))


def test_repeated_exclusions_count_once(items: rt.Float32Array) -> None:
    retriever = rt.Retriever(items)
    user = _unit(np.random.default_rng(6), 1)[0]
    exclude = np.array([3, 3, 9, 9, 9], dtype=np.int64)
    assert retriever.search(user, 20, exclude)[0].tolist() == _brute_force(items, user, 20, exclude).tolist()


@pytest.mark.parametrize("kind", rt.KINDS)
def test_the_same_vectors_build_the_same_index(items: rt.Float32Array, kind: str) -> None:
    spec = rt.IndexSpec(kind, m=8, ef_search=16, nlist=16, nprobe=2)
    a, b = rt.Retriever(items, spec), rt.Retriever(items, spec)
    rng = np.random.default_rng(7)
    for _ in range(10):
        user = _unit(rng, 1)[0]
        assert a.search(user, 30)[0].tolist() == b.search(user, 30)[0].tolist()


def test_an_approximate_index_can_miss(items: rt.Float32Array) -> None:
    # One cell of 32: the search scores only that cell's movies, so the answer is not the exact one. (Otherwise the
    # recall measurement would be measuring nothing.)
    retriever = rt.Retriever(items, rt.IndexSpec("ivf", nlist=32, nprobe=1))
    rng = np.random.default_rng(8)
    misses = 0
    for _ in range(10):
        user = _unit(rng, 1)[0]
        ids, _ = retriever.search(user, 100)
        misses += len(set(_brute_force(items, user, 100, np.empty(0, dtype=np.int64)).tolist()) - set(ids.tolist()))
    assert misses > 0


@settings(max_examples=30, deadline=None)
@given(
    n_items=st.integers(1, 40),
    n=st.integers(1, 50),
    seed=st.integers(0, 2**32 - 1),
    excluded=st.integers(0, 45),
)
def test_flat_matches_brute_force_on_any_small_catalogue(n_items: int, n: int, seed: int, excluded: int) -> None:
    rng = np.random.default_rng(seed)
    items = _unit(rng, n_items, 4)
    user = _unit(rng, 1, 4)[0]
    exclude = rng.choice(n_items, size=min(excluded, n_items), replace=False).astype(np.int64)
    ids, _ = rt.Retriever(items).search(user, n, exclude)
    want = _brute_force(items, user, n, exclude)
    # Two movies can score equal in float32 but not in float64 (or the reverse), so compare as sets of scores
    # rounded to float32's precision when the order differs.
    if ids.tolist() != want.tolist():
        got_scores = np.sort((items[ids].astype(np.float64) @ user.astype(np.float64)).round(5))
        want_scores = np.sort((items[want].astype(np.float64) @ user.astype(np.float64)).round(5))
        assert got_scores.tolist() == want_scores.tolist()
    assert len(ids) == len(want)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kind": "lsh"}, "index kind"),
        ({"m": 0}, "m must be"),
        ({"ef_search": 1.5}, "ef_search must be"),
        ({"nlist": True}, "nlist must be"),
        ({"nlist": 4, "nprobe": 5}, "nprobe"),
    ],
)
def test_bad_specs_are_refused(items: rt.Float32Array, changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rt.Retriever(items, rt.IndexSpec(**changes))  # type: ignore[arg-type]


def test_bad_inputs_are_refused(items: rt.Float32Array) -> None:
    with pytest.raises(ValueError, match="nlist"):
        rt.Retriever(items[:10], rt.IndexSpec("ivf", nlist=16, nprobe=1))
    with pytest.raises(ValueError, match="non-empty"):
        rt.Retriever(np.empty((0, D), dtype=np.float32))
    bad = items.copy()
    bad[3, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        rt.Retriever(bad)
    retriever = rt.Retriever(items)
    with pytest.raises(ValueError, match="dimensional"):
        retriever.search(np.zeros(D + 1, dtype=np.float32), 5)
    with pytest.raises(ValueError, match="n must be"):
        retriever.search(items[0], 0)
    with pytest.raises(ValueError, match="not finite"):
        retriever.search(np.full(D, np.inf, dtype=np.float32), 5)
