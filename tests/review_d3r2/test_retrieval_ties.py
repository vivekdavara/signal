"""Review D3 round 2: Retriever.search's rescoring margin and its tie rule."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from signalrec import evaluate as ev
from signalrec import retrieval as rt


def test_a_tie_group_wider_than_the_margin_keeps_the_smaller_indices() -> None:
    """Defect: ``search`` asks FAISS for only ``n + RESCORE_MARGIN`` candidates and applies the (score desc, index asc)
    rule to those. FAISS's flat index does not return a tie group's smallest indices, so when more than
    ``RESCORE_MARGIN`` movies tie across the cut, the cut keeps whichever tied movies FAISS happened to return, not the
    smallest indices the docstring and ``evaluate.top_k`` promise (twins: movies with identical vectors, which the
    module docstring says occur at the cut).

    Input: 60 two-dimensional movies, 50..59 at score 2 and 0..49 tied at score 1 (identical vectors); flat index;
    query (1, 0); n = 15; nothing excluded.
    Expected: ``evaluate.top_k``'s order, [50..59, 0, 1, 2, 3, 4]. Actual: [50..59, 10, 11, 12, 13, 14].
    """
    items = np.zeros((60, 2), dtype=np.float32)
    items[:, 0] = 1.0
    items[50:, 0] = 2.0
    query = np.array([1.0, 0.0], dtype=np.float32)
    n = 15
    ids, _ = rt.Retriever(items, rt.IndexSpec("flat")).search(query, n)
    scores = (items.astype(np.float64) @ query.astype(np.float64))[None, :]
    want = ev.top_k(scores, sp.csr_matrix((1, len(items))), n)[0]
    assert ids.tolist() == want.tolist()


def test_the_hnsw_search_list_is_the_documented_max_of_ef_search_and_n() -> None:
    """Claim mismatch (low severity): the module docstring says ``Retriever`` searches HNSW with
    ``max(ef_search, n)``, and ``IndexSpec.label`` (the benchmark tables' row names) reports ``efSearch`` as given; the
    code uses ``max(ef_search, n + RESCORE_MARGIN)``, so README's "efSearch 200" row at n = 200 searched with 216.
    (DESIGN.md and IndexSpec's field comment still say FAISS itself takes the maximum, which the day-3 review refuted.)

    Input: HNSW with ef_search = 200 over 300 random movies; search for n = 200.
    Expected: efSearch 200 (the docstring's max(200, 200)). Actual: 216.
    """
    rng = np.random.default_rng(0)
    items = rng.standard_normal((300, 8)).astype(np.float32)
    retriever = rt.Retriever(items, rt.IndexSpec("hnsw", ef_search=200))
    seen: list[int] = []
    original = retriever._params

    def spy(exclude: rt.IntArray, k: int) -> tuple[object, list[object]]:
        params, keep = original(exclude, k)
        seen.append(int(params.efSearch))
        return params, keep

    retriever._params = spy  # type: ignore[method-assign]
    retriever.search(items[0], 200)
    # Adapted with the fix: the docs now state the rule the code needs (FAISS must keep at least as many as it is
    # asked for, and it is asked for n + RESCORE_MARGIN), so the documented value is max(200, 200 + 16).
    assert seen == [max(200, 200 + rt.RESCORE_MARGIN)]


def test_scores_are_the_float64_dot_products_exactly() -> None:
    """Round 2's surviving mutant: rescoring in float32 and casting the result passed every test (they compared scores
    with a tolerance). The order and the cut are decided in float64, so the scores must be exactly those."""
    rng = np.random.default_rng(3)
    items = rng.normal(size=(500, 32)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    user = items[7] + 0.1 * rng.normal(size=32).astype(np.float32)
    ids, scores = rt.Retriever(items).search(user, 40)
    assert scores.dtype == np.float64
    want = items[ids].astype(np.float64) @ np.asarray(user, dtype=np.float32).astype(np.float64)
    assert np.array_equal(scores, want)
    assert not np.array_equal(scores, (items[ids] @ user.astype(np.float32)).astype(np.float64))  # float32 differs
