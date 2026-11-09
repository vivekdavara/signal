"""Candidate retrieval over the two-tower item vectors with FAISS (DESIGN.md, "Day 3: FAISS, ranking, the feature
store").

The user tower gives a unit vector per user and the item tower one per movie, so the best candidates are the movies with
the largest inner products: maximum inner product search. Three indexes:

- **flat** (``IndexFlatIP``): every movie scored, the exact answer. The reference.
- **hnsw** (``IndexHNSWFlat``): a navigable small-world graph; a query walks it greedily, keeping the ``ef_search``
  best nodes seen. ``Retriever`` searches with ``max(ef_search, k)``, where k is the number of candidates it asks FAISS
  for (``n + RESCORE_MARGIN``, more when widening, below): FAISS's search list stops at ``efSearch`` itself (review of
  day 3: at 16 for n = 200 it returned about 0.65 of the exact top 200; this docstring had said FAISS takes the
  maximum, which it doesn't). So "efSearch 200" for n = 200 searches with 216.
- **ivf** (``IndexIVFFlat``): k-means cells over the movies; a query scores only the movies in its ``nprobe`` nearest
  cells.

A user's history is excluded inside the search with an ID selector (``IDSelectorNot`` over their movies), so the
index returns the best ``n`` movies they have not seen, rather than the best ``n`` overall with their history to be
filtered afterwards (which would leave fewer than ``n`` for a heavy user). One query per call: that is the serving path
(one request, one user), and a selector belongs to one query.

The candidates come back in the protocol's order: rescored in float64 from the item vectors, best first, ties to the
smaller movie index, as ``evaluate.top_k`` ranks (review of day 3: FAISS's flat index returned tied movies largest index
first). FAISS scores in float32, so a movie just below the cut in float64 can come out just above it: the index is asked
for ``RESCORE_MARGIN`` more candidates than requested, and the cut is made after rescoring (the same review: one of 149
users' held-out movies, tied with a twin at the 30th place, was pushed out by such a movie). If the last candidate
fetched is not clearly below the n-th, a tie group may run past the margin (round 2: 50 tied movies, the cut inside
them, kept indices 10-14 instead of 0-4), so the search is repeated with twice as many until it is, or the index is
exhausted.

Builds run on one thread, so the same vectors give the same index (HNSW's parallel insertion is not deterministic).
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from typing import Any

if sys.platform == "darwin":
    # PyTorch's and faiss-cpu's macOS wheels each bundle their own libomp, and they only coexist in one order
    # (measured on day 3 with torch 2.15 and faiss-cpu 1.15.1, in .scratch/d3/order.py):
    # - faiss loaded first: torch's first parallel region segfaults (exit 139), with or without the flag below;
    # - torch first, without the flag: the second runtime aborts the process (OMP Error #15);
    # - torch first, with KMP_DUPLICATE_LIB_OK=TRUE: one-query searches after training work (2,000 of them), but a
    #   5-query batch search hung. Retriever only ever searches one query per call.
    # So torch is imported here, before faiss, whenever it is installed: every process gets the order the tests run
    # in, whichever module its caller imported first (scripts/bench_retrieval.py imported faiss first and crashed).
    # Intel documents the flag as unsupported; the tests check FAISS against a NumPy brute force in that setting.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    with contextlib.suppress(ImportError):  # no torch in this environment: only one runtime
        import torch  # noqa: F401

import faiss
import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]

KINDS = ("flat", "hnsw", "ivf")
RESCORE_MARGIN = 16
# How far below the n-th score (float64) the last fetched candidate must be before the cut is trusted: well above
# float32's rounding of a 64-dimensional dot product of unit vectors (about 64 x 6e-8).
CUT_TOLERANCE = 1e-5
FAISS_VERSION: str = faiss.__version__
# FAISS's bundled stubs lack some classes (SearchParametersHNSW) and type every index as the class first assigned.
_faiss: Any = faiss


@dataclass(frozen=True)
class IndexSpec:
    kind: str = "flat"
    m: int = 32  # hnsw: links per node
    ef_construction: int = 200  # hnsw: candidates kept while inserting
    ef_search: int = 200  # hnsw: candidates kept while searching; Retriever searches with max(ef_search, k)
    nlist: int = 64  # ivf: k-means cells
    nprobe: int = 8  # ivf: cells scored per query

    def check(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"index kind must be one of {KINDS}, got {self.kind!r}")
        for name in ("m", "ef_construction", "ef_search", "nlist", "nprobe"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if self.nprobe > self.nlist:
            raise ValueError(f"nprobe ({self.nprobe}) can't exceed nlist ({self.nlist})")

    def label(self) -> str:
        if self.kind == "hnsw":
            return f"hnsw (M={self.m}, efConstruction={self.ef_construction}, efSearch={self.ef_search})"
        if self.kind == "ivf":
            return f"ivf (nlist={self.nlist}, nprobe={self.nprobe})"
        return "flat (exact)"


def set_threads(n: int) -> int:
    """Set FAISS's OpenMP thread count; returns the previous one."""
    if type(n) is not int or n < 1:
        raise ValueError(f"threads must be an integer >= 1, got {n!r}")
    before: int = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(n)
    return before


def _single_thread() -> int:
    return set_threads(1)


def build(items: Float32Array, spec: IndexSpec) -> Any:
    """An index over ``items`` (one row per movie; row i gets id i)."""
    spec.check()
    if items.ndim != 2 or len(items) == 0:
        raise ValueError(f"expected a non-empty matrix of item vectors, got shape {items.shape}")
    if not np.isfinite(items).all():
        raise ValueError("an item vector is not finite")
    vectors = np.ascontiguousarray(items, dtype=np.float32)
    n, d = vectors.shape
    threads = _single_thread()
    index: Any
    try:
        if spec.kind == "flat":
            index = faiss.IndexFlatIP(d)
        elif spec.kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, spec.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = spec.ef_construction
        else:
            if spec.nlist > n:
                raise ValueError(f"nlist ({spec.nlist}) can't exceed the number of movies ({n})")
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, spec.nlist, faiss.METRIC_INNER_PRODUCT)
            index.cp.seed = 1234  # k-means initialisation (FAISS's default, written down)
            index.train(vectors)  # FAISS's Python wrapper keeps a reference to the quantizer
        index.add(vectors)
    finally:
        faiss.omp_set_num_threads(threads)
    return index


class Retriever:
    """The best ``n`` unseen movies for one user vector."""

    def __init__(self, items: Float32Array, spec: IndexSpec | None = None) -> None:
        self.spec = spec or IndexSpec()
        self.index = build(items, self.spec)
        self.items: Float32Array = np.array(items, dtype=np.float32)  # a copy: rescoring must see what was indexed
        self.n_items, self.dim = items.shape

    def _params(self, exclude: IntArray, k: int) -> tuple[Any, list[Any]]:
        keep: list[Any] = []  # the selectors hold raw pointers into each other: keep them alive for the search
        params: Any
        if self.spec.kind == "hnsw":
            params = _faiss.SearchParametersHNSW()
            params.efSearch = max(self.spec.ef_search, k)  # k includes the margin
        elif self.spec.kind == "ivf":
            params = _faiss.SearchParametersIVF()
            params.nprobe = self.spec.nprobe
        else:
            params = _faiss.SearchParameters()
        if len(exclude):
            ids = np.ascontiguousarray(exclude, dtype=np.int64)
            batch = faiss.IDSelectorBatch(len(ids), faiss.swig_ptr(ids))
            not_seen = faiss.IDSelectorNot(batch)
            keep += [ids, batch, not_seen]
            params.sel = not_seen
        return params, keep

    def search(self, user: Float32Array, n: int, exclude: IntArray | None = None) -> tuple[IntArray, Float64Array]:
        """Up to ``n`` movie ids, none of them in ``exclude``, and their inner products in float64: best first, ties to
        the smaller index.

        Fewer than ``n`` when fewer movies are left, or when an approximate index finds fewer (an IVF query whose
        ``nprobe`` cells hold fewer unseen movies).
        """
        if type(n) is not int or n < 1:
            raise ValueError(f"n must be an integer >= 1, got {n!r}")
        query = np.ascontiguousarray(np.asarray(user, dtype=np.float32).reshape(1, -1))
        if query.shape[1] != self.dim:
            raise ValueError(f"a {query.shape[1]}-dimensional query for {self.dim}-dimensional items")
        if not np.isfinite(query).all():
            raise ValueError("the query vector is not finite")
        excluded = np.empty(0, dtype=np.int64) if exclude is None else np.unique(np.asarray(exclude, dtype=np.int64))
        k = min(n + RESCORE_MARGIN, self.n_items)
        while True:
            params, keep = self._params(excluded, k)
            _, ids = self.index.search(query, k, params=params)
            del keep
            found = ids[0][ids[0] >= 0].astype(np.int64)
            scores = self.items[found].astype(np.float64) @ query[0].astype(np.float64)
            order = np.lexsort((found, -scores))  # score descending, then index ascending
            # Done when the index has nothing more to give, or the cut lies clearly above everything fetched beyond
            # it: then no movie left out could tie with or beat the n-th in float64. Otherwise (a tie group wider
            # than the margin straddles the cut; review round 2 of day 3) fetch twice as many and look again.
            exhausted = len(found) < k or k >= self.n_items
            if exhausted or len(found) <= n or scores[order[-1]] < scores[order[n - 1]] - CUT_TOLERANCE:
                break
            k = min(2 * k, self.n_items)
        order = order[:n]
        out_ids: IntArray = found[order]
        out_scores: Float64Array = scores[order]
        return out_ids, out_scores
