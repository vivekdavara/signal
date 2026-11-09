"""Candidate retrieval over the two-tower item vectors with FAISS (DESIGN.md, "Day 3: FAISS, ranking, the feature
store").

The user tower gives a unit vector per user and the item tower one per movie, so the best candidates are the movies with
the largest inner products: maximum inner product search. Three indexes:

- **flat** (``IndexFlatIP``): every movie scored, the exact answer. The reference.
- **hnsw** (``IndexHNSWFlat``): a navigable small-world graph; a query walks it greedily, keeping the ``ef_search``
  best nodes seen. FAISS searches with ``max(ef_search, n)``, so asking for 200 candidates searches with at least 200.
- **ivf** (``IndexIVFFlat``): k-means cells over the movies; a query scores only the movies in its ``nprobe`` nearest
  cells.

A user's history is excluded inside the search with an ID selector (``IDSelectorNot`` over their movies), so the
index returns the best ``n`` movies they have not seen, rather than the best ``n`` overall with their history to be
filtered afterwards (which would leave fewer than ``n`` for a heavy user). One query per call: that is the serving path
(one request, one user), and a selector belongs to one query.

Builds run on one thread, so the same vectors give the same index (HNSW's parallel insertion is not deterministic).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

if sys.platform == "darwin":
    # PyTorch's and faiss-cpu's macOS wheels each bundle their own libomp. In a process that has used both (the tests,
    # an experiment that trains and then retrieves), the second runtime to start aborts the process with OMP Error #15
    # unless this is set. Intel documents it as an unsupported escape hatch; the tests run FAISS against a NumPy brute
    # force in that same process, so a wrong answer would fail them. Serving imports no torch (DESIGN.md, "Day 4"),
    # and Linux wheels use libgomp, which has no such check.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]

KINDS = ("flat", "hnsw", "ivf")
# FAISS's bundled stubs lack some classes (SearchParametersHNSW) and type every index as the class first assigned.
_faiss: Any = faiss


@dataclass(frozen=True)
class IndexSpec:
    kind: str = "flat"
    m: int = 32  # hnsw: links per node
    ef_construction: int = 200  # hnsw: candidates kept while inserting
    ef_search: int = 200  # hnsw: candidates kept while searching (FAISS uses max(ef_search, n))
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


def _single_thread() -> int:
    before: int = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    return before


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
        self.n_items, self.dim = items.shape

    def _params(self, exclude: IntArray) -> tuple[Any, list[Any]]:
        keep: list[Any] = []  # the selectors hold raw pointers into each other: keep them alive for the search
        params: Any
        if self.spec.kind == "hnsw":
            params = _faiss.SearchParametersHNSW()
            params.efSearch = self.spec.ef_search
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

    def search(self, user: Float32Array, n: int, exclude: IntArray | None = None) -> tuple[IntArray, Float32Array]:
        """Up to ``n`` movie ids, best first, none of them in ``exclude``, and their inner products.

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
        params, keep = self._params(excluded)
        k = min(n, self.n_items)
        scores, ids = self.index.search(query, k, params=params)
        del keep
        found = ids[0] >= 0
        out_ids: IntArray = ids[0][found].astype(np.int64)
        out_scores: Float32Array = scores[0][found].astype(np.float32)
        return out_ids, out_scores
