"""The request path: one user in, their top k movies out (DESIGN.md, "Day 4: serving and experiments").

Stages, each timed:

1. **features**: the user's features and rated set from Redis, one MULTI/EXEC (``FeatureStore.serving_view``).
2. **user tower**: their vector from the last ``max_history`` positives, in NumPy (``tower.UserTower``).
3. **retrieve**: the best ``n_candidates`` movies by inner product (FAISS, exact by default), excluding inside the
   search every movie the filter would remove: the ones the user rated (or, for the offline protocol's comparison,
   only the ones they liked) and any excluded genres. Excluding in the search, not after, means a heavy user or a
   wide genre filter still gets ``n_candidates`` eligible movies.
4. **popularity**: the candidates' popularity from Redis, one HMGET.
5. **rank**: the ranker's eight features and scores (``ranking.py``), candidates re-ordered by score; the control arm
   of an experiment skips the model and keeps the retrieved order.
6. **filter**: the business rules checked again on the final list, then the top k. Nothing should be left to remove:
   a movie that is would be a bug in the exclusion above, so it raises (a 500), rather than being dropped silently.

The features read in stage 1 and the popularity in stage 4 are separate reads, so an event landing between them is
seen by one and not the other; both are counts that move by one per event, so a request ranks with features at most
one event apart.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from signalrec import ranking as rk
from signalrec import retrieval as rt
from signalrec.featurestore import FeatureStore
from signalrec.tower import Model

IntArray = npt.NDArray[np.int64]
Float64Array = npt.NDArray[np.float64]

N_CANDIDATES = 200
STAGES = ("features", "user tower", "retrieve", "popularity", "rank", "filter")
EXCLUDE = ("rated", "liked")
# FAISS's OpenMP threads for the process (None leaves them alone). A request searches one query, which gains nothing
# from 8 threads, and concurrent requests each starting a parallel region oversubscribe the cores: measured on day 4,
# with FAISS's default of 8 threads the server's own pipeline time at 16 concurrent requests was 9.3 ms (p50), and an
# open loop at 600 requests/s collapsed to 341 served per second. Requests run in parallel instead.
SEARCH_THREADS = 1


@dataclass(frozen=True)
class Recommendation:
    movies: IntArray  # dense movie indices, best first
    scores: Float64Array  # the ranker's scores (the retrieval scores when the ranker was skipped)
    candidates: int  # how many retrieval returned
    seconds: dict[str, float] = field(default_factory=dict)  # per stage, in STAGES order

    @property
    def total(self) -> float:
        return sum(self.seconds.values())


class Recommender:
    def __init__(
        self,
        model: Model,
        store: FeatureStore,
        n_candidates: int = N_CANDIDATES,
        index: rt.IndexSpec | None = None,
        search_threads: int | None = SEARCH_THREADS,
    ) -> None:
        model.check()
        if type(n_candidates) is not int or n_candidates < 1:
            raise ValueError(f"n_candidates must be an integer >= 1, got {n_candidates!r}")
        if store.catalogue.n_items != model.n_items or store.catalogue.genre_names != model.catalogue.genre_names:
            raise ValueError("the feature store's catalogue is not the model's")
        self.model = model
        self.store = store
        self.n_candidates = n_candidates
        self.retriever = rt.Retriever(model.items, index or rt.IndexSpec("flat"))
        if search_threads is not None:
            rt.set_threads(search_threads)
        self.context = rk.Context.build(model.catalogue, model.items)
        genre_of = {name: g for g, name in enumerate(model.catalogue.genre_names)}
        self._genre_index = genre_of
        members: list[list[int]] = [[] for _ in genre_of]
        for movie, genres in enumerate(model.catalogue.movie_genres):
            for g in genres:
                members[g].append(movie)
        self._genre_movies = [np.array(m, dtype=np.int64) for m in members]
        self._has_genre = np.zeros((len(genre_of), model.n_items), dtype=bool)
        for g, movies in enumerate(self._genre_movies):
            self._has_genre[g, movies] = True

    def genres(self, names: Sequence[str]) -> list[int]:
        """Genre names → indices; an unknown name is a ``KeyError`` naming it."""
        out = []
        for name in names:
            if name not in self._genre_index:
                raise KeyError(f"unknown genre {name!r}; the catalogue has {list(self._genre_index)}")
            out.append(self._genre_index[name])
        return sorted(set(out))

    def recommend(
        self,
        user: int,
        k: int = 10,
        exclude_genres: Sequence[str] = (),
        rank: bool = True,
        exclude: str = "rated",
    ) -> Recommendation:
        """The top ``k`` movies for dense user index ``user``; fewer when fewer than ``k`` movies are eligible."""
        if type(k) is not int or k < 1:
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        if exclude not in EXCLUDE:
            raise ValueError(f"exclude must be one of {EXCLUDE}, got {exclude!r}")
        genre_ids = self.genres(exclude_genres)
        clock = time.perf_counter
        seconds: dict[str, float] = {}

        t = clock()
        features, rated = self.store.serving_view(user)
        seconds["features"] = clock() - t

        t = clock()
        vector = self.model.tower.vector(features.recent)
        seconds["user tower"] = clock() - t

        t = clock()
        removed = [features.seen] if exclude == "liked" else [rated, features.seen]
        removed += [self._genre_movies[g] for g in genre_ids]
        excluded = np.unique(np.concatenate(removed)).astype(np.int64)
        n = max(self.n_candidates, k)
        candidates, retrieval_scores = self.retriever.search(vector, n, excluded)
        seconds["retrieve"] = clock() - t

        t = clock()
        popularity = np.zeros(self.model.n_items, dtype=np.int64)
        popularity[candidates] = self.store.popularity_of(candidates)
        seconds["popularity"] = clock() - t

        t = clock()
        if rank and len(candidates):
            matrix = rk.candidate_features(features, candidates, retrieval_scores, popularity, self.context)
            scores = self.model.ranker.finite_scores(matrix)
            order = np.argsort(-scores, kind="stable")  # ties keep the retrieved order, as RankerWeights.rerank
            movies, final_scores = candidates[order], scores[order]
        else:
            movies, final_scores = candidates, retrieval_scores
        seconds["rank"] = clock() - t

        t = clock()
        bad = np.isin(movies, excluded)
        if genre_ids:
            in_genre = np.asarray(self._has_genre[genre_ids].any(axis=0), dtype=bool)
            bad |= in_genre[movies]
        if bad.any():  # retrieval excluded these already; reaching here is a bug, not a filter doing its job
            raise AssertionError(f"{int(bad.sum())} excluded movies reached the final list")
        movies, final_scores = movies[:k], final_scores[:k]
        seconds["filter"] = clock() - t

        return Recommendation(movies.astype(np.int64), final_scores.astype(np.float64), len(candidates), seconds)
