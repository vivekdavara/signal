"""The request path: one user in, their top k movies out (DESIGN.md, "Day 4: serving and experiments").

Stages, each timed:

1. **features**: the user's features and rated set from Redis, one MULTI/EXEC (``FeatureStore.serving_view``).
2. **user tower**: their vector from the last ``max_history`` positives, in NumPy (``tower.UserTower``).
3. **retrieve**: the best ``n_candidates`` movies by inner product (FAISS, exact by default; a one-query search runs
   on the calling thread: measured on day 4, it starts no OpenMP worker whatever FAISS's thread setting), excluding
   inside the search every movie the filter would remove: the ones the user rated (or, for the offline protocol's
   comparison, only the ones they liked) and any excluded genres. Excluding in the search, not after, means a heavy
   user or a wide genre filter still gets ``n_candidates`` eligible movies.
4. **popularity**: the candidates' popularity from Redis, one HMGET.
5. **rank**: the ranker's eight features and scores (``ranking.py``), candidates re-ordered by score; the control arm
   of an experiment skips the model and keeps the retrieved order.
6. **filter**: the business rules checked again on the final list, then the top k. Nothing should be left to remove:
   a movie that is would be a bug in the exclusion above, so it raises (a 500), rather than being dropped silently.

**New users (day 5).** A user with no positives yet (a new MovieLens id, or one who has only rated movies below the
positive threshold) gets the most popular eligible movies instead: the two-tower model was trained only on positives
that had at least one earlier positive, so its vector for an empty history was never trained (``scripts/cold_start.py``
measures both). Popularity is read whole (one HGETALL) and ties go to the smaller movie index, as the offline
baseline's; the ranker is skipped, as its features include the retrieval score. ``user=None`` is a user the store has
never seen: nothing to read, nothing rated. ``Recommendation.source`` says which path answered.

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

from signalrec import features as fe
from signalrec import ranking as rk
from signalrec import retrieval as rt
from signalrec.featurestore import FeatureStore
from signalrec.tower import Model

IntArray = npt.NDArray[np.int64]
Float64Array = npt.NDArray[np.float64]

N_CANDIDATES = 200
STAGES = ("features", "user tower", "retrieve", "popularity", "rank", "filter")
EXCLUDE = ("rated", "liked")
SOURCES = ("two-tower", "popularity")


@dataclass(frozen=True)
class Recommendation:
    movies: IntArray  # dense movie indices, best first
    scores: Float64Array  # the ranker's scores (the retrieval scores when the ranker was skipped)
    candidates: int  # how many retrieval returned
    seconds: dict[str, float] = field(default_factory=dict)  # per stage, in STAGES order
    source: str = "two-tower"  # or "popularity": a user with no positives (SOURCES)

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
    ) -> None:
        model.check()
        if type(n_candidates) is not int or n_candidates < 1:
            raise ValueError(f"n_candidates must be an integer >= 1, got {n_candidates!r}")
        if store.catalogue.n_items != model.n_items or store.catalogue.genre_names != model.catalogue.genre_names:
            raise ValueError("the feature store's catalogue is not the model's")
        # The tower reads the last max_history positives from the store's recent list, which keeps store.recent of
        # them: a longer history would be cut silently, serving another model than the one evaluated (review of day 4).
        if model.tower.max_history > store.recent:
            raise ValueError(
                f"the model reads {model.tower.max_history} recent positives but the store keeps {store.recent}"
            )
        self.model = model
        self.store = store
        self.n_candidates = n_candidates
        self.retriever = rt.Retriever(model.items, index or rt.IndexSpec("flat"))
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
        self._nobody = fe.user_features(np.empty(0, dtype=np.int64), model.catalogue)

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
        user: int | None,
        k: int = 10,
        exclude_genres: Sequence[str] = (),
        rank: bool = True,
        exclude: str = "rated",
    ) -> Recommendation:
        """The top ``k`` movies for dense user index ``user`` (None: a user the store has never seen); fewer when fewer
        than ``k`` movies are eligible."""
        if type(k) is not int or k < 1:
            raise ValueError(f"k must be an integer >= 1, got {k!r}")
        if exclude not in EXCLUDE:
            raise ValueError(f"exclude must be one of {EXCLUDE}, got {exclude!r}")
        genre_ids = self.genres(exclude_genres)
        clock = time.perf_counter
        seconds: dict[str, float] = {}

        t = clock()
        if user is None:
            features, rated = self._nobody, np.empty(0, dtype=np.int64)
        else:
            features, rated = self.store.serving_view(user)
        seconds["features"] = clock() - t

        removed = [features.seen] if exclude == "liked" else [rated, features.seen]
        removed += [self._genre_movies[g] for g in genre_ids]
        excluded = np.unique(np.concatenate(removed)).astype(np.int64)
        n = max(self.n_candidates, k)
        if len(features.recent) == 0:
            return self._most_popular(k, n, excluded, genre_ids, seconds)

        t = clock()
        vector = self.model.tower.vector(features.recent)
        seconds["user tower"] = clock() - t

        t = clock()
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
        self._check(movies, excluded, genre_ids)
        movies, final_scores = movies[:k], final_scores[:k]
        seconds["filter"] = clock() - t

        return Recommendation(movies.astype(np.int64), final_scores.astype(np.float64), len(candidates), seconds)

    def _most_popular(
        self, k: int, n: int, excluded: IntArray, genre_ids: list[int], seconds: dict[str, float]
    ) -> Recommendation:
        """A user with no positives: the ``n`` most popular eligible movies (ties to the smaller index), top ``k``."""
        clock = time.perf_counter
        seconds["user tower"] = 0.0
        t = clock()
        popularity = self.store.popularity()
        seconds["popularity"] = clock() - t

        t = clock()
        order = np.argsort(-popularity, kind="stable")
        eligible = np.ones(self.model.n_items, dtype=bool)
        eligible[excluded] = False
        candidates = order[eligible[order]][:n].astype(np.int64)
        seconds["retrieve"] = clock() - t
        seconds["rank"] = 0.0

        t = clock()
        self._check(candidates, excluded, genre_ids)
        movies = candidates[:k]
        seconds["filter"] = clock() - t
        timed = {stage: seconds[stage] for stage in STAGES}
        scores = popularity[movies].astype(np.float64)
        return Recommendation(movies, scores, len(candidates), timed, source="popularity")

    def _check(self, movies: IntArray, excluded: IntArray, genre_ids: list[int]) -> None:
        bad = np.isin(movies, excluded)
        if genre_ids:
            in_genre = np.asarray(self._has_genre[genre_ids].any(axis=0), dtype=bool)
            bad |= in_genre[movies]
        if bad.any():  # retrieval excluded these already; reaching here is a bug, not a filter doing its job
            raise AssertionError(f"{int(bad.sum())} excluded movies reached the final list")
