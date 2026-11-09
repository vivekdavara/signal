"""The features a recommendation is made from, computed offline from the processed data (DESIGN.md, "Day 3").

The same features come from Redis online (``featurestore.py``). Both sides produce a ``UserFeatures`` and a popularity
array, and the ranker's feature vectors are computed from those alone (``ranking.py``), so training-serving skew can
only come from the two sources disagreeing, which a test checks for every user.

- **Per user:** the most recent ``RECENT`` positives (oldest first, in the split's (time, movie id) order), every movie
  they liked (for the history filter), their positives per genre, and the sum and count of the release years of the
  movies they liked (for their typical year).
- **Per movie:** popularity, the number of positives it has received.
- **Static catalogue** (not features: they don't change with events): each movie's genres and release year.

Every count is an integer, so offline and online agree exactly, not within a tolerance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from signalrec.prepare import Prepared

IntArray = npt.NDArray[np.int64]

RECENT = 50
NO_YEAR = -1
_YEAR = re.compile(r"\((\d{4})\)$")


def release_year(title: str) -> int:
    """The year in a MovieLens title's trailing "(1995)", or ``NO_YEAR`` (some ml-latest-small titles have none)."""
    match = _YEAR.search(title.strip())
    return int(match.group(1)) if match else NO_YEAR


@dataclass(frozen=True)
class Catalogue:
    """Static movie data: genre names, each movie's genre indices and release year."""

    genre_names: tuple[str, ...]
    movie_genres: tuple[tuple[int, ...], ...]  # sorted, distinct genre indices per movie
    years: IntArray  # NO_YEAR when the title has none

    @property
    def n_items(self) -> int:
        return len(self.movie_genres)

    @classmethod
    def from_prepared(cls, p: Prepared) -> Catalogue:
        names = tuple(sorted({g for movie in p.genres for g in movie}))
        column = {name: i for i, name in enumerate(names)}
        movie_genres = tuple(tuple(sorted({column[g] for g in movie})) for movie in p.genres)
        years = np.array([release_year(t) for t in p.titles], dtype=np.int64)
        return cls(names, movie_genres, years)


@dataclass(frozen=True)
class UserFeatures:
    recent: IntArray  # the last RECENT positives, oldest first
    seen: IntArray  # every positive, sorted by movie index
    genre_counts: IntArray  # positives per genre, indexed like Catalogue.genre_names
    positives: int
    year_sum: int  # over the positives whose movie has a year
    year_count: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UserFeatures):
            return NotImplemented
        return (
            np.array_equal(self.recent, other.recent)
            and np.array_equal(self.seen, other.seen)
            and np.array_equal(self.genre_counts, other.genre_counts)
            and (self.positives, self.year_sum, self.year_count) == (other.positives, other.year_sum, other.year_count)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Event:
    """A positive: ``user`` liked ``movie`` at ``time`` (dense indices, Unix seconds)."""

    user: int
    movie: int
    time: int


def events(p: Prepared, include_val: bool) -> list[Event]:
    """Every training positive (and validation item, if ``include_val``) in replay order: by (time, movie, user).

    Restricted to one user that is the split's (time, movie id) order, so replaying them in this order rebuilds each
    user's recent list in the order offline features use.
    """
    users = [p.train_users]
    items = [p.train_items]
    times = [p.train_times]
    if include_val:
        users.append(np.arange(p.n_users, dtype=np.int64))
        items.append(p.val_items)
        times.append(p.val_times)
    u, i, t = (np.concatenate(a).astype(np.int64) for a in (users, items, times))
    order = np.lexsort((u, i, t))
    return [Event(int(u[k]), int(i[k]), int(t[k])) for k in order]


def user_sequences(p: Prepared, include_val: bool) -> list[IntArray]:
    """Each user's positives in the split's (time, movie id) order."""
    # Training rows are sorted by (user, time, item) and the validation item comes after all of a user's training rows.
    bounds = np.searchsorted(p.train_users, np.arange(p.n_users + 1, dtype=np.int64))
    out = []
    for u in range(p.n_users):
        items = p.train_items[bounds[u] : bounds[u + 1]].astype(np.int64)
        if include_val:
            items = np.append(items, np.int64(p.val_items[u]))
        out.append(items)
    return out


def user_features(items: IntArray, catalogue: Catalogue, recent: int = RECENT) -> UserFeatures:
    """One user's features from their positives in order (``recent``: how many of the latest to keep)."""
    counts = np.zeros(len(catalogue.genre_names), dtype=np.int64)
    for movie in items.tolist():
        for g in catalogue.movie_genres[movie]:
            counts[g] += 1
    years = catalogue.years[items]
    known = years[years != NO_YEAR]
    return UserFeatures(
        recent=items[len(items) - min(recent, len(items)) :].astype(np.int64),
        seen=np.unique(items).astype(np.int64),
        genre_counts=counts,
        positives=len(items),
        year_sum=int(known.sum()),
        year_count=len(known),
    )


def offline(p: Prepared, include_val: bool, recent: int = RECENT) -> tuple[list[UserFeatures], IntArray]:
    """Every user's features and every movie's popularity, from the training positives (+ validation items)."""
    catalogue = Catalogue.from_prepared(p)
    sequences = user_sequences(p, include_val)
    every = np.concatenate(sequences) if sequences else np.empty(0, dtype=np.int64)
    popularity = np.bincount(every, minlength=p.n_items).astype(np.int64)
    return [user_features(s, catalogue, recent) for s in sequences], popularity
