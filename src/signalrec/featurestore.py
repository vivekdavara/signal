"""The online feature store in Redis (DESIGN.md, "Day 3: FAISS, ranking, the feature store"; port 6383).

Keys, under a namespace (``signal`` by default; the tests use a fresh one each):

- ``<ns>:u:<user>:recent``: a list of the user's last ``RECENT`` movies, oldest first (RPUSH, then LTRIM).
- ``<ns>:u:<user>:seen``: a set of every movie they liked (the history filter).
- ``<ns>:u:<user>:genres``: a hash, genre index → positives.
- ``<ns>:u:<user>:stats``: a hash with ``positives``, ``year_sum`` and ``year_count``.
- ``<ns>:u:<user>:rated``: a set of every movie they rated, liked or not (day 4: serving never recommends a movie the
  user has rated; ``seen`` holds positives only, because the processed data keeps only positives).
- ``<ns>:popularity``: a hash, movie index → positives.

An event is applied by one Lua script, so it is atomic (a reader sees all of an event's updates or none) and
**idempotent**: the script adds the movie to ``seen`` first and changes nothing else if it was already there, so a
redelivered event is a no-op. That matches the data: a (user, movie) pair is a positive at most once.

Events must arrive in time order per user for ``recent`` to be in time order: the store appends. The replay of the
processed data does that (``features.events``); a live event that arrives late is appended as the most recent, a
limitation the README states.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import redis

from signalrec.features import NO_YEAR, RECENT, Catalogue, Event, IntArray, UserFeatures

DEFAULT_URL = "redis://127.0.0.1:6383/0"
DEFAULT_NAMESPACE = "signal"
# Letters, digits, '-' and '_' only: ':' separates key parts, and SCAN's glob characters (* ? [ ] \) in a namespace
# made clear() delete other namespaces' keys (review of day 3).
_NAMESPACE = re.compile(r"[A-Za-z0-9_-]+")
INT64_MAX = 2**63 - 1

# KEYS: recent, seen, genres, stats, popularity, rated. ARGV: movie, recent cap, year (or -1), genre indices...
# A positive is also a rating, so it joins ``rated`` in the same script (day 4).
# A script is atomic for other clients but not rolled back if a command fails halfway, so every key's type is checked
# before the first write (review of day 4: a rated key of another type failed the last SADD and kept the rest).
_RECORD = """
local expected = {'list', 'set', 'hash', 'hash', 'hash', 'set'}
for i = 1, 6 do
  local t = redis.call('TYPE', KEYS[i])['ok']
  if t ~= 'none' and t ~= expected[i] then
    return redis.error_reply('WRONGTYPE ' .. KEYS[i] .. ' holds a ' .. t .. ', not a ' .. expected[i])
  end
end
if redis.call('SADD', KEYS[2], ARGV[1]) == 0 then
  return 0
end
redis.call('RPUSH', KEYS[1], ARGV[1])
redis.call('LTRIM', KEYS[1], -tonumber(ARGV[2]), -1)
for i = 4, #ARGV do
  redis.call('HINCRBY', KEYS[3], ARGV[i], 1)
end
redis.call('HINCRBY', KEYS[4], 'positives', 1)
if tonumber(ARGV[3]) >= 0 then
  redis.call('HINCRBY', KEYS[4], 'year_sum', ARGV[3])
  redis.call('HINCRBY', KEYS[4], 'year_count', 1)
end
redis.call('HINCRBY', KEYS[5], ARGV[1], 1)
redis.call('SADD', KEYS[6], ARGV[1])
return 1
"""


class FeatureStoreError(RuntimeError):
    """Redis holds what the store didn't write (a wrong type, a non-integer count, a movie outside the catalogue)."""


@dataclass(frozen=True)
class ReplayStats:
    events: int
    applied: int  # the rest were duplicates
    seconds: float


class FeatureStore:
    def __init__(
        self, client: redis.Redis, catalogue: Catalogue, namespace: str = DEFAULT_NAMESPACE, recent: int = RECENT
    ) -> None:
        if not isinstance(namespace, str) or not _NAMESPACE.fullmatch(namespace):
            raise ValueError(f"namespace must be letters, digits, '-' or '_', got {namespace!r}")
        if type(recent) is not int or recent < 1:
            raise ValueError(f"recent must be an integer >= 1, got {recent!r}")
        self.client = client
        self.catalogue = catalogue
        self.namespace = namespace
        self.recent = recent
        self._record: Any = client.register_script(_RECORD)

    @classmethod
    def connect(cls, catalogue: Catalogue, url: str = DEFAULT_URL, namespace: str = DEFAULT_NAMESPACE) -> FeatureStore:
        return cls(redis.Redis.from_url(url), catalogue, namespace)

    def _user_keys(self, user: int) -> list[str]:
        base = f"{self.namespace}:u:{user}"
        return [f"{base}:recent", f"{base}:seen", f"{base}:genres", f"{base}:stats"]

    def _rated_key(self, user: int) -> str:
        return f"{self.namespace}:u:{user}:rated"

    @property
    def _popularity_key(self) -> str:
        return f"{self.namespace}:popularity"

    def _args(self, event: Event) -> tuple[list[str], list[int]]:
        # Integers only: a user of 3.0 or True would be written under u:3.0 or u:True, keys user(3) never reads
        # (review of day 3). NumPy integers are fine; bools are not.
        for name in ("user", "movie"):
            value = getattr(event, name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise ValueError(f"event {name} must be an integer, got {value!r}")
        if not 0 <= event.movie < self.catalogue.n_items:
            raise ValueError(f"movie {event.movie} is outside the catalogue of {self.catalogue.n_items}")
        if event.user < 0:
            raise ValueError(f"user {event.user} is negative")
        year = int(self.catalogue.years[event.movie])
        keys = [*self._user_keys(int(event.user)), self._popularity_key, self._rated_key(int(event.user))]
        return keys, [
            int(event.movie),
            self.recent,
            year if year != NO_YEAR else -1,
            *self.catalogue.movie_genres[event.movie],
        ]

    def record(self, event: Event) -> bool:
        """Apply one event; False if it was already applied (a duplicate). A key of the wrong type is a
        ``FeatureStoreError`` and nothing is written."""
        keys, args = self._args(event)
        try:
            return bool(self._record(keys=keys, args=args))
        except redis.ResponseError as exc:
            if "WRONGTYPE" in str(exc):
                raise FeatureStoreError(f"user {event.user}: {exc}") from None
            raise

    def record_rating(self, event: Event) -> bool:
        """A rating that is not a positive: the movie joins the user's ``rated`` set only (no feature changes).
        False if it was already there."""
        keys, _ = self._args(event)
        try:
            return bool(self.client.sadd(keys[5], int(event.movie)))
        except redis.ResponseError as exc:  # one command: it fails whole
            raise FeatureStoreError(f"user {event.user}: {exc}") from None

    def replay_ratings(self, events: Iterable[Event], batch: int = 1000) -> ReplayStats:
        """``record_rating`` for each event, ``batch`` per round trip; as idempotent as ``replay``."""
        t0 = time.perf_counter()
        total = applied = pending = 0
        pipe = self.client.pipeline(transaction=False)
        for event in events:
            keys, _ = self._args(event)
            pipe.sadd(keys[5], int(event.movie))
            pending += 1
            total += 1
            if pending == batch:
                applied += sum(int(r) for r in pipe.execute())
                pending = 0
        if pending:
            applied += sum(int(r) for r in pipe.execute())
        return ReplayStats(total, applied, time.perf_counter() - t0)

    def replay(self, events: Iterable[Event], batch: int = 1000) -> ReplayStats:
        """Apply events in order, ``batch`` scripts per round trip (a pipeline, not a transaction: each script is
        atomic on its own, and order is kept).

        An invalid event raises before its batch is sent; earlier batches have been applied (replaying again is safe:
        applied events are duplicates).
        """
        t0 = time.perf_counter()
        total = applied = 0
        pipe = self.client.pipeline(transaction=False)
        pending = 0
        for event in events:
            keys, args = self._args(event)
            self._record(keys=keys, args=args, client=pipe)
            pending += 1
            total += 1
            if pending == batch:
                applied += sum(int(r) for r in pipe.execute())
                pending = 0
        if pending:
            applied += sum(int(r) for r in pipe.execute())
        return ReplayStats(total, applied, time.perf_counter() - t0)

    def user(self, user: int) -> UserFeatures:
        """One user's features (all zeros and empty for a user with no events)."""
        return self._read(user, with_rated=False)[0]

    def serving_view(self, user: int) -> tuple[UserFeatures, IntArray]:
        """One user's features and every movie they rated (sorted), from one MULTI/EXEC: what a request reads."""
        features, rated = self._read(user, with_rated=True)
        assert rated is not None
        return features, rated

    def _read(self, user: int, with_rated: bool) -> tuple[UserFeatures, IntArray | None]:
        # As for events: user(3.0) would read u:3.0, not user 3's keys (review round 2 of day 3).
        if isinstance(user, bool) or not isinstance(user, int | np.integer) or user < 0:
            raise ValueError(f"user must be an integer >= 0, got {user!r}")
        user = int(user)
        recent_key, seen_key, genres_key, stats_key = self._user_keys(user)
        pipe = self.client.pipeline(transaction=True)  # one consistent snapshot: MULTI/EXEC
        pipe.lrange(recent_key, 0, -1)
        pipe.smembers(seen_key)
        pipe.hgetall(genres_key)
        pipe.hgetall(stats_key)
        if with_rated:
            pipe.smembers(self._rated_key(user))
        try:
            replies = pipe.execute()
        except redis.ResponseError as exc:  # WRONGTYPE: a key holds another type
            raise FeatureStoreError(f"user {user}: {exc}") from None
        recent_raw, seen_raw, genres_raw, stats_raw = replies[:4]
        n_genres = len(self.catalogue.genre_names)
        genre_counts = np.zeros(n_genres, dtype=np.int64)
        for genre, count in genres_raw.items():
            g = self._int(genre, f"user {user}'s genre index")
            if not 0 <= g < n_genres:
                raise FeatureStoreError(f"user {user}: genre index {g} outside the {n_genres} genres")
            genre_counts[g] = self._int(count, f"user {user}'s genre count")
        stats = {self._text(k): self._int(v, f"user {user}'s {self._text(k)}") for k, v in stats_raw.items()}
        unknown = set(stats) - {"positives", "year_sum", "year_count"}
        if unknown:
            raise FeatureStoreError(f"user {user}: unexpected stats fields {sorted(unknown)}")
        recent = np.array([self._movie(m, user) for m in recent_raw], dtype=np.int64)
        seen = np.array(sorted(self._movie(m, user) for m in seen_raw), dtype=np.int64)
        rated = None
        if with_rated:
            rated = np.array(sorted(self._movie(m, user) for m in replies[4]), dtype=np.int64)
        features = UserFeatures(
            recent=recent,
            seen=seen,
            genre_counts=genre_counts,
            positives=stats.get("positives", 0),
            year_sum=stats.get("year_sum", 0),
            year_count=stats.get("year_count", 0),
        )
        return features, rated

    def popularity_of(self, movies: IntArray) -> IntArray:
        """Positives for each of ``movies`` (one HMGET; 0 for a movie nobody has liked): what a request reads, instead
        of the whole hash (4.8 ms on ml-1m, day 3)."""
        ids = [int(m) for m in np.asarray(movies, dtype=np.int64)]
        for m in ids:
            if not 0 <= m < self.catalogue.n_items:
                raise ValueError(f"movie {m} is outside the catalogue of {self.catalogue.n_items}")
        out = np.zeros(len(ids), dtype=np.int64)
        if not ids:
            return out
        try:
            raw = self.client.hmget(self._popularity_key, ids)
        except redis.ResponseError as exc:
            raise FeatureStoreError(f"popularity: {exc}") from None
        for i, count in enumerate(raw):
            if count is not None:
                out[i] = self._int(count, "a popularity count")
        return out

    def popularity(self) -> IntArray:
        """Positives per movie, as an array over the catalogue."""
        try:
            raw = self.client.hgetall(self._popularity_key)
        except redis.ResponseError as exc:
            raise FeatureStoreError(f"popularity: {exc}") from None
        out = np.zeros(self.catalogue.n_items, dtype=np.int64)
        for movie, count in raw.items():
            out[self._movie(movie, None)] = self._int(count, "a popularity count")
        return out

    def clear(self) -> int:
        """Delete every key of this namespace; returns how many."""
        deleted = 0
        for key in self.client.scan_iter(match=f"{self.namespace}:*", count=1000):
            deleted += int(self.client.delete(key))
        return deleted

    @staticmethod
    def _text(raw: Any) -> str:
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    @staticmethod
    def _int(raw: Any, what: str) -> int:
        text = FeatureStore._text(raw)
        # int() accepts " 7", "+7" and "7_0"; Redis's HINCRBY writes none of them. Every value the store writes is a
        # count, a sum of years or an index: never negative, and within HINCRBY's int64 (review of day 3: -1 and 2^70
        # were returned as features, or raised OverflowError).
        if not (text.isascii() and text.isdigit()):
            raise FeatureStoreError(f"{what} is {text!r}, not a non-negative integer")
        value = int(text)
        if value > INT64_MAX:
            raise FeatureStoreError(f"{what} is {text!r}, beyond what Redis can hold")
        return value

    def _movie(self, raw: Any, user: int | None) -> int:
        movie = self._int(raw, "a movie index")
        if not 0 <= movie < self.catalogue.n_items:
            owner = "popularity" if user is None else f"user {user}"
            raise FeatureStoreError(f"{owner}: movie {movie} outside the catalogue of {self.catalogue.n_items}")
        return movie
