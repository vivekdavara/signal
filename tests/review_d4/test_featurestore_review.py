"""Review D4: featurestore.py's event atomicity with the day-4 rated set."""

from __future__ import annotations

import pytest
import redis

from signalrec import features as fe
from signalrec.featurestore import FeatureStore, FeatureStoreError


def test_an_event_that_fails_on_the_rated_set_leaves_no_partial_writes(fresh: FeatureStore) -> None:
    """Defect (featurestore.py:44-60, _RECORD): the day-4 ``SADD rated`` is the script's last write. A Lua script is
    atomic for other clients but not rolled back on an error, so when the rated key holds another type, SADD seen,
    RPUSH recent, the genre/stats/popularity increments are all kept and the call errors. The store's docstring
    promises "a reader sees all of an event's updates or none", and the event can never be completed: a retry finds
    the movie in ``seen`` and returns 0 (``applied: False``), so ``rated`` never gets it.

    Input: ``<ns>:u:0:rated`` set to a string, then record(Event(0, 0, 0)).
    Expected: the call fails and the user's seen set, recent list and the popularity hash are unchanged (empty).
    Actual: the call fails, and seen = {0}, recent = [0], popularity {0: 1} were written.
    """
    store = fresh
    ns = store.namespace
    store.client.set(f"{ns}:u:0:rated", "x")
    with pytest.raises((redis.ResponseError, FeatureStoreError)):
        store.record(fe.Event(0, 0, 0))
    leftovers = {
        "seen": store.client.smembers(f"{ns}:u:0:seen"),
        "recent": store.client.lrange(f"{ns}:u:0:recent", 0, -1),
        "popularity": store.client.hgetall(f"{ns}:popularity"),
        "stats": store.client.hgetall(f"{ns}:u:0:stats"),
    }
    assert all(not v for v in leftovers.values()), f"partial writes from a failed event: {leftovers}"
