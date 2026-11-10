"""Review D4 round 2: featurestore.py's event script after round 1's type checks."""

from __future__ import annotations

import pytest

from signalrec import features as fe
from signalrec.featurestore import FeatureStore, FeatureStoreError


def test_a_non_integer_count_fails_the_event_before_any_write(fresh: FeatureStore) -> None:
    """Defect (featurestore.py:46-71, _RECORD): round 1 checks every key's TYPE before the first write, but a key of
    the right type can still fail a later command: HINCRBY on a hash field that isn't an integer (the stats hash's
    ``positives``, a genre count, a popularity count) errors after SADD seen, RPUSH recent and the genre increments.
    The script isn't rolled back, so the event is half applied and a retry is a no-op (the movie is in ``seen``):
    exactly the round-1 defect, through another command. ``FeatureStoreError``'s own docstring names "a non-integer
    count" as what Redis may hold.

    Input: ``<ns>:u:0:stats`` with ``positives = "abc"``, then record(Event(0, 0, 0)).
    Expected: FeatureStoreError, and seen, recent and genres unchanged (empty).
    Actual: a bare redis.ResponseError ("hash value is not an integer"), with seen = {0}, recent = [0] and the genre
    counts written.
    """
    store = fresh
    ns = store.namespace
    store.client.hset(f"{ns}:u:0:stats", "positives", "abc")
    with pytest.raises(FeatureStoreError):
        store.record(fe.Event(0, 0, 0))
    leftovers = {
        "seen": store.client.smembers(f"{ns}:u:0:seen"),
        "recent": store.client.lrange(f"{ns}:u:0:recent", 0, -1),
        "genres": store.client.hgetall(f"{ns}:u:0:genres"),
        "popularity": store.client.hgetall(f"{ns}:popularity"),
    }
    assert all(not v for v in leftovers.values()), f"partial writes from a failed event: {leftovers}"


def test_a_popularity_count_at_int64_max_fails_the_event_before_any_write(fresh: FeatureStore) -> None:
    """Defect (featurestore.py:46-71, _RECORD): as above, through the popularity hash: an increment that would
    overflow fails the script's second-last write, after seen, recent, genres and stats were changed.

    Input: ``<ns>:popularity`` field 0 = 2**63 - 1, then record(Event(0, 0, 0)).
    Expected: FeatureStoreError, user 0's seen set empty.
    Actual: a bare redis.ResponseError ("increment or decrement would overflow"), seen = {0}.
    """
    store = fresh
    ns = store.namespace
    store.client.hset(f"{ns}:popularity", "0", str(2**63 - 1))
    with pytest.raises(FeatureStoreError):
        store.record(fe.Event(0, 0, 0))
    assert not store.client.smembers(f"{ns}:u:0:seen")


@pytest.mark.parametrize("method", ["replay", "replay_ratings"])
def test_replays_map_a_wrong_type_to_feature_store_error(fresh: FeatureStore, method: str) -> None:
    """Defect (featurestore.py:153-190, replay and replay_ratings): round 1 maps WRONGTYPE to FeatureStoreError in
    record and record_rating only. The replays (what ``signalrec load-store`` runs) still raise a bare
    redis.ResponseError, which cli.main doesn't catch: a traceback instead of "signalrec: ...", and callers catching
    FeatureStoreError (as the API does) miss it.

    Input: ``<ns>:u:1:rated`` set to a string, then replay / replay_ratings([Event(1, 0, 0)]).
    Expected: FeatureStoreError.
    Actual: redis.ResponseError ("Command # 1 ... of pipeline caused error: WRONGTYPE ...").
    """
    store = fresh
    store.client.set(f"{store.namespace}:u:1:rated", "x")
    with pytest.raises(FeatureStoreError):
        getattr(store, method)([fe.Event(1, 0, 0)])
