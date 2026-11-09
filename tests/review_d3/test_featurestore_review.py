"""Review D3: featurestore.py's namespace isolation, read-back validation and event validation.

Needs Redis (``SIGNALREC_REDIS_URL``, default redis://127.0.0.1:6383/0); skips without one unless
``SIGNALREC_REQUIRE_REDIS=1``. Every key written is under a fresh ``test-<uuid>`` prefix and deleted afterwards.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import numpy as np
import pytest
import redis

from signalrec import features as fe
from signalrec.featurestore import FeatureStore, FeatureStoreError

URL = os.environ.get("SIGNALREC_REDIS_URL", "redis://127.0.0.1:6383/0")
CATALOGUE = fe.Catalogue(("Drama", "War"), ((0,), (0, 1), (1,)), np.array([1990, 2000, fe.NO_YEAR], dtype=np.int64))


@pytest.fixture(scope="module")
def client() -> redis.Redis:
    c = redis.Redis.from_url(URL)
    try:
        c.ping()
    except redis.ConnectionError as exc:
        if os.environ.get("SIGNALREC_REQUIRE_REDIS") == "1":
            pytest.fail(f"SIGNALREC_REQUIRE_REDIS=1 but no Redis at {URL}: {exc}")
        pytest.skip(f"no Redis at {URL}")
    return c


@pytest.fixture
def prefix(client: redis.Redis) -> Iterator[str]:
    """A fresh ``test-<uuid>`` prefix; every key under it is deleted afterwards (an exact prefix, no glob)."""
    base = f"test-{uuid.uuid4().hex}"
    yield base
    for key in list(client.scan_iter(match=f"{base}*", count=1000)):
        client.delete(key)


@pytest.fixture
def store(client: redis.Redis, prefix: str) -> FeatureStore:
    return FeatureStore(client, CATALOGUE, namespace=f"{prefix}x")


@pytest.mark.parametrize("glob", ["?", "*"])
def test_clear_never_deletes_another_namespaces_keys(
    client: redis.Redis, prefix: str, store: FeatureStore, glob: str
) -> None:
    """Defect: ``clear`` SCANs with ``match=f"{namespace}:*"`` and the namespace check refuses only ':' and '', so a
    namespace holding a glob character matches other namespaces: clearing ``<p>?`` (or ``<p>*``) deletes every key of
    ``<p>x``.

    Input: namespace ``<p>x`` with one event recorded (5 keys), then ``FeatureStore(namespace=<p>?).clear()``.
    Expected: the glob namespace is refused, or its clear deletes nothing of ``<p>x``. Actual: it deletes all 5 keys.
    """
    assert store.record(fe.Event(1, 1, 10))
    before = sorted(client.scan_iter(match=f"{store.namespace}:*"))
    assert len(before) == 5
    try:
        other = FeatureStore(client, CATALOGUE, namespace=f"{prefix}{glob}")
    except ValueError:
        return  # refusing glob characters fixes it too
    other.clear()
    assert sorted(client.scan_iter(match=f"{store.namespace}:*")) == before
    assert store.user(1).positives == 1


@pytest.mark.parametrize(
    ("key", "field", "value"),
    [
        ("u:2:stats", "positives", "-1"),
        ("u:2:stats", "year_count", "-1"),
        ("u:2:genres", "0", "-4"),
        ("u:2:stats", "year_sum", str(2**70)),
        ("u:2:genres", "0", str(2**70)),
    ],
)
def test_counts_redis_cant_hold_are_an_error_not_a_feature(
    client: redis.Redis, store: FeatureStore, key: str, field: str, value: str
) -> None:
    """Defect: ``FeatureStoreError`` is for "Redis holds what the store didn't write", but ``_int`` accepts any signed
    decimal. A negative count (the script only increments) is returned as a feature, silently: year_count -1 makes
    the user's mean year negative, positives -1 makes ``candidate_features`` fail on ``log1p(-1)``. A value beyond
    int64 (HINCRBY can't write one) is returned as is in the stats, or raises OverflowError from NumPy for a genre.

    Input: one garbage field in user 2's stats or genres hash.
    Expected: ``FeatureStoreError``. Actual: a UserFeatures holding the value, or OverflowError.
    """
    client.hset(f"{store.namespace}:{key}", field, value)
    with pytest.raises(FeatureStoreError):
        store.user(2)


@pytest.mark.parametrize("value", ["-3", str(2**70)])
def test_popularity_redis_cant_hold_is_an_error(client: redis.Redis, store: FeatureStore, value: str) -> None:
    """Defect: as above for the popularity hash. -3 comes back as popularity -3 (``log1p(-3)`` is NaN in the ranker's
    log-popularity feature); 2^70 raises OverflowError.

    Input: popularity hash field "0" set to the value.
    Expected: ``FeatureStoreError``. Actual: array [-3, 0, 0], or OverflowError.
    """
    client.hset(f"{store.namespace}:popularity", "0", value)
    with pytest.raises(FeatureStoreError):
        store.popularity()


@pytest.mark.parametrize(("user", "index"), [(3.0, 3), (True, 1)])
def test_an_event_is_stored_where_its_user_is_read(store: FeatureStore, user: object, index: int) -> None:
    """Defect: ``_args`` checks only ``user < 0``, and the keys are ``f"<ns>:u:{user}:..."``, so an event whose user is
    3.0 (a float from JSON, say) or True is written under ``u:3.0`` / ``u:True``, keys ``store.user(3)`` /
    ``store.user(1)`` never read: recorded (returns True) yet absent from that user's features.

    Input: ``record(Event(user=3.0, movie=1, time=10))``, and ``Event(user=True, ...)``.
    Expected: refused (ValueError, as for a negative user), or visible as user 3 / user 1. Actual: neither.
    """
    try:
        store.record(fe.Event(user, 1, 10))  # type: ignore[arg-type]
    except ValueError:
        return
    assert store.user(index).positives == 1
