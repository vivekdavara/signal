"""Review D3 round 2: the event-type fix closed the write side only; ``FeatureStore.user`` still takes any id.

Needs Redis (``SIGNALREC_REDIS_URL``, default redis://127.0.0.1:6383/0); skips without one unless
``SIGNALREC_REQUIRE_REDIS=1``. Every key written is under a fresh ``test-<uuid>`` namespace and deleted afterwards.
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
def store(client: redis.Redis) -> Iterator[FeatureStore]:
    namespace = f"test-{uuid.uuid4().hex}"
    yield FeatureStore(client, CATALOGUE, namespace=namespace)
    for key in list(client.scan_iter(match=f"{namespace}:*", count=1000)):
        client.delete(key)


@pytest.mark.parametrize("user", [3.0, np.float64(3.0)], ids=["float", "numpy-float"])
def test_reading_a_user_by_a_non_integer_id_is_refused_like_writing_one(store: FeatureStore, user: object) -> None:
    """Defect (left open by the fix): ``_args`` now refuses an event whose user is 3.0 ("would be written under u:3.0,
    keys user(3) never reads"), but ``user()`` formats whatever it gets into the key, so ``user(3.0)`` reads ``u:3.0``
    and returns an empty user, silently, for a user with events: the same key split, from the read side.

    Input: one event for user 3 (movie 1), then ``store.user(3.0)``.
    Expected: a ValueError (as ``record`` raises for user 3.0), or user 3's features. Actual: no error; positives 0,
    empty recent and seen lists.
    """
    assert store.record(fe.Event(3, 1, 0))
    try:
        got = store.user(user)  # type: ignore[arg-type]
    except ValueError:
        return
    assert got.positives == 1, f"user({user!r}) read keys user 3's event was never written to: {got}"


def test_the_largest_int64_is_a_count_and_one_more_is_not(client: redis.Redis) -> None:
    """Round 2's surviving mutant: ``> INT64_MAX`` as ``>=`` passed, as nothing read back exactly 2^63 - 1."""
    store = FeatureStore(client, CATALOGUE, namespace=f"test-{uuid.uuid4().hex}")
    try:
        store.client.hset(f"{store.namespace}:u:2:stats", "year_sum", str(2**63 - 1))
        assert store.user(2).year_sum == 2**63 - 1
        store.client.hset(f"{store.namespace}:u:2:stats", "year_sum", str(2**63))
        with pytest.raises(FeatureStoreError, match="beyond"):
            store.user(2)
    finally:
        store.clear()


@pytest.mark.parametrize("user", [True, False])
def test_a_bool_user_is_refused_on_write_and_read(client: redis.Redis, user: bool) -> None:
    """Round 2's surviving mutant: dropping the bool refusal passed, because ``int(True)`` writes user 1's keys and
    round 1's test accepted either outcome. True is not a user id: both directions refuse it."""
    store = FeatureStore(client, CATALOGUE, namespace=f"test-{uuid.uuid4().hex}")
    try:
        with pytest.raises(ValueError, match="integer"):
            store.record(fe.Event(user, 0, 1))
        with pytest.raises(ValueError, match="integer"):
            store.user(user)
        assert store.client.keys(f"{store.namespace}:*") == []
    finally:
        store.clear()
