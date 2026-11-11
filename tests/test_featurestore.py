"""features.py and featurestore.py: offline features by brute force, and Redis giving exactly the same features.

The Redis tests need a server: ``SIGNALREC_REDIS_URL`` (default redis://127.0.0.1:6383/0, the project's port). Without
one they skip, unless ``SIGNALREC_REQUIRE_REDIS=1`` (CI sets it, with a Redis service container), when they fail.
Each test writes under its own namespace and deletes it afterwards.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import redis

from signalrec import features as fe
from signalrec import movielens, prepare, synthetic
from signalrec.featurestore import FeatureStore, FeatureStoreError
from signalrec.prepare import Prepared

URL = os.environ.get("SIGNALREC_REDIS_URL", "redis://127.0.0.1:6383/0")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=60, n_items=90)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


@pytest.fixture(scope="module")
def catalogue(prepared: Prepared) -> fe.Catalogue:
    return fe.Catalogue.from_prepared(prepared)


@pytest.fixture(scope="module")
def client() -> redis.Redis:
    c = redis.Redis.from_url(URL)
    try:
        c.ping()
    except redis.ConnectionError as exc:
        if os.environ.get("SIGNALREC_REQUIRE_REDIS") == "1":
            pytest.fail(f"SIGNALREC_REQUIRE_REDIS=1 but no Redis at {URL}: {exc}")
        pytest.skip(f"no Redis at {URL} (start one on 6383, or set SIGNALREC_REDIS_URL)")
    return c


@pytest.fixture
def store(client: redis.Redis, catalogue: fe.Catalogue) -> Iterator[FeatureStore]:
    s = FeatureStore(client, catalogue, namespace=f"test-{uuid.uuid4().hex}")
    yield s
    s.clear()


# --- offline features -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "year"),
    [
        ("Toy Story (1995)", 1995),
        ("City of Lost Children, The (Cité des enfants perdus, La) (1995)", 1995),
        ("Babylon 5", fe.NO_YEAR),
        ("1984 (1984) ", 1984),
        ("Blade Runner 2049", fe.NO_YEAR),  # a number in the title is not a year in brackets
        ("Some Film (1995) Extended", fe.NO_YEAR),  # only a trailing year counts
    ],
)
def test_release_year(title: str, year: int) -> None:
    assert fe.release_year(title) == year


def test_events_replay_each_users_positives_in_split_order(prepared: Prepared) -> None:
    for include_val in (False, True):
        events = fe.events(prepared, include_val)
        assert len(events) == len(prepared.train_items) + (prepared.n_users if include_val else 0)
        keys = [(e.time, e.movie, e.user) for e in events]
        assert keys == sorted(keys)
        per_user: dict[int, list[int]] = {}
        for e in events:
            per_user.setdefault(e.user, []).append(e.movie)
        for u, items in enumerate(fe.user_sequences(prepared, include_val)):
            assert per_user.get(u, []) == items.tolist()


def test_user_sequences_are_the_training_rows_then_validation(prepared: Prepared) -> None:
    for u, items in enumerate(fe.user_sequences(prepared, include_val=True)):
        train = prepared.train_items[prepared.train_users == u].tolist()  # rows sorted by (user, time, item)
        assert items.tolist() == [*train, int(prepared.val_items[u])]


def test_offline_features_by_brute_force(prepared: Prepared, catalogue: fe.Catalogue) -> None:
    users, popularity = fe.offline(prepared, include_val=True, recent=4)
    history = prepared.history(include_val=True)
    assert popularity.tolist() == np.asarray(history.sum(axis=0)).ravel().astype(int).tolist()
    for u, got in enumerate(users):
        items = fe.user_sequences(prepared, include_val=True)[u].tolist()
        assert got.recent.tolist() == items[-4:]
        assert got.seen.tolist() == sorted(history[u].indices.tolist())
        assert got.positives == len(items)
        want_genres = [0] * len(catalogue.genre_names)
        years = []
        for movie in items:
            for genre in set(prepared.genres[movie]):
                want_genres[catalogue.genre_names.index(genre)] += 1
            year = fe.release_year(prepared.titles[movie])
            if year != fe.NO_YEAR:
                years.append(year)
        assert got.genre_counts.tolist() == want_genres
        assert (got.year_sum, got.year_count) == (sum(years), len(years))


def test_user_features_of_nobody() -> None:
    catalogue = fe.Catalogue(("Drama",), ((0,),), np.array([1990], dtype=np.int64))
    nobody = fe.user_features(np.empty(0, dtype=np.int64), catalogue)
    assert nobody.positives == nobody.year_count == 0
    assert len(nobody.recent) == len(nobody.seen) == 0


def test_features_compare_by_value() -> None:
    catalogue = fe.Catalogue(("Drama", "War"), ((0,), (0, 1)), np.array([1990, fe.NO_YEAR], dtype=np.int64))
    a = fe.user_features(np.array([0, 1], dtype=np.int64), catalogue)
    assert a == fe.user_features(np.array([0, 1], dtype=np.int64), catalogue)
    assert a != fe.user_features(np.array([1, 0], dtype=np.int64), catalogue)  # same set, other order
    assert (a.year_sum, a.year_count) == (1990, 1)  # the second movie has no year
    assert a.genre_counts.tolist() == [2, 1]


# --- online == offline ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("include_val", [False, True])
def test_online_features_equal_offline_for_every_user(
    prepared: Prepared, store: FeatureStore, include_val: bool
) -> None:
    """The training-serving skew test: replay the events into Redis, read every user back, compare exactly."""
    events = fe.events(prepared, include_val)
    stats = store.replay(events, batch=37)  # a batch size that doesn't divide the event count
    assert stats.events == stats.applied == len(events)
    offline_users, offline_popularity = fe.offline(prepared, include_val)
    for u in range(prepared.n_users):
        assert store.user(u) == offline_users[u], u
    assert store.popularity().tolist() == offline_popularity.tolist()


def test_the_recent_list_is_trimmed_to_the_last_n(prepared: Prepared, client: redis.Redis) -> None:
    catalogue = fe.Catalogue.from_prepared(prepared)
    store = FeatureStore(client, catalogue, namespace=f"test-{uuid.uuid4().hex}", recent=3)
    try:
        store.replay(fe.events(prepared, include_val=True))
        offline_users, _ = fe.offline(prepared, include_val=True, recent=3)
        longest = max(range(prepared.n_users), key=lambda u: offline_users[u].positives)
        assert offline_users[longest].positives > 3  # so the trim is exercised
        for u in range(prepared.n_users):
            got = store.user(u)
            assert len(got.recent) == min(3, offline_users[u].positives)
            assert got == offline_users[u]
    finally:
        store.clear()


def test_a_redelivered_event_changes_nothing(prepared: Prepared, store: FeatureStore) -> None:
    events = fe.events(prepared, include_val=False)
    store.replay(events)
    before = [store.user(u) for u in range(prepared.n_users)], store.popularity().tolist()
    again = store.replay(events[::3])
    assert again.applied == 0
    assert not store.record(events[0])
    assert ([store.user(u) for u in range(prepared.n_users)], store.popularity().tolist()) == before


def test_one_event_updates_every_feature(store: FeatureStore, catalogue: fe.Catalogue) -> None:
    movie = next(m for m in range(catalogue.n_items) if catalogue.movie_genres[m] and catalogue.years[m] != fe.NO_YEAR)
    assert store.record(fe.Event(user=5, movie=movie, time=100))
    got = store.user(5)
    assert got.recent.tolist() == got.seen.tolist() == [movie]
    assert got.positives == 1
    assert (got.year_sum, got.year_count) == (int(catalogue.years[movie]), 1)
    assert [g for g, c in enumerate(got.genre_counts.tolist()) if c] == list(catalogue.movie_genres[movie])
    assert store.popularity()[movie] == 1
    assert store.user(6) == fe.user_features(np.empty(0, dtype=np.int64), catalogue)  # untouched


def test_namespaces_are_separate_and_clear_removes_only_its_own(
    client: redis.Redis, catalogue: fe.Catalogue, store: FeatureStore
) -> None:
    other = FeatureStore(client, catalogue, namespace=f"test-{uuid.uuid4().hex}")
    try:
        store.record(fe.Event(1, 0, 10))
        other.record(fe.Event(1, 1, 10))
        assert store.user(1).seen.tolist() == [0]
        assert other.user(1).seen.tolist() == [1]
        assert catalogue.movie_genres[0]  # so the event writes a genres hash
        assert store.clear() == 6  # recent, seen, genres, stats, popularity, rated (day 4)
        assert other.user(1).seen.tolist() == [1]
    finally:
        other.clear()


def test_a_bad_event_is_refused_before_redis(store: FeatureStore, catalogue: fe.Catalogue) -> None:
    for event in (fe.Event(0, catalogue.n_items, 1), fe.Event(0, -1, 1), fe.Event(-1, 0, 1)):
        with pytest.raises(ValueError, match=r"outside the catalogue|negative"):
            store.record(event)
        with pytest.raises(ValueError, match=r"outside the catalogue|negative"):
            store.replay([fe.Event(0, 0, 0), event])
    assert store.client.keys(f"{store.namespace}:*") == []  # the replay refused before sending its batch


@pytest.mark.parametrize(
    ("key", "setup", "message"),
    [
        ("u:2:stats", lambda c, k: c.hset(k, "positives", "seven"), "non-negative integer"),
        ("u:2:stats", lambda c, k: c.hset(k, "positives", " 7"), "non-negative integer"),
        ("u:2:stats", lambda c, k: c.hset(k, "clicks", "1"), "unexpected stats fields"),
        ("u:2:genres", lambda c, k: c.hset(k, "999", "1"), "outside the"),
        ("u:2:recent", lambda c, k: c.rpush(k, "100000"), "outside the catalogue"),
        ("u:2:seen", lambda c, k: c.set(k, "x"), "WRONGTYPE"),
    ],
)
def test_garbage_in_redis_is_an_error_not_a_feature(store: FeatureStore, key: str, setup: object, message: str) -> None:
    setup(store.client, f"{store.namespace}:{key}")  # type: ignore[operator]
    with pytest.raises(FeatureStoreError, match=message):
        store.user(2)


def test_garbage_popularity_is_an_error(store: FeatureStore) -> None:
    store.client.hset(f"{store.namespace}:popularity", "0", "1.5")
    with pytest.raises(FeatureStoreError, match="non-negative integer"):
        store.popularity()
    store.client.delete(f"{store.namespace}:popularity")
    store.client.set(f"{store.namespace}:popularity", "x")
    with pytest.raises(FeatureStoreError, match="WRONGTYPE"):
        store.popularity()


@pytest.mark.parametrize(("namespace", "recent"), [("", 5), ("a:b", 5), ("ok", 0), ("ok", 2.0)])
def test_bad_store_settings(client: redis.Redis, catalogue: fe.Catalogue, namespace: str, recent: object) -> None:
    with pytest.raises(ValueError):
        FeatureStore(client, catalogue, namespace=namespace, recent=recent)  # type: ignore[arg-type]


def test_ranker_features_from_redis_equal_offline_ones_bit_for_bit(prepared: Prepared, store: FeatureStore) -> None:
    """The skew test carried through to the ranker's input: the same feature vectors from either source."""
    from signalrec import ranking as rk

    store.replay(fe.events(prepared, include_val=True))
    offline_users, offline_popularity = fe.offline(prepared, include_val=True)
    online_popularity = store.popularity()
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(prepared.n_items, 8)).astype(np.float32)
    context = rk.Context.build(store.catalogue, vectors)
    for u in range(prepared.n_users):
        candidates = rng.choice(prepared.n_items, size=20, replace=False).astype(np.int64)
        scores = rng.normal(size=20)
        a = rk.candidate_features(offline_users[u], candidates, scores, offline_popularity, context)
        b = rk.candidate_features(store.user(u), candidates, scores, online_popularity, context)
        assert np.array_equal(a, b), u


# --- the rated set and the request path's reads (day 4) ---------------------------------------------------------


def test_a_positive_is_also_rated(store: FeatureStore) -> None:
    assert store.record(fe.Event(4, 7, 100))
    features, rated = store.serving_view(4)
    assert rated.tolist() == [7]
    assert features.seen.tolist() == [7]


def test_a_rating_that_is_not_a_positive_changes_no_feature(store: FeatureStore, catalogue: fe.Catalogue) -> None:
    store.record(fe.Event(2, 3, 100))
    before = store.user(2)
    popularity = store.popularity()
    assert store.record_rating(fe.Event(2, 11, 101))
    assert not store.record_rating(fe.Event(2, 11, 102))  # already rated: a no-op
    assert store.user(2) == before
    np.testing.assert_array_equal(store.popularity(), popularity)
    assert store.serving_view(2)[1].tolist() == [3, 11]
    # A later positive on a movie the user had rated lower still counts as a positive, once.
    assert store.record(fe.Event(2, 11, 103))
    assert store.user(2).seen.tolist() == [3, 11]
    assert store.serving_view(2)[1].tolist() == [3, 11]
    assert not store.record(fe.Event(2, 11, 104))


def test_bad_ratings_are_refused(store: FeatureStore, catalogue: fe.Catalogue) -> None:
    for event in (fe.Event(1, catalogue.n_items, 0), fe.Event(-1, 0, 0), fe.Event(True, 0, 0)):
        with pytest.raises(ValueError):
            store.record_rating(event)
    with pytest.raises(ValueError):
        store.replay_ratings([fe.Event(1, 0, 0), fe.Event(1, -5, 0)])
    assert store.client.keys(f"{store.namespace}:*") == []  # refused before its batch was sent


def test_the_serving_view_is_one_transaction(store: FeatureStore, monkeypatch: pytest.MonkeyPatch) -> None:
    store.record(fe.Event(1, 2, 10))
    store.record_rating(fe.Event(1, 5, 11))
    seen: list[bool] = []
    real = store.client.pipeline

    def spy(transaction: bool = True, shard_hint: object = None) -> object:
        seen.append(transaction)
        return real(transaction=transaction, shard_hint=shard_hint)

    monkeypatch.setattr(store.client, "pipeline", spy)
    features, rated = store.serving_view(1)
    assert seen == [True]
    assert features == store.user(1)
    assert features.seen.tolist() == [2]
    assert rated.tolist() == [2, 5]


def test_a_serving_view_of_a_user_with_nothing(store: FeatureStore, catalogue: fe.Catalogue) -> None:
    features, rated = store.serving_view(42)
    assert features == fe.user_features(np.empty(0, dtype=np.int64), catalogue)
    assert rated.tolist() == []


def test_a_foreign_value_in_the_rated_set_is_an_error(store: FeatureStore, catalogue: fe.Catalogue) -> None:
    store.client.sadd(f"{store.namespace}:u:1:rated", catalogue.n_items)
    with pytest.raises(FeatureStoreError, match="outside the catalogue"):
        store.serving_view(1)
    store.client.delete(f"{store.namespace}:u:1:rated")
    store.client.set(f"{store.namespace}:u:1:rated", "x")
    with pytest.raises(FeatureStoreError):
        store.serving_view(1)


def test_popularity_of_the_candidates_equals_the_whole_hash(store: FeatureStore, prepared: Prepared) -> None:
    store.replay(fe.events(prepared, include_val=True))
    everything = store.popularity()
    rng = np.random.default_rng(0)
    for _ in range(20):
        movies = rng.choice(prepared.n_items, size=int(rng.integers(0, 40)), replace=False).astype(np.int64)
        np.testing.assert_array_equal(store.popularity_of(movies), everything[movies])
    never = np.flatnonzero(everything == 0)
    if len(never):
        assert store.popularity_of(never[:3]).tolist() == [0] * len(never[:3])
    with pytest.raises(ValueError, match="outside the catalogue"):
        store.popularity_of(np.array([0, prepared.n_items], dtype=np.int64))
    store.client.hset(f"{store.namespace}:popularity", "0", "-3")
    with pytest.raises(FeatureStoreError):
        store.popularity_of(np.array([0], dtype=np.int64))


def test_rated_events_stop_before_each_users_test_movie(tmp_path: Path) -> None:
    raw = synthetic.write(tmp_path / "raw", n_users=60, n_items=90)
    dataset = movielens.load("synthetic", raw)
    p = prepare.prepare(dataset, prepare.Params())
    events = fe.rated_events(p, dataset.ratings)
    keys = [(e.time, e.movie, e.user) for e in events]
    assert keys == sorted(keys)
    by_user: dict[int, set[int]] = {}
    for e in events:
        assert e.time < p.test_times[e.user]
        by_user.setdefault(e.user, set()).add(e.movie)
    # Brute force from the raw rows.
    r = dataset.ratings
    want: dict[int, set[int]] = {}
    user_of = {int(raw_id): u for u, raw_id in enumerate(p.user_raw_ids.tolist())}
    item_of = {int(raw_id): i for i, raw_id in enumerate(p.item_raw_ids.tolist())}
    lower = 0
    for raw_u, raw_i, rating, t in zip(
        r.users.tolist(), r.items.tolist(), r.ratings.tolist(), r.times.tolist(), strict=True
    ):
        if raw_u in user_of and raw_i in item_of and t < p.test_times[user_of[raw_u]]:
            want.setdefault(user_of[raw_u], set()).add(item_of[raw_i])
            lower += rating < 4
    assert by_user == want
    assert lower > 0  # the synthetic data has ratings below 4, so the set is more than the positives
    for u in range(p.n_users):
        assert int(p.test_items[u]) not in by_user.get(u, set())


# --- the registry of new users (day 5) ---------------------------------------------------------------------------


def test_new_users_are_numbered_from_the_base_once_each(store: FeatureStore) -> None:
    assert store.registered(9001, 60) is None
    assert store.register(9001, 60) == (60, True)
    assert store.register(42, 60) == (61, True)
    assert store.register(9001, 60) == (60, False)  # idempotent: a retried first event
    assert store.registered(9001, 60) == 60
    assert store.registered(42, 60) == 61
    assert store.registered(7, 60) is None


def test_looking_up_a_user_writes_nothing(store: FeatureStore, client: redis.Redis) -> None:
    assert store.registered(5, 60) is None
    assert list(client.scan_iter(match=f"{store.namespace}:*")) == []


def test_a_registry_for_another_model_is_refused(store: FeatureStore) -> None:
    store.register(1, 60)
    with pytest.raises(FeatureStoreError, match="numbers new users from 60, not 61"):
        store.register(2, 61)
    with pytest.raises(FeatureStoreError, match="from 60, not 61"):
        store.registered(1, 61)
    assert store.registered(2, 60) is None  # the refused call allocated nothing


def test_a_deleted_entry_never_frees_its_index(store: FeatureStore, client: redis.Redis) -> None:
    store.register(1, 10)
    store.register(2, 10)
    client.hdel(f"{store.namespace}:registry", 1)
    assert store.register(3, 10) == (12, True)  # not 11, which user 2 holds


def test_concurrent_first_events_of_one_user_get_one_index(store: FeatureStore) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(lambda _: store.register(777, 60), range(64)))
    assert {dense for dense, _ in results} == {60}
    assert sum(created for _, created in results) == 1


@pytest.mark.parametrize(
    ("suffix", "setup", "message"),
    [
        ("registry", lambda c, k: c.set(k, "x"), "WRONGTYPE"),
        ("registry:count", lambda c, k: c.hset(k, "a", 1), "WRONGTYPE"),
        ("registry:base", lambda c, k: c.rpush(k, 1), "WRONGTYPE"),
        ("registry:count", lambda c, k: c.set(k, "-3"), "BADCOUNT"),
        ("registry:count", lambda c, k: c.set(k, "1.5"), "BADCOUNT"),
    ],
)
def test_a_damaged_registry_is_an_error_and_nothing_is_written(
    store: FeatureStore, client: redis.Redis, suffix: str, setup: object, message: str
) -> None:
    key = f"{store.namespace}:{suffix}"
    setup(client, key)  # type: ignore[operator]
    before = {k: client.dump(k) for k in client.scan_iter(match=f"{store.namespace}:*")}
    with pytest.raises(FeatureStoreError, match=message):
        store.register(1, 60)
    after = {k: client.dump(k) for k in client.scan_iter(match=f"{store.namespace}:*")}
    assert after == before


@pytest.mark.parametrize("value", ["abc", "-1", "59", " 61"])
def test_a_foreign_index_in_the_registry_is_an_error(store: FeatureStore, client: redis.Redis, value: str) -> None:
    store.register(1, 60)
    client.hset(f"{store.namespace}:registry", 2, value)
    with pytest.raises(FeatureStoreError):
        store.registered(2, 60)
    with pytest.raises(FeatureStoreError):
        store.register(2, 60)


def test_an_entry_without_a_base_is_an_error(store: FeatureStore, client: redis.Redis) -> None:
    client.hset(f"{store.namespace}:registry", 2, 70)
    with pytest.raises(FeatureStoreError, match="no base"):
        store.registered(2, 60)


@pytest.mark.parametrize(("raw", "base"), [(-1, 60), (1, -1), (True, 60), (1.0, 60), (2**63, 60), (1, "60")])
def test_bad_registry_arguments_are_refused_before_redis(store: FeatureStore, raw: object, base: object) -> None:
    with pytest.raises(ValueError):
        store.register(raw, base)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.registered(raw, base)  # type: ignore[arg-type]


def test_clear_removes_the_registry(store: FeatureStore) -> None:
    store.register(1, 60)
    assert store.clear() == 3
    assert store.registered(1, 60) is None


def test_a_registry_with_more_entries_than_its_count_is_refused(store: FeatureStore, client: redis.Redis) -> None:
    store.register(1, 60)
    client.hset(f"{store.namespace}:registry", 2, 1)  # a second entry the counter never issued
    before = {k: client.dump(k) for k in client.scan_iter(match=f"{store.namespace}:*")}
    with pytest.raises(FeatureStoreError, match="more than its count"):
        store.register(3, 60)
    assert {k: client.dump(k) for k in client.scan_iter(match=f"{store.namespace}:*")} == before


def test_indices_far_past_the_model_are_exact(store: FeatureStore, client: redis.Redis) -> None:
    """Day 5's review: the index was computed in Lua doubles and written with 14 significant digits."""
    big = 2**62
    assert store.register(1, big) == (big, True)
    client.set(f"{store.namespace}:registry:count", "999999999999998")
    assert store.register(2, big) == (big + 999_999_999_999_998, True)
    assert store.registered(2, big) == big + 999_999_999_999_998
    assert store.registered(1, big) == big
    with pytest.raises(ValueError):
        store.register(3, big + 1)


@pytest.mark.parametrize(
    "setup",
    [
        {"registry": {"7": "1", "8": "2"}, "registry:count": "1", "registry:base": "60"},  # more entries than counted
        {"registry": {"7": "1"}, "registry:count": "0001", "registry:base": "60"},  # a count with leading zeros
        {"registry": {"7": "1"}, "registry:base": "60"},  # no counter
    ],
)
def test_both_paths_refuse_a_damaged_registry_for_anyone(
    store: FeatureStore, client: redis.Redis, setup: dict[str, object]
) -> None:
    """Round 2 of the day-5 review: registered() must refuse what register() refuses, also for an id not in it."""
    for suffix, value in setup.items():
        key = f"{store.namespace}:{suffix}"
        if isinstance(value, dict):
            client.hset(key, mapping=value)
        else:
            client.set(key, str(value))
    for raw in (7, 99):
        with pytest.raises(FeatureStoreError):
            store.register(raw, 60)
        with pytest.raises(FeatureStoreError):
            store.registered(raw, 60)


def test_a_full_registry_still_resolves_its_users(store: FeatureStore, client: redis.Redis) -> None:
    client.set(f"{store.namespace}:registry:count", str(10**15 - 1))
    client.set(f"{store.namespace}:registry:base", "60")
    assert store.register(5, 60) == (60 + 10**15 - 1, True)  # the last ordinal
    assert store.register(5, 60) == (60 + 10**15 - 1, False)
    assert store.registered(5, 60) == 60 + 10**15 - 1
    with pytest.raises(FeatureStoreError, match="full"):
        store.register(6, 60)
    assert store.registered(6, 60) is None
