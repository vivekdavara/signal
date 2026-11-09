"""features.py and featurestore.py: offline features by brute force, and Redis giving exactly the same features.

The Redis tests need a server: ``SIGNALREC_REDIS_URL`` (default redis://127.0.0.1:6383/0, the project's port). Without
one they skip, unless ``SIGNALREC_REQUIRE_REDIS=1`` (CI sets it, with a Redis service container), when they fail.
Each test writes under its own namespace and deletes it afterwards.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

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
        assert store.clear() == 5  # recent, seen, genres, stats, popularity
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
