"""Review D4: tests that kill mutants the existing suite let survive. Each passes on the original code; the docstring
names the mutant (the source line and the change) it kills."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import redis
from fastapi.testclient import TestClient

from signalrec import abtest, export, movielens
from signalrec import features as fe
from signalrec import serving as sv
from signalrec.api import MAX_K, create_app
from signalrec.featurestore import FeatureStore
from signalrec.prepare import Prepared
from signalrec.tower import Model

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import bench_serving as bs


@pytest.fixture
def store(client: redis.Redis, model: Model, prepared: Prepared, dataset: movielens.Dataset) -> Iterator[FeatureStore]:
    s = FeatureStore(client, model.catalogue, namespace=f"test-{uuid.uuid4().hex}")
    export.load_store(s, prepared, dataset.ratings)
    yield s
    s.clear()


def _api(model: Model, store: FeatureStore) -> TestClient:
    return TestClient(create_app(sv.Recommender(model, store, n_candidates=40)))


# --- api.py ---------------------------------------------------------------------------------------------------------


def test_a_rating_at_min_rating_is_a_positive_and_one_below_is_not(model: Model, store: FeatureStore) -> None:
    """Kills api.py:113 ``float(event.rating) >= min_rating`` -> ``>`` (prepare's positives are rating >= min_rating,
    so a 4-star rating at min_rating 4 must update the features), and api.py:49's fallback 4.0 -> 3.0 (the fixture
    model's info has no min_rating: the fallback must be prepare's default, so 3.5 is not a positive)."""
    api = _api(model, store)
    user = int(model.user_raw_ids[9])
    seen = set(store.user(9).seen.tolist())
    movies = [m for m in range(model.n_items) if m not in seen][:2]
    at = api.post("/events", json={"user": user, "movie": int(model.item_raw_ids[movies[0]]), "rating": 4})
    assert at.json() == {"applied": True, "positive": True}
    below = api.post("/events", json={"user": user, "movie": int(model.item_raw_ids[movies[1]]), "rating": 3.5})
    assert below.json()["positive"] is False


def test_k_equal_to_max_k_is_accepted(model: Model, store: FeatureStore) -> None:
    """Kills api.py:87 ``Query(ge=1, le=MAX_K)`` -> ``lt=MAX_K`` (k = 100 is in range and must be served)."""
    r = _api(model, store).get(f"/recommendations/{int(model.user_raw_ids[0])}", params={"k": MAX_K})
    assert r.status_code == 200 and 0 < len(r.json()["items"]) <= MAX_K


def test_a_redis_timeout_is_a_503(model: Model, store: FeatureStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kills api.py:65 ``redis.ConnectionError | redis.TimeoutError`` -> ``redis.ConnectionError`` (a Redis that
    stops answering is unavailable, a 503, not a 500)."""

    def timeout(*args: object, **kwargs: object) -> None:
        raise redis.TimeoutError("Timeout reading from socket")

    api = _api(model, store)
    monkeypatch.setattr(store, "serving_view", timeout)
    monkeypatch.setattr(store, "record", timeout)
    monkeypatch.setattr(store.client, "ping", timeout)
    user, movie = int(model.user_raw_ids[0]), int(model.item_raw_ids[0])
    assert api.get(f"/recommendations/{user}").status_code == 503
    assert api.post("/events", json={"user": user, "movie": movie, "rating": 5}).status_code == 503
    assert api.get("/health").status_code == 503


# --- serving.py -----------------------------------------------------------------------------------------------------


def test_a_liked_movie_missing_from_the_rated_set_is_still_never_served(fresh: FeatureStore, model: Model) -> None:
    """Kills serving.py:121 ``else [rated, features.seen]`` -> ``else [rated]``. ``rated`` holds every positive only
    for events written by the day-4 script; a store written before day 4 (or a positive whose rated SADD failed) has
    liked movies in ``seen`` alone, and the default filter must still remove them."""
    rec = sv.Recommender(model, fresh, n_candidates=30)
    top = rec.recommend(4, k=5).movies.tolist()
    for movie in top[:3]:  # a positive as a day-3 store held it: seen, without the rated set
        fresh.client.sadd(f"{fresh.namespace}:u:4:seen", movie)
    after = rec.recommend(4, k=5).movies.tolist()
    assert not set(top[:3]) & set(after)


# --- scripts/bench_serving.py ---------------------------------------------------------------------------------------


async def _fast(i: int) -> float:
    await asyncio.sleep(0.001)
    return 0.0005


def test_the_achieved_rate_does_not_count_the_lead_in_time() -> None:
    """Kills bench_serving.py:220 ``len(timed) / (last_done - first_due)`` -> ``/ (last_done - start)`` (the
    lead-in's time divided into the timed requests: 200/s with a lead-in as long as the run reads as 100/s)."""
    row = asyncio.run(bs.open_loop(_fast, rate=200, seconds=0.5, lead_in=0.5))
    assert 170 < row["achieved (req/s)"] < 230


def test_a_client_that_falls_behind_shows_its_lateness() -> None:
    """Kills bench_serving.py:176 ``sent = time.perf_counter()`` -> ``sent = due`` (lateness always zero). A client
    whose event loop is blocked 10 ms per request can send at most 100/s: asked for 400/s it falls behind, and the
    lateness must say so."""

    async def blocking(i: int) -> float:
        time.sleep(0.01)  # blocks the client's loop, as a CPU-bound client would
        return 0.0

    row = asyncio.run(bs.open_loop(blocking, rate=400, seconds=0.2, lead_in=0.0))
    assert row["client lateness, sent - scheduled (µs)"]["p99"] > 50_000


def test_no_request_is_sent_before_its_scheduled_time_at_a_high_rate() -> None:
    """Kills bench_serving.py:188 ``if delay > 0:`` -> ``if delay > 0.002:`` (a request due within 2 ms is sent at
    once, early: at 1,000/s every request is early, and latency from the schedule is understated by up to 2 ms).
    The existing trace test runs at 100/s, where every delay is 10 ms."""
    trace: list[dict[str, float]] = []
    asyncio.run(bs.open_loop(_fast, rate=1000, seconds=0.1, lead_in=0.0, trace=trace))
    assert len(trace) == 100
    early = [t for t in trace if t["sent"] < t["due"]]
    assert not early, f"{len(early)} requests sent before their scheduled time"


# --- serving.py, tower.py, abtest.py: consistency checks and statistics ------------------------------------------


def test_a_store_whose_genre_names_differ_is_refused(client: redis.Redis, model: Model) -> None:
    """Kills serving.py:68 dropping ``or store.catalogue.genre_names != model.catalogue.genre_names``: the store
    counts genres by index, so a store built with the names in another order feeds the ranker's genre affinity the
    wrong genres; the existing test only changes the movie count."""
    names = model.catalogue.genre_names
    swapped = (names[1], names[0], *names[2:])
    other = fe.Catalogue(swapped, model.catalogue.movie_genres, model.catalogue.years)
    with pytest.raises(ValueError, match="catalogue"):
        sv.Recommender(model, FeatureStore(client, other, namespace="unused"))


def test_duplicate_raw_ids_are_refused(model: Model) -> None:
    """Kills tower.py:160 ``(np.diff(ids) <= 0).any()`` -> ``< 0`` (two dense movies or users with one MovieLens id:
    movie_index/user_index answer one of them, and the other can never be addressed)."""
    from dataclasses import replace

    from signalrec.tower import ModelError

    items = model.item_raw_ids.copy()
    items[1] = items[0]
    users = model.user_raw_ids.copy()
    users[1] = users[0]
    for bad in (replace(model, item_raw_ids=items), replace(model, user_raw_ids=users)):
        with pytest.raises(ModelError, match="strictly ascending"):
            bad.check()


def test_a_ranker_that_does_not_end_in_one_score_is_refused(model: Model) -> None:
    """Kills tower.py:170 ``if not r.layers or width != 1`` -> ``if not r.layers`` (a ranker with two outputs would
    be served by its first column, silently)."""
    from dataclasses import replace

    from signalrec import ranking as rk
    from signalrec.tower import ModelError

    rng = np.random.default_rng(1)
    f = len(rk.FEATURES)
    two = rk.RankerWeights(np.zeros(f), np.ones(f), ((rng.normal(size=(2, f)), rng.normal(size=2)),))
    with pytest.raises(ModelError, match="one score"):
        replace(model, ranker=two).check()


def _gap(ids: np.ndarray) -> int:
    """A raw id between two of ``ids`` (or just below the first), so searchsorted lands on a real index."""
    gaps = (int(a) + 1 for a, b in zip(ids[:-1].tolist(), ids[1:].tolist(), strict=True) if b - a > 1)
    return next(gaps, int(ids[0]) - 1)


def test_ids_between_known_ids_are_unknown(model: Model, store: FeatureStore) -> None:
    """Kills tower.py:178 and tower.py:184 dropping ``or self.item_raw_ids[at] != raw`` (and the user one): an unknown
    MovieLens id that sorts between two known ones resolves to the next known movie or user, so a POST /events for a
    movie not in the catalogue records another movie, and GET /recommendations for an unknown user serves another
    user's list. The existing tests use max + 1 only, which the ``at == len`` branch catches."""
    movie, user = _gap(model.item_raw_ids), _gap(model.user_raw_ids)
    with pytest.raises(KeyError):
        model.movie_index(movie)
    with pytest.raises(KeyError):
        model.user_index(user)
    api = _api(model, store)
    assert api.get(f"/recommendations/{user}").status_code == 404
    known_user = int(model.user_raw_ids[0])
    assert api.post("/events", json={"user": known_user, "movie": movie, "rating": 5}).status_code == 404


def test_non_finite_item_vectors_are_refused(model: Model) -> None:
    """Kills tower.py:145 dropping ``or not np.isfinite(self.items).all()`` (a NaN item vector would be served: FAISS
    ranks NaN inner products arbitrarily)."""
    from dataclasses import replace

    from signalrec.tower import ModelError

    items = model.items.copy()
    items[3, 0] = np.nan
    with pytest.raises(ModelError, match="finite"):
        replace(model, items=items).check()


def test_a_zero_share_puts_nobody_in_the_treatment() -> None:
    """Kills abtest.py:49 ``bucket(...) < round(share * BUCKETS)`` -> ``<=`` (bucket 0's users, 1 in 10,000, get the
    treatment of an experiment that is switched off, and every share gets one bucket too many)."""
    off = abtest.Experiment("review-d4-off", 0.0)
    in_bucket_0 = [u for u in range(1, 200_000) if abtest.bucket(off.name, u) == 0]
    assert in_bucket_0, "no user in bucket 0 among 200,000"
    assert all(abtest.assign(off, u) == abtest.CONTROL for u in in_bucket_0)
    tenth = abtest.Experiment("review-d4-off", 0.1)
    in_bucket_1000 = [u for u in range(1, 200_000) if abtest.bucket(tenth.name, u) == 1000]
    assert in_bucket_1000 and all(abtest.assign(tenth, u) == abtest.CONTROL for u in in_bucket_1000)


def test_the_interval_uses_each_arms_own_size() -> None:
    """Kills abtest.py:123 with the arms' sizes swapped in the standard error (equal arms hide it): 30 of 100 against
    300 of 1000 by hand."""
    import math

    from scipy import stats

    c = abtest.compare(30, 100, 450, 1000)
    se = math.sqrt(0.3 * 0.7 / 100 + 0.45 * 0.55 / 1000)
    z = float(stats.norm.ppf(0.975))
    assert c.ci95 == pytest.approx((0.15 - z * se, 0.15 + z * se), abs=1e-12)


# --- featurestore.py ------------------------------------------------------------------------------------------------


def test_replay_ratings_counts_the_last_partial_batch(fresh: FeatureStore) -> None:
    """Kills featurestore.py:149 ``applied += sum(...)`` dropped for the last, partial batch (``load-store`` prints
    "ratings into the rated sets: N of M new" and N would leave out up to batch - 1 new ratings; the existing test
    only checks 0 < N <= M)."""
    events = [fe.Event(u, m, 0) for u in range(3) for m in range(5)]
    stats = fresh.replay_ratings(events, batch=4)  # 15 events: three full batches and one of 3
    assert (stats.events, stats.applied) == (15, 15)
    again = fresh.replay_ratings(events[:7], batch=100)
    assert (again.events, again.applied) == (7, 0)
