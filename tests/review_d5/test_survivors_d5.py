"""Review D5: tests that kill the mutants of today's code that the existing tests let survive.

Each passes on commit 886242f and fails on the mutant its docstring names (mutation log:
.scratch/review-d5-notes/mutation-log.txt; the mutants are in .scratch/review-d5-notes/mutate.py).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from signalrec import features as fe
from signalrec import serving as sv
from signalrec import twotower as tt
from signalrec.featurestore import FeatureStore, FeatureStoreError
from signalrec.prepare import Prepared
from signalrec.tower import Model

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import cold_start  # noqa: E402


def snapshot(store: FeatureStore) -> dict[bytes, Any]:
    return {k: store.client.dump(k) for k in store.client.scan_iter(match=f"{store.namespace}:*")}


# --- featurestore.py: the registry ---------------------------------------------------------------------------------


def test_a_count_of_18_digits_is_refused_and_nothing_is_written(fresh: FeatureStore) -> None:
    """Kills F4 (featurestore.py:116, ``string.len(count) <= 15`` -> ``<= 18``): with 18 digits allowed, a count of
    10^17 passes the check, and INCR + HSET write an index tostring formats as "1e+17" before Python refuses it."""
    fresh.client.set(f"{fresh.namespace}:registry:count", "1" + "0" * 17)
    before = snapshot(fresh)
    with pytest.raises(FeatureStoreError, match="BADCOUNT"):
        fresh.register(1, 60)
    assert snapshot(fresh) == before


def test_raw_id_0_is_a_valid_registry_argument(fresh: FeatureStore, api: TestClient) -> None:
    """Kills F15 (featurestore.py:178, ``0 <= value`` -> ``0 < value``): GET /recommendations/0 is a valid path
    (``DIGITS`` allows "0"), an id the model doesn't know, so it must reach the registry lookup and get the popular
    list, not a ValueError (an unhandled 500)."""
    assert fresh.registered(0, 60) is None
    assert fresh.register(0, 60) == (60, True)
    r = api.get("/recommendations/0", params={"k": 3})
    assert r.status_code == 200, r.text
    assert r.json()["source"] == "popularity"


# --- serving.py: the popular path and its cache ---------------------------------------------------------------------


def test_the_popular_path_meets_a_k_larger_than_n_candidates(fresh: FeatureStore, model: Model) -> None:
    """Kills S8 (serving.py:210, ``[:n]`` -> ``[: self.n_candidates]``): n is max(n_candidates, k), so k = 12 with
    n_candidates = 5 still gets 12 movies, the 12 most popular (ties to the smaller index)."""
    for user, movie in [(1, 9), (2, 9), (1, 4)]:
        fresh.record(fe.Event(user, movie, 0))
    got = sv.Recommender(model, fresh, n_candidates=5).recommend(None, k=12)
    assert got.movies.tolist() == [9, 4, 0, 1, 2, 3, 5, 6, 7, 8, 10, 11]
    assert got.candidates == 12


def test_the_popularity_stage_times_the_read(
    fresh: FeatureStore, model: Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kills S12 (serving.py:205, the popular path's ``seconds["popularity"] = clock() - t`` -> ``0.0``): with the list
    read every time, the read's cost is in the "popularity" stage (Server-Timing and the bench report it)."""
    real = fresh.popularity

    def slow() -> fe.IntArray:
        time.sleep(0.03)
        return real()

    monkeypatch.setattr(fresh, "popularity", slow)
    got = sv.Recommender(model, fresh, popular_ttl=0).recommend(None, k=3)
    assert got.seconds["popularity"] >= 0.03


def test_the_final_check_runs_on_the_popular_path(fresh: FeatureStore, model: Model) -> None:
    """Kills S13 (serving.py:215, the popular path's ``self._check(...)`` removed): the final check is what turns an
    exclusion bug into a 500 instead of a silent leak (DESIGN.md, day 4). Injected bug: the exclusion list for one
    genre is emptied, so its movies pass the eligibility mask; the check (which reads ``_has_genre``) must raise."""
    rec = sv.Recommender(model, fresh)
    g = model.catalogue.movie_genres[0][0]
    rec._genre_movies[g] = np.empty(0, dtype=np.int64)  # the injected exclusion bug
    with pytest.raises(AssertionError, match="excluded movies reached the final list"):
        rec.recommend(None, k=model.n_items, exclude_genres=[model.catalogue.genre_names[g]])


# --- scripts/cold_start.py: measure() ---------------------------------------------------------------------------------


SHARE = 0.3
CONFIG = tt.Config(dim=8, hidden=16, epochs=1, batch_size=64, max_history=3)


def test_measure_feeds_the_right_users_histories_popularity_and_intervals(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kills C4 (popularity from ``p``, held-out users counted), C5 (popularity without validation items), C6 (the
    replayed histories without the validation item, so not the last h before the test movie), C7 and C8 (the paired
    and unpaired intervals the wrong way round), C9 (the trained-on comparison against popularity), C10 (the groups
    swapped), C13 (the paired value reversed), C15 (targets not aligned with the users).

    The two rank functions are replaced by spies returning fixed ranks: two-tower rank 1 for the held-out group
    (NDCG 1) and 3 for the trained-on group (NDCG 0.5), popularity rank 11 (NDCG 0), so every reported value and
    interval is known exactly; the spies also record what measure() passed them."""
    drop = cold_start.held_out(prepared.user_raw_ids, SHARE)
    held, trained_on = np.flatnonzero(drop), np.flatnonzero(~drop)
    assert 0 < len(held) != len(trained_on)
    everyone = fe.user_sequences(prepared, include_val=True)
    seen: dict[str, list[Any]] = {"tower": [], "popularity": []}

    def tower_ranks(tower: Any, items: Any, histories: list[Any], targets: Any) -> Any:
        seen["tower"].append(([h.tolist() for h in histories], targets.tolist()))
        return np.full(len(targets), 1 if len(targets) == len(held) else 3, dtype=np.int64)

    def popularity_ranks(popularity: Any, histories: list[Any], targets: Any) -> Any:
        seen["popularity"].append((popularity.copy(), [h.tolist() for h in histories], targets.tolist()))
        return np.full(len(targets), 11, dtype=np.int64)

    monkeypatch.setattr(cold_start, "two_tower_ranks", tower_ranks)
    monkeypatch.setattr(cold_start, "popularity_ranks", popularity_ranks)
    result = cold_start.measure(prepared, CONFIG, SHARE)

    want_popularity = np.zeros(prepared.n_items, dtype=np.int64)
    for u in trained_on:
        np.add.at(want_popularity, everyone[u], 1)
    i = 0
    for h in cold_start.HISTORIES:
        for users in (held, trained_on):
            want_histories = [
                (s if h is None else s[len(s) - min(h, len(s)) :]).tolist() for s in (everyone[u] for u in users)
            ]
            want_targets = prepared.test_items[users].tolist()
            assert seen["tower"][i] == (want_histories, want_targets)
            popularity, histories, targets = seen["popularity"][i]
            np.testing.assert_array_equal(popularity, want_popularity)
            assert (histories, targets) == (want_histories, want_targets)
            i += 1
        if h == 1:  # the last positive before the test movie is the validation item
            assert want_histories == [[int(prepared.val_items[u])] for u in trained_on]
        row = result["by known positives"][cold_start._label(h)]
        assert row["held out"]["two-tower ndcg@10"]["value"] == pytest.approx(1.0)
        assert row["trained on"]["two-tower ndcg@10"]["value"] == pytest.approx(0.5)
        assert row["held out"]["popularity ndcg@10"]["value"] == pytest.approx(0.0)
        paired = row["held out: two-tower minus popularity, ndcg@10 (paired)"]
        assert paired["value"] == pytest.approx(1.0) and paired["ci95"] == pytest.approx([1.0, 1.0])
        unpaired = row["two-tower ndcg@10: held out minus trained on (unpaired)"]
        assert unpaired["value"] == pytest.approx(0.5) and unpaired["ci95"] == pytest.approx([0.5, 0.5])
    assert i == len(seen["tower"]) == 2 * len(cold_start.HISTORIES)


def test_measure_refuses_a_share_that_holds_nobody_out_before_training(
    prepared: Prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kills C12 (cold_start.py:135, ``drop.all() or not drop.any()`` -> ``drop.all()``): a share that holds nobody
    out is refused with its own message, before any training (the mutant trains, then fails on an empty group)."""
    assert not cold_start.held_out(prepared.user_raw_ids, 1e-9).any()

    def no_training(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("trained although nobody is held out")

    monkeypatch.setattr(tt, "train", no_training)
    with pytest.raises(ValueError, match="leaves nothing to compare"):
        cold_start.measure(prepared, CONFIG, 1e-9)


# --- api.py: registration ---------------------------------------------------------------------------------------------


def test_an_event_of_the_models_user_0_is_recorded_under_index_0(
    api: TestClient, model: Model, store: FeatureStore
) -> None:
    """Kills A7 (api.py:145, ``if user is None`` -> ``if not user``): the model's first user has dense index 0, a
    known user; their event must go to index 0, not register them as a new user."""
    raw0 = int(model.user_raw_ids[0])
    _, rated = store.serving_view(0)
    movie = next(m for m in range(model.n_items) if m not in set(rated.tolist()))
    r = api.post("/events", json={"user": raw0, "movie": int(model.item_raw_ids[movie]), "rating": 5})
    assert r.status_code == 200 and r.json()["new_user"] is False
    assert store.registered(raw0, len(model.user_raw_ids)) is None
    assert movie in store.user(0).seen.tolist()


def test_a_damaged_registry_on_a_first_event_is_a_500_naming_it(
    api: TestClient, model: Model, store: FeatureStore
) -> None:
    """Kills A9 (api.py:149, the events handler catching only ``redis.RedisError``): ``register`` refuses a damaged
    registry with a FeatureStoreError, which must become the API's named 500 ("values in Redis the store didn't
    write are a 500 naming them"), not an unhandled exception."""
    store.client.set(f"{store.namespace}:registry", "not a hash")
    new_user = int(model.user_raw_ids.max()) + 1
    r = api.post("/events", json={"user": new_user, "movie": int(model.item_raw_ids[0]), "rating": 5})
    assert r.status_code == 500
    assert r.json()["detail"].startswith("feature store: registry: WRONGTYPE")
