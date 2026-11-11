"""Review D5: defects in today's code (registry, popular path, API registration, bench_new_users.py).

Each test fails on commit 886242f because of the defect its docstring names.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from signalrec import serving as sv
from signalrec.featurestore import FeatureStore, FeatureStoreError
from signalrec.tower import Model

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import bench_new_users  # noqa: E402


def snapshot(store: FeatureStore) -> dict[bytes, Any]:
    return {k: store.client.dump(k) for k in store.client.scan_iter(match=f"{store.namespace}:*")}


# --- 1. featurestore.py:123: the index is computed in Lua doubles and formatted by tostring (%.14g) -----------------


def test_a_base_of_10_to_the_14_is_numbered_from_the_base(fresh: FeatureStore) -> None:
    """Defect: featurestore.py:123, ``dense = tostring(tonumber(ARGV[2]) + n - 1)``. Lua 5.1's tostring formats numbers
    with "%.14g", so any index >= 10^14 is written as "1e+14" (and above 2^53 the double sum is also inexact).
    ``_check_registry_args`` accepts any base in [0, 2^63).

    Input: register(1, base=10**14) on an empty registry.
    Expected: (10**14, True), as for any base.
    Actual: the script writes base, count and the entry "1e+14", then ``_int`` raises FeatureStoreError; the user is
    left with an index nothing can read (every later register/registered of raw id 1 raises).
    """
    assert fresh.register(1, 10**14) == (10**14, True)


def test_a_count_the_script_accepts_never_yields_a_garbled_index(fresh: FeatureStore) -> None:
    """Defect: featurestore.py:116-124. The count check allows up to 15 digits ("a count below 10^15"), but tostring
    keeps 14 significant digits, so a count of 10^14 (which passes the check) gives the index 60 + 10^14 written as
    "1.0000000000006e+14": the check is one digit too loose, and the script writes before Python sees the garbage.

    Input: ``<ns>:registry:count`` = "100000000000000", then register(1, 60).
    Expected: either the right index (10**14 + 60, True), or a FeatureStoreError with nothing written (the day-4
    convention: a script is never half-applied).
    Actual: FeatureStoreError *after* INCR, SET base and HSET of a non-integer index: the registry is damaged.
    """
    fresh.client.set(f"{fresh.namespace}:registry:count", "100000000000000")
    before = snapshot(fresh)
    try:
        got = fresh.register(1, 60)
    except FeatureStoreError:
        assert snapshot(fresh) == before, "the refused registration wrote to the registry"
    else:
        assert got == (10**14 + 60, True)


# --- 2. featurestore.py:115-122: a missing counter or base is taken for an empty registry --------------------------


@pytest.mark.parametrize("lost", ["count", "base"])
def test_a_lost_counter_or_base_never_hands_out_an_index_already_held(fresh: FeatureStore, lost: str) -> None:
    """Defect: featurestore.py:115-122. The script treats a missing ``registry:count`` as 0 and a missing
    ``registry:base`` as "first registration" even when the registry hash already holds entries, so it re-issues an
    index another raw id holds: two MovieLens users then share one dense index (one store history). ``registered``
    already calls an entry without a base an error; ``register`` does not.

    Input: register(1, 60) -> 60, register(2, 60) -> 61; then DEL ``registry:count`` and register(3, 60) (case
    "count"), or DEL ``registry:base`` and register(3, 59), another model's base (case "base").
    Expected: a FeatureStoreError (a damaged registry), or an index neither user 1 nor user 2 holds.
    Actual: user 3 gets 60 (count lost: user 1's index) or 61 (base lost: user 2's index, and base silently becomes 59).
    """
    ns = fresh.namespace
    assert fresh.register(1, 60) == (60, True)
    assert fresh.register(2, 60) == (61, True)
    fresh.client.delete(f"{ns}:registry:{lost}")
    base = 60 if lost == "count" else 59
    try:
        dense, _ = fresh.register(3, base)
    except FeatureStoreError:
        return
    others = {int(fresh.client.hget(f"{ns}:registry", r) or b"-1") for r in (1, 2)}
    assert dense not in others, f"user 3 got index {dense}, which user 1 or 2 holds ({sorted(others)})"


# --- 3. api.py:45 + 146: POST /events has no upper bound on the user id ------------------------------------------


def test_an_event_for_a_user_id_beyond_int64_is_not_a_500(api: TestClient, model: Model) -> None:
    """Defect: api.py:45 (``user: StrictInt = Field(ge=1)``, no upper bound) with api.py:146: an unknown user now goes
    to ``store.register``, whose argument check raises ValueError for ids >= 2^63, and ValueError is not among the
    caught exceptions (api.py:149), so it is an unhandled 500. On day 4 the same request was a 404.

    Input: POST /events {"user": 2**63, "movie": <a catalogue movie>, "rating": 5}.
    Expected: a 4xx (422: out of range, or 404 as before).
    Actual: 500 Internal Server Error (ValueError: raw id must be an integer in [0, 2^63)).
    """
    movie = int(model.item_raw_ids[0])
    r = api.post("/events", json={"user": 2**63, "movie": movie, "rating": 5})
    assert 400 <= r.status_code < 500, (r.status_code, r.text)


def test_a_user_registered_by_an_event_can_get_recommendations(api: TestClient, model: Model) -> None:
    """Defect: api.py:45 vs api.py:39/109. POST /events accepts and registers any id up to 2^63 - 1, but the GET path
    only takes up to 18 digits (``DIGITS``), so a 19-digit user is registered, has their events recorded, and can never
    be served: the two endpoints disagree on what a user id is.

    Input: POST /events {"user": 10**18, "movie": m, "rating": 5}, then GET /recommendations/1000000000000000000.
    Expected: either the POST is refused (422), or the GET serves the user (200).
    Actual: POST 200 {"new_user": true}, GET 422 ("String should match pattern").
    """
    movie = int(model.item_raw_ids[0])
    post = api.post("/events", json={"user": 10**18, "movie": movie, "rating": 5})
    if post.status_code == 200:
        get = api.get(f"/recommendations/{10**18}")
        assert get.status_code == 200, f"registered by POST ({post.json()}) but GET is {get.status_code}"


# --- 4. serving.py:159-161: the exclusion set is built outside every timed stage ------------------------------


class _SlowUnique:
    """numpy, except that ``unique`` takes 50 ms longer: stands in for an exclusion set that is expensive to build."""

    DELAY = 0.05

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)

    @staticmethod
    def unique(*args: Any, **kwargs: Any) -> Any:
        time.sleep(_SlowUnique.DELAY)
        return np.unique(*args, **kwargs)


@pytest.mark.parametrize("who", ["known", "nobody"])
def test_the_stage_times_include_building_the_exclusions(
    loaded: FeatureStore, model: Model, monkeypatch: pytest.MonkeyPatch, who: str
) -> None:
    """Defect: serving.py:159-161. Today's change moved ``removed``/``excluded`` (concatenate + np.unique of the rated
    set, the liked set and the excluded genres) out of the timed "retrieve" stage to before the popular-path branch,
    where no stage times it. ``Recommendation.seconds``, the Server-Timing header and bench_new_users.py's "total"
    (the sum of the stages) all leave it out, on both paths. Measured on the ml-1m model: a new user with 2,000 rated
    movies, 984 µs wall vs 882 µs total; ``user=None``, 12.5 µs wall vs 8.3 µs total (the bench reports 7 µs).

    Input: serving's numpy with ``unique`` slowed by 50 ms; recommend(0) (a loaded user) and recommend(None).
    Expected: ``rec.total`` >= 50 ms (some stage includes building the exclusions, as "retrieve" did on day 4).
    Actual: ``rec.total`` is a few ms: the 50 ms are in no stage.
    """
    rec = sv.Recommender(model, loaded)
    monkeypatch.setattr(sv, "np", _SlowUnique())
    got = rec.recommend(0 if who == "known" else None, k=10)
    assert got.source == ("two-tower" if who == "known" else "popularity")
    assert got.total >= _SlowUnique.DELAY, f"stages sum to {got.total * 1e3:.2f} ms; 50 ms were spent in none"


# --- 5. bench_new_users.py:89-97: the "kept 1 s" runs never see the list expire -----------------------------------


def test_the_kept_one_second_run_measures_an_expiring_list(
    loaded: FeatureStore, model: Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect: bench_new_users.py:89-97 (and ``bench``). The "popular list kept 1 s" runs are 2,000 requests after a
    200-request lead-in, about 0.3 s in all on ml-1m (139 µs each), so the list read in the lead-in never expires
    during the timed requests: those rows measure popular_ttl = infinity, not 1 s, and their tails leave out the
    refresh (one 2.1 ms read per second of traffic). results/ml-1m/new-users-bench.json shows it: the "popularity"
    stage's max is 0.5 µs ("new users ... kept 1 s") and 0.21 µs ("nobody ... kept 1 s"), i.e. never a Redis read.

    Input: ``bench_new_users.bench`` as main() runs it (Recommender(model, store, popular_ttl=1.0), 2,000 requests,
    200 lead-in, k=10) for users=None, counting store.popularity() reads.
    Expected: at least one read among the timed requests (the run spans at least one expiry), so the row is a 1 s TTL's.
    Actual: exactly one read in all, during the untimed lead-in.
    """
    reads: list[int] = []
    real = loaded.popularity
    calls = [0]

    rec = sv.Recommender(model, loaded, popular_ttl=sv.POPULAR_TTL)
    real_recommend = rec.recommend

    def counting_recommend(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return real_recommend(*args, **kwargs)

    def counting_popularity() -> Any:
        reads.append(calls[0])
        return real()

    monkeypatch.setattr(loaded, "popularity", counting_popularity)
    monkeypatch.setattr(rec, "recommend", counting_recommend)
    lead_in, requests = 200, 2000
    bench_new_users.bench(rec, [None] * requests, lead_in, 10)
    timed_reads = [c for c in reads if c > lead_in]
    assert timed_reads, f"popularity read at request(s) {reads} only: the timed {requests} never saw the list expire"
