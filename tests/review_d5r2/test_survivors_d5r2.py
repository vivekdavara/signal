"""Review D5, round 2: tests that kill mutants of round 1's fixes and of serve --workers that the suite let live.
Each passes on bb4a08d."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from signalrec import abtest, cli
from signalrec import serving as sv
from signalrec import tower as tw
from signalrec.featurestore import FeatureStore, FeatureStoreError
from tests.serving.conftest import URL

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import bench_new_users

BASE = 6040


# --- registry -------------------------------------------------------------------------------------------------------


def test_one_entry_more_than_the_count_is_refused(fresh: FeatureStore) -> None:
    """Kills `tonumber(count) < entries - 1`: HLEN 2 with count 1 must be refused, not only a larger gap."""
    registry, count, base = fresh._registry_keys
    fresh.client.hset(registry, mapping={"7": "1", "8": "2"})
    fresh.client.set(count, "1")
    fresh.client.set(base, str(BASE))
    with pytest.raises(FeatureStoreError, match="BADREGISTRY"):
        fresh.register(9, BASE)
    assert fresh.client.get(count) == b"1" and fresh.client.hlen(registry) == 2  # nothing written


@pytest.mark.parametrize("lost", ["count", "base"])
def test_a_registry_with_one_part_lost_is_refused(fresh: FeatureStore, lost: str) -> None:
    """Kills `not (count or base)`: with entries and the base but no counter (or the counter but no base), the next
    INCR would re-issue ordinal 1, which user 7 holds."""
    registry, count, base = fresh._registry_keys
    fresh.client.hset(registry, "7", "1")
    fresh.client.set(count, "1")
    fresh.client.set(base, str(BASE))
    fresh.client.delete(count if lost == "count" else base)
    with pytest.raises(FeatureStoreError, match="BADREGISTRY"):
        fresh.register(9, BASE)
    assert fresh.client.hlen(registry) == 1


def test_ordinal_zero_is_refused_on_both_paths(fresh: FeatureStore) -> None:
    """Kills `ordinal < 0`: ordinal 0 would be index base - 1, the model's own last user."""
    registry, count, base = fresh._registry_keys
    fresh.client.hset(registry, "7", "0")
    fresh.client.set(count, "1")
    fresh.client.set(base, str(BASE))
    with pytest.raises(FeatureStoreError, match="ordinal 0"):
        fresh.registered(7, BASE)
    with pytest.raises(FeatureStoreError, match="ordinal 0"):
        fresh.register(7, BASE)


def test_registered_refuses_an_ordinal_one_past_the_count(fresh: FeatureStore) -> None:
    """Kills `ordinal > count + 1`: ordinal count + 1 is the next one INCR hands out, to someone else."""
    registry, count, base = fresh._registry_keys
    fresh.client.hset(registry, "7", "2")
    fresh.client.set(count, "1")
    fresh.client.set(base, str(BASE))
    with pytest.raises(FeatureStoreError, match=r"outside 1\.\.1"):
        fresh.registered(7, BASE)


def test_registered_refuses_a_registry_without_its_counter(fresh: FeatureStore) -> None:
    """Kills dropping `or count_raw is None`: a lost counter is a FeatureStoreError (a 500 naming it), not a crash."""
    registry, _count, base = fresh._registry_keys
    fresh.client.hset(registry, "7", "1")
    fresh.client.set(base, str(BASE))
    with pytest.raises(FeatureStoreError, match="no base or counter"):
        fresh.registered(7, BASE)


def test_a_sixteen_digit_count_allocates_nobody(fresh: FeatureStore) -> None:
    """Kills `string.len(count) <= 16`: from count 10^15 a new user's index could pass base + 10^15 - 1."""
    _, count, base = fresh._registry_keys
    fresh.client.set(count, str(10**15))
    fresh.client.set(base, str(BASE))
    with pytest.raises(FeatureStoreError, match="BADCOUNT"):
        fresh.register(9, BASE)
    assert fresh.client.get(count) == str(10**15).encode()


def test_the_largest_base_is_two_to_the_62(fresh: FeatureStore) -> None:
    """Kills `MAX_BASE = 2**62 - 1` and `high = INT64_MAX` for the base: 2^62 is accepted, 2^62 + 1 is not."""
    assert fresh.register(7, 2**62) == (2**62, True)
    assert fresh.registered(7, 2**62) == 2**62
    with pytest.raises(ValueError, match="base"):
        fresh.registered(7, 2**62 + 1)


# --- serving: the exclusions are timed inside "retrieve" on both paths -----------------------------------------------


class _SlowList(list):  # type: ignore[type-arg]
    def __getitem__(self, i: Any) -> Any:
        time.sleep(0.05)
        return super().__getitem__(i)


@pytest.mark.parametrize("user", [0, None], ids=["two-tower", "popular"])
def test_building_the_exclusions_is_part_of_retrieve(model: tw.Model, loaded: FeatureStore, user: int | None) -> None:
    """Kills dropping `exclusions +` from either path's retrieve: a genre exclusion that takes 50 ms to build."""
    rec = sv.Recommender(model, loaded, n_candidates=40)
    rec._genre_movies = _SlowList(rec._genre_movies)
    out = rec.recommend(user, k=5, exclude_genres=[model.catalogue.genre_names[0]])
    assert out.source == ("popularity" if user is None else "two-tower")
    assert list(out.seconds) == list(sv.STAGES)
    assert out.seconds["retrieve"] >= 0.05
    assert all(v >= 0 for v in out.seconds.values())


# --- serve --workers ------------------------------------------------------------------------------------------------


def _serve_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str, model: str | None = None) -> list[Any]:
    import uvicorn

    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k not in cli.SERVE_ENV.values()})
    calls: list[Any] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    argv = ["serve", "synthetic", "--model", model or str(tmp_path / "model"), "--redis", URL, "--workers", "2"]
    calls.append(cli.main([*argv, *extra]))
    return calls


def test_workers_settings_round_trip_exactly(tmp_path: Path, model: tw.Model, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kills a lossy popular_ttl or share in the environment (str(round(...)), :.1f): every bit comes back."""
    tw.save(model, tmp_path / "model")
    share, ttl = 1 / 3, 0.123456789
    calls = _serve_workers(tmp_path, monkeypatch, "--popular-ttl", repr(ttl), "--experiment", f"ranker-v1:{share!r}")
    assert calls[-1] == 0
    assert cli._ttl(os.environ[cli.SERVE_ENV["popular_ttl"]]) == ttl
    assert cli._experiment(os.environ[cli.SERVE_ENV["experiment"]]) == abtest.Experiment("ranker-v1", share)
    app = cli.app_from_env()
    assert app.state.recommender.popular_ttl == ttl
    # The experiment reaches the worker's app: /health names it.
    from fastapi.testclient import TestClient

    health = TestClient(app).get("/health")
    assert health.status_code == 200 and health.json()["experiment"] == "ranker-v1"


def test_workers_without_an_experiment_hand_none(
    tmp_path: Path, model: tw.Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    tw.save(model, tmp_path / "model")
    assert _serve_workers(tmp_path, monkeypatch)[-1] == 0
    assert os.environ[cli.SERVE_ENV["experiment"]] == ""
    from fastapi.testclient import TestClient

    assert TestClient(cli.app_from_env()).get("/health").json()["experiment"] is None


def test_workers_get_an_absolute_model_path_from_a_relative_one(
    tmp_path: Path, model: tw.Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kills dropping `.resolve()`: a worker started elsewhere must find the model the parent checked."""
    tw.save(model, tmp_path / "model")
    monkeypatch.chdir(tmp_path)
    assert _serve_workers(tmp_path, monkeypatch, model="model")[-1] == 0
    assert os.environ[cli.SERVE_ENV["model"]] == str((tmp_path / "model").resolve())


def test_workers_with_no_model_fail_once_in_the_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kills dropping the parent's tw.load: no model is exit 1 before uvicorn starts a worker."""
    calls = _serve_workers(tmp_path, monkeypatch, model=str(tmp_path / "missing"))
    assert calls == [1]


def test_app_from_env_names_a_missing_popular_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {k: v for k, v in os.environ.items() if k not in cli.SERVE_ENV.values()}
    env.update({cli.SERVE_ENV["model"]: "m", cli.SERVE_ENV["redis"]: URL, cli.SERVE_ENV["namespace"]: "ns"})
    monkeypatch.setattr(os, "environ", env)
    with pytest.raises(RuntimeError, match="SIGNALREC_SERVE_POPULAR_TTL"):
        cli.app_from_env()


# --- bench_new_users.bench ------------------------------------------------------------------------------------------


class _Store:
    def __init__(self) -> None:
        self.calls = 0

    def popularity(self) -> np.ndarray:
        self.calls += 1
        return np.zeros(3, dtype=np.int64)


class _Rec:
    def __init__(self, fail_at: int | None = None) -> None:
        self.store = _Store()
        self.sent: list[float] = []
        self.fail_at = fail_at

    def recommend(self, user: Any, k: int) -> Any:
        self.sent.append(time.perf_counter())
        if self.fail_at is not None and len(self.sent) > self.fail_at:
            raise RuntimeError("boom")
        self.store.popularity()  # every request reads the hash
        return SimpleNamespace(source="popularity", seconds={s: 0.0 for s in sv.STAGES}, total=0.0)


def test_bench_paces_counts_only_timed_reads_and_restores() -> None:
    """Kills no pacing (due = start), `i > lead_in`, counting lead-in reads, and not deleting the shadow."""
    rec = _Rec()
    row = bench_new_users.bench(rec, [None] * 6, lead_in=4, k=10, rate=100.0)  # type: ignore[arg-type]
    assert len(rec.sent) == 10
    assert rec.sent[-1] - rec.sent[0] >= 9 / 100 - 0.002  # request i is due at i / rate
    assert row["popularity reads while timed"] == 6
    assert "popularity" not in vars(rec.store) and rec.store.calls == 10


def test_bench_restores_popularity_after_an_exception() -> None:
    rec = _Rec(fail_at=3)
    with pytest.raises(RuntimeError, match="boom"):
        bench_new_users.bench(rec, [None] * 6, lead_in=2, k=10, rate=1000.0)  # type: ignore[arg-type]
    assert "popularity" not in vars(rec.store)
