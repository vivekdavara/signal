"""scripts/bench_new_users.py against the synthetic model and store: every kind of request takes the path it names."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from signalrec import tower as tw
from signalrec.featurestore import FeatureStore
from signalrec.tower import Model
from tests.serving.conftest import URL

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bench_new_users.py"
sys.path.insert(0, str(SCRIPT.parent))
import bench_new_users  # noqa: E402


def test_each_kind_takes_its_path(tmp_path: Path, model: Model, loaded: FeatureStore) -> None:
    tw.save(model, tmp_path / "model")
    out = tmp_path / "bench.json"
    argv = ["synthetic", "--model", str(tmp_path / "model"), "--redis", URL, "--namespace", loaded.namespace]
    assert bench_new_users.main([*argv, "--requests", "30", "--lead-in", "5", "--out", str(out)]) == 0
    runs = json.loads(out.read_text())["runs"]
    assert list(runs) == [
        "known users",
        "new users, popular list read every time (popular_ttl 0)",
        "new users, popular list kept 1 s",
        "nobody, popular list read every time (popular_ttl 0)",
        "nobody, popular list kept 1 s",
    ]
    assert runs["known users"]["source"] == ["two-tower"]
    for name, row in runs.items():
        assert row["requests"] == 30 and row["total (µs)"]["p50"] > 0
        if name != "known users":
            assert row["source"] == ["popularity"]


def test_an_index_with_data_is_not_a_new_user(tmp_path: Path, model: Model, fresh: FeatureStore) -> None:
    from signalrec import features as fe

    tw.save(model, tmp_path / "model")
    fresh.record(fe.Event(len(model.user_raw_ids) + 10**9, 0, 0))
    argv = ["synthetic", "--model", str(tmp_path / "model"), "--redis", URL, "--namespace", fresh.namespace]
    assert bench_new_users.main([*argv, "--requests", "5", "--out", str(tmp_path / "x.json")]) == 1
    assert not (tmp_path / "x.json").exists()


def test_requests_are_paced_and_refreshes_counted(model: Model, fresh: FeatureStore) -> None:
    from signalrec import serving as sv

    rec = sv.Recommender(model, fresh, popular_ttl=0.05)
    row = bench_new_users.bench(rec, [None] * 60, 10, 5, rate=200.0)  # 70 requests at 200/s: 0.35 s
    assert row["seconds"] >= 69 / 200
    # The list expires every 50 ms over the 0.3 s of timed requests: about 6 reads, never one per request.
    assert 3 <= row["popularity reads while timed"] <= 12
    assert row["total (µs)"]["p99.9"] >= row["total (µs)"]["p99"]
    assert "popularity" not in vars(fresh)  # the counting wrapper is gone again
