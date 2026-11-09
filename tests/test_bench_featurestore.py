"""scripts/bench_featurestore.py end to end on the synthetic dataset (needs Redis, like tests/test_featurestore.py)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import redis

from signalrec import cli

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_featurestore.py"
sys.path.insert(0, str(SCRIPT.parent))
import bench_featurestore  # noqa: E402

URL = os.environ.get("SIGNALREC_REDIS_URL", "redis://127.0.0.1:6383/0")


def test_the_benchmark_replays_reads_back_and_cleans_up(tmp_path: Path) -> None:
    client = redis.Redis.from_url(URL)
    try:
        client.ping()
    except redis.ConnectionError as exc:
        if os.environ.get("SIGNALREC_REQUIRE_REDIS") == "1":
            pytest.fail(f"SIGNALREC_REQUIRE_REDIS=1 but no Redis at {URL}: {exc}")
        pytest.skip(f"no Redis at {URL}")
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "60", "--items", "90"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    out = tmp_path / "featurestore.json"
    argv = ["synthetic", "--data-dir", str(tmp_path), "--url", URL, "--repeats", "2", "--batch", "7", "--out", str(out)]
    assert bench_featurestore.main(argv) == 0
    result = json.loads(out.read_text())
    assert result["users whose online features differ from offline"] == 0
    assert result["movies whose online popularity differs from offline"] == 0
    assert len(result["replay"]["runs"]) == 2
    read = result["read one user (FeatureStore.user)"]
    assert 0 < read["p50_us"] <= read["p95_us"] <= read["p99_us"]
    assert client.keys("bench-*") == []  # every namespace deleted
