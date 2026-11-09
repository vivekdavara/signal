"""scripts/bench_retrieval.py end to end on the synthetic dataset: the exact index must score recall 1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from signalrec import cli

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_retrieval.py"
sys.path.insert(0, str(SCRIPT.parent))
import bench_retrieval  # noqa: E402


def test_the_benchmark_runs_and_the_exact_index_is_exact(tmp_path: Path) -> None:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "250", "--items", "200"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    source = tmp_path / "twotower.json"
    two_tower = ["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", "0.1:on", "--histories", "5"]
    assert (
        cli.main([*two_tower, "--epochs", "1", "--seeds", "0", "--knn-grid", "5:0", "--quiet", "--out", str(source)])
        == 0
    )
    out = tmp_path / "retrieval.json"
    argv = ["synthetic", "--data-dir", str(tmp_path), "--from", str(source), "--n", "10", "--out", str(out)]
    assert bench_retrieval.main([*argv, "--synthetic-items", "3000", "--synthetic-queries", "30"]) == 0
    result = json.loads(out.read_text())
    real = {row["index"]: row for row in result["real catalogue"]["indexes"]}
    assert real["flat (exact)"]["recall@10 of the exact top 10"] == 1.0
    assert real["flat (exact)"]["queries below 0.99 recall@10"] == 0
    assert len(real) == len(bench_retrieval.REAL_INDEXES)
    for row in real.values():
        assert 0 <= row["recall@10 of the exact top 10"] <= 1
        assert row["latency"]["p50_us"] <= row["latency"]["p95_us"] <= row["latency"]["p99_us"]
    synthetic = {row["index"]: row for row in result["synthetic catalogue"]["indexes"]}
    assert synthetic["flat (exact)"]["recall@10 of the exact top 10"] == 1.0
    assert "SYNTHETIC" in result["synthetic catalogue"]


def test_the_synthetic_catalogue_is_unit_vectors_near_real_ones() -> None:
    rng = np.random.default_rng(0)
    items = rng.normal(size=(50, 8)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    synthetic = bench_retrieval.synthetic_catalogue(items, 400, noise=0.1, seed=1)
    assert synthetic.shape == (400, 8) and synthetic.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(synthetic, axis=1), 1.0, rtol=1e-5)
    nearest = (synthetic @ items.T).max(axis=1)
    assert (nearest > 0.9).all()  # each one close to the movie it was drawn from
    again = bench_retrieval.synthetic_catalogue(items, 400, noise=0.1, seed=1)
    assert np.array_equal(synthetic, again)
