"""Review D4 round 2: scripts/bench_serving.py's open-loop summary with failures."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import bench_serving as bs


def test_fast_failures_cannot_make_the_percentiles_look_better() -> None:
    """Defect (bench_serving.py:213, summarise_open): round 1 counts a failed request at the time it failed, and the
    docstring says that means "failures can't make the percentiles look better". They can: a request that fails fast
    (an HTTP 503 when Redis is down, a 404, a refused connection: make_send raises on any non-200 at once) adds a
    latency far below the successful ones, so p50/p95/p99 drop compared with leaving it out. A run where 10% of
    requests error in 0.1 ms reports a better median than the same run without the errors.

    Input: 90 timed rows with latencies 1..90 ms, and 10 timed failures that failed 0.1 ms after they were due.
    Expected: every percentile at least what the 90 successes alone give (a failure is at least as bad as the slowest
    success, e.g. counted as infinitely slow).
    Actual: p50 45 ms -> 40 ms, p95 86 -> 85 ms, p99 90 -> 89 ms.
    """
    start = 0.0
    rows = [(i, float(i), float(i), i + (i + 1) / 1000, 0.0) for i in range(90)]
    failures = [(90 + j, "RuntimeError('HTTP 503')", float(90 + j), 90 + j + 0.0001) for j in range(10)]
    alone = bs.summarise_open(100.0, 0, 90, start, rows, [])["latency from scheduled time (µs)"]
    both = bs.summarise_open(100.0, 0, 100, start, rows, failures)["latency from scheduled time (µs)"]
    # Adapted after the fix: a percentile decided by a failure is None, "slower than any success" (worst of all).
    worse = {p: both[p] is None or both[p] >= alone[p] for p in ("p50", "p95", "p99")}
    assert all(worse.values()), f"successes alone {alone}, with fast failures {both}"
