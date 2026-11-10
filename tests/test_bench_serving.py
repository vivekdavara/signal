"""scripts/bench_serving.py's measuring code: the percentile, the Server-Timing sum, and both load loops against a fake
server whose latency is known."""

from __future__ import annotations

import asyncio
import math
import sys
import time
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_serving as bs


@given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=300), st.sampled_from(["50", "95", "99", "99.9", "100"]))
def test_percentile_is_the_nearest_rank(values: list[float], p: str) -> None:
    ordered = sorted(values)
    got = bs.percentile(values, p)
    # The smallest value with at least p% of the values at or below it.
    for i, v in enumerate(ordered):
        if (i + 1) * 1000 >= round(float(p) * 10) * len(ordered):
            assert got == v
            break


def test_percentile_ranks_are_exact_not_float() -> None:
    values = list(range(1, 1001))
    assert bs.percentile(values, "99.9") == 999  # ceil(99.9 / 100 * 1000) is 1000 in doubles
    assert bs.percentile(values, "50") == 500
    assert bs.percentile(values, "100") == 1000
    assert bs.percentile([7.0], "1") == 7.0
    for bad in ("0", "101", "-5"):
        with pytest.raises(ValueError):
            bs.percentile(values, bad)
    with pytest.raises(ValueError):
        bs.percentile([], "50")


def test_server_timing_is_summed_in_seconds() -> None:
    header = "features;dur=0.672, user-tower;dur=0.087, retrieve;dur=0.529, popularity;dur=0.698"
    assert bs.server_total(header) == pytest.approx((0.672 + 0.087 + 0.529 + 0.698) / 1000)
    assert bs.server_total("") == 0.0
    assert bs.server_total('x;desc="a";dur=1.5') == pytest.approx(0.0015)


def _fake(latency: float, fail_every: int = 0) -> tuple[bs.Send, list[int]]:
    calls: list[int] = []

    async def send(i: int) -> float:
        calls.append(i)
        await asyncio.sleep(latency)
        if fail_every and i % fail_every == 0:
            raise RuntimeError("HTTP 500")
        return latency / 2

    return send, calls


def test_the_open_loop_keeps_its_schedule_whatever_the_server_does() -> None:
    # A server taking 50 ms at 200 requests/s: a closed loop of one would send 20/s; the open loop still sends 200/s
    # and its latencies count from the scheduled times.
    send, calls = _fake(0.05)
    row = asyncio.run(bs.open_loop(send, rate=200, seconds=0.5, lead_in=0.1, offset=1000))
    assert row["lead-in requests (untimed)"] == 20
    assert row["scheduled (timed)"] == 100
    assert row["completed (timed)"] == 100 and row["failed"] == 0
    assert sorted(calls) == list(range(1000, 1120))
    lat = row["latency from scheduled time (µs)"]
    assert lat["p50"] >= 50_000 and lat["p99"] < 150_000  # loose: CI runners oversleep
    assert row["server pipeline (µs)"]["p50"] == pytest.approx(25_000)
    assert 150 < row["achieved (req/s)"] < 250


def test_open_loop_failures_are_counted_not_dropped() -> None:
    send, _ = _fake(0.001, fail_every=10)
    row = asyncio.run(bs.open_loop(send, rate=500, seconds=0.2, lead_in=0.0))
    assert row["scheduled (timed)"] == 100
    assert row["failed"] == 10 and row["completed (timed)"] == 90
    assert row["first failures"] and "HTTP 500" in row["first failures"][0]


def test_a_slow_server_shows_up_as_latency_from_the_schedule() -> None:
    # Capacity 1 at a time (a lock), 10 ms each, asked for 200/s: the queue grows, and so must the measured latency.
    lock = asyncio.Lock()

    async def send(i: int) -> float:
        async with lock:
            await asyncio.sleep(0.01)
        return 0.01

    row = asyncio.run(bs.open_loop(send, rate=200, seconds=0.3, lead_in=0.0))
    lat = row["latency from scheduled time (µs)"]
    assert lat["max"] > 200_000  # 60 requests at 100/s of capacity: the last waits about 0.3 s
    assert row["client lateness, sent - scheduled (µs)"]["p99"] < 50_000  # the client itself stayed on time


def test_the_closed_loop_waits_for_each_answer() -> None:
    send, calls = _fake(0.01)
    row = asyncio.run(bs.closed_loop(send, concurrency=2, seconds=0.2))
    # Two clients, 10 ms each: about 40 requests, never more than 2 in flight.
    assert 16 <= row["requests"] <= 44
    assert len(calls) == row["requests"]
    assert row["latency (µs)"]["p50"] >= 10_000
    assert math.isclose(row["throughput (req/s)"], row["requests"] / row["seconds"])


def test_the_trace_records_every_timed_request() -> None:
    send, _ = _fake(0.002)
    trace: list[dict[str, float]] = []
    asyncio.run(bs.open_loop(send, rate=100, seconds=0.2, lead_in=0.1, trace=trace))
    assert [t["i"] for t in sorted(trace, key=lambda t: t["i"])] == list(range(10, 30))
    assert all(t["due"] <= t["sent"] <= t["done"] for t in trace)


def test_the_parts_of_a_schedule_send_every_request_once_at_its_time() -> None:
    send, calls = _fake(0.001)

    async def both() -> list[tuple[list[bs.Row], list[tuple[int, str]]]]:
        start = time.perf_counter() + 0.02
        return list(await asyncio.gather(*(bs.open_loop_part(send, 400, 80, start, j, 3) for j in range(3))))

    parts = asyncio.run(both())
    rows = [r for part, _ in parts for r in part]
    assert sorted(r[0] for r in rows) == list(range(80))
    assert sorted(calls) == list(range(80))
    for j, (part, _) in enumerate(parts):
        assert {r[0] % 3 for r in part} == {j}
    start = min(r[1] for r in rows)
    assert all(r[1] == pytest.approx(start + r[0] / 400) for r in rows)  # one schedule, whichever part sent it


def test_lead_in_failures_are_not_counted_with_the_timed_ones() -> None:
    send, _ = _fake(0.001, fail_every=5)
    row = asyncio.run(bs.open_loop(send, rate=500, seconds=0.1, lead_in=0.1))
    assert row["lead-in requests (untimed)"] == 50 and row["scheduled (timed)"] == 50
    assert row["failed"] == 10 and row["completed (timed)"] == 40
