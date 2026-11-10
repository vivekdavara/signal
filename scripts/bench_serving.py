"""Serving latency (DESIGN.md, "Day 4: serving and experiments"): per stage in-process, then end to end over HTTP in a
closed loop and an open loop at fixed rates.

1. **Stages, in-process.** ``Recommender.recommend`` for every user once (a seeded order, k = 10), after an untimed
   lead-in; p50/p95/p99/max per stage and in total. No HTTP: the pipeline's own cost.
2. **Closed loop over HTTP.** ``signalrec serve`` in a subprocess (one uvicorn process); ``c`` clients each send their
   next request when the last answer arrives, for ``--seconds``. Throughput and latency at each concurrency. A closed
   loop slows its clients down when the server is slow, so its latencies hide queueing (coordinated omission): it
   measures capacity, not the latency users would see at a given load.
3. **Open loop over HTTP.** Requests are scheduled at a fixed rate (evenly spaced) whatever the server does, and each
   latency is measured **from its scheduled time**, so time a request spent waiting to be sent counts. A paced
   lead-in at the same rate is sent first and not timed. The client's own lateness (sent − scheduled) is reported, so
   a client that can't keep up is visible rather than hidden in the numbers.

Every response's ``Server-Timing`` header is summed, so the HTTP numbers can be split into the pipeline's time and the
rest (framework, HTTP, client, scheduling).

Needs the model (``signalrec export``) and the store (``signalrec load-store``). Take it on a quiet machine.

    python scripts/bench_serving.py ml-1m --out results/ml-1m/serving-bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import redis
from redis._parsers import _HiredisParser, _RESP3Parser

from signalrec import prepare
from signalrec import tower as tw
from signalrec.featurestore import FeatureStore
from signalrec.serving import STAGES, Recommender

PERCENTILES = ("50", "95", "99")
PORT = 8116
# redis-py parses replies in Python unless hiredis (its C parser, the redis[hiredis] extra) is installed. The stage
# benchmark runs with each, the same code otherwise: the before and after of day 4's one optimization.
PARSERS = {"python (RESP3)": _RESP3Parser, "hiredis": _HiredisParser}


def percentile(values: Sequence[float], p: str) -> float:
    """The nearest-rank percentile: the smallest value with at least p% of the values at or below it.

    The rank is computed exactly (``Fraction``), not in floats: ``ceil(99.9 / 100 * 1000)`` is 1000 in doubles.
    """
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    q = Fraction(p) / 100
    if not 0 < q <= 1:
        raise ValueError(f"percentile must be in (0, 100], got {p}")
    rank = -(-q.numerator * len(ordered) // q.denominator)  # ceil(q * n) in integers
    return ordered[max(rank, 1) - 1]


def summary_us(seconds: Sequence[float]) -> dict[str, float]:
    """p50/p95/p99/max in microseconds."""
    us = [s * 1e6 for s in seconds]
    out = {f"p{p}": percentile(us, p) for p in PERCENTILES}
    out["max"] = max(us)
    out["mean"] = float(np.mean(us))
    return out


def server_total(header: str) -> float:
    """The sum of a Server-Timing header's durations, in seconds."""
    total = 0.0
    for part in header.split(","):
        for field in part.split(";")[1:]:
            name, _, value = field.strip().partition("=")
            if name == "dur":
                total += float(value) / 1000
    return total


# --- 1. stages ------------------------------------------------------------------------------------------------------


def bench_stages(rec: Recommender, users: Sequence[int], lead_in: int, k: int) -> dict[str, Any]:
    for u in users[:lead_in]:
        rec.recommend(u, k=k)
    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    totals = []
    for u in users:
        r = rec.recommend(u, k=k)
        for s in STAGES:
            per_stage[s].append(r.seconds[s])
        totals.append(r.total)
    return {
        "requests": len(users),
        "lead-in (untimed)": lead_in,
        "k": k,
        "stages (µs)": {s: summary_us(v) for s, v in per_stage.items()},
        "total (µs)": summary_us(totals),
    }


# --- 2 and 3. HTTP --------------------------------------------------------------------------------------------------

Send = Callable[[int], Awaitable[float]]  # one request for a user; returns the server's Server-Timing total


def make_send(client: httpx.AsyncClient, raw_users: Sequence[int], k: int) -> Send:
    async def send(i: int) -> float:
        r = await client.get(f"/recommendations/{raw_users[i % len(raw_users)]}", params={"k": k})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return server_total(r.headers.get("server-timing", ""))

    return send


async def closed_loop(send: Send, concurrency: int, seconds: float, offset: int = 0) -> dict[str, Any]:
    latencies: list[float] = []
    server: list[float] = []
    counter = [offset]
    stop = time.perf_counter() + seconds

    async def worker() -> None:
        while time.perf_counter() < stop:
            i = counter[0]
            counter[0] += 1
            t0 = time.perf_counter()
            server.append(await send(i))
            latencies.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "seconds": elapsed,
        "throughput (req/s)": len(latencies) / elapsed,
        "latency (µs)": summary_us(latencies),
        "server pipeline (µs)": summary_us(server),
    }


async def open_loop(
    send: Send,
    rate: float,
    seconds: float,
    lead_in: float,
    offset: int = 0,
    trace: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Requests every 1/rate seconds for lead_in + seconds; only those scheduled after the lead-in are timed."""
    n_lead = round(rate * lead_in)
    n = n_lead + round(rate * seconds)
    rows: list[tuple[int, float, float, float, float]] = []  # index, due, sent, done, server
    failures: list[str] = []
    start = time.perf_counter() + 0.05

    async def one(i: int, due: float) -> None:
        sent = time.perf_counter()
        try:
            server = await send(offset + i)
        except Exception as exc:  # a failed request is counted, never dropped from the books
            failures.append(repr(exc))
            return
        rows.append((i, due, sent, time.perf_counter(), server))

    tasks = []
    for i in range(n):
        due = start + i / rate
        delay = due - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(i, due)))
    await asyncio.gather(*tasks)
    timed = [r for r in rows if r[0] >= n_lead]
    if trace is not None:
        trace.extend({"i": r[0], "due": r[1] - start, "sent": r[2] - start, "done": r[3] - start} for r in timed)
    latency = [done - due for _, due, _, done, _ in timed]
    lateness = [sent - due for _, due, sent, _, _ in timed]
    first_due = min((r[1] for r in timed), default=start)
    last_done = max((r[3] for r in timed), default=start)
    return {
        "rate (req/s)": rate,
        "scheduled (timed)": n - n_lead,
        "lead-in requests (untimed)": n_lead,
        "completed (timed)": len(timed),
        "failed": len(failures),
        "first failures": failures[:3],
        "achieved (req/s)": len(timed) / (last_done - first_due) if timed else 0.0,
        "latency from scheduled time (µs)": summary_us(latency) if latency else None,
        "client lateness, sent - scheduled (µs)": summary_us(lateness) if lateness else None,
        "server pipeline (µs)": summary_us([r[4] for r in timed]) if timed else None,
    }


def start_server(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    command = [sys.executable, "-m", "signalrec.cli", "serve", args.dataset, "--port", str(args.port)]
    if args.model:
        command += ["--model", str(args.model)]
    command += ["--redis", args.redis, "--namespace", args.namespace]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the server exited: {proc.stderr.read().decode() if proc.stderr else ''}")
        try:
            if httpx.get(f"http://127.0.0.1:{args.port}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError("the server didn't answer /health within 60 s")


async def bench_http(args: argparse.Namespace, raw_users: Sequence[int]) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=args.max_connections, max_keepalive_connections=args.max_connections)
    out: dict[str, Any] = {"closed loop": [], "open loop": []}
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", limits=limits, timeout=30) as client:
        send = make_send(client, raw_users, args.k)
        offset = 0
        await closed_loop(send, 4, args.lead_in)  # untimed warm-up of the server and the connections
        for c in args.concurrency:
            row = await closed_loop(send, c, args.seconds, offset)
            offset += row["requests"]
            out["closed loop"].append(row)
            print(
                f"closed loop c={c}: {row['throughput (req/s)']:.0f} req/s, "
                f"p50 {row['latency (µs)']['p50']:.0f} µs, p99 {row['latency (µs)']['p99']:.0f} µs",
                flush=True,
            )
        trace: list[dict[str, float]] | None = [] if args.trace else None
        for rate in args.rates:
            row = await open_loop(send, rate, args.seconds, args.lead_in, offset, trace)
            offset += row["scheduled (timed)"] + row["lead-in requests (untimed)"]
            out["open loop"].append(row)
            lat = row["latency from scheduled time (µs)"] or {}
            print(
                f"open loop {rate:.0f} req/s: achieved {row['achieved (req/s)']:.0f}, failed {row['failed']}, "
                f"p50 {lat.get('p50', float('nan')):.0f} µs, p99 {lat.get('p99', float('nan')):.0f} µs",
                flush=True,
            )
        if trace is not None and args.trace:
            args.trace.write_text(json.dumps(trace) + "\n")
    return out


def _rates(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, help="default: models/<dataset>")
    parser.add_argument("--redis", default="redis://127.0.0.1:6383/0")
    parser.add_argument("--namespace", default="signal")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lead-in-requests", type=int, default=1000, help="untimed, before the stage benchmark")
    parser.add_argument("--lead-in", type=float, default=2.0, help="seconds, untimed, before each HTTP run")
    parser.add_argument("--seconds", type=float, default=10.0, help="each HTTP run's timed length")
    parser.add_argument("--concurrency", type=_ints, default=[1, 4, 16])
    parser.add_argument("--rates", type=_rates, default=[100.0, 200.0, 400.0])
    parser.add_argument("--max-connections", type=int, default=256)
    parser.add_argument("--no-http", action="store_true", help="the in-process stages only")
    parser.add_argument("--trace", type=Path, help="write every timed open-loop request's due/sent/done times here")
    parser.add_argument("--out", type=Path, help="default: results/<dataset>/serving-bench.json")
    args = parser.parse_args(argv)

    model = tw.load(args.model or Path("models") / args.dataset)
    prepared = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    users = np.random.default_rng(0).permutation(prepared.n_users).tolist()

    def recommender(parser: Any) -> Recommender:
        pool = redis.ConnectionPool.from_url(args.redis, parser_class=parser)
        return Recommender(model, FeatureStore(redis.Redis(connection_pool=pool), model.catalogue, args.namespace))

    rec = recommender(_HiredisParser)

    t0, slept0 = time.time(), time.time() - time.monotonic()
    result: dict[str, Any] = {
        "command": "python " + " ".join(["scripts/bench_serving.py", *(sys.argv[1:] if argv is None else argv)]),
        "dataset": args.dataset,
        "candidates": rec.n_candidates,
        "index": rec.retriever.spec.label(),
    }
    result["stages, in-process"] = {}
    for name, parser_class in PARSERS.items():
        row = bench_stages(recommender(parser_class), users, args.lead_in_requests, args.k)
        result["stages, in-process"][name] = row
        total = row["total (µs)"]
        print(
            f"in-process, {name} parser: p50 {total['p50']:.0f} µs, p95 {total['p95']:.0f} µs, "
            f"p99 {total['p99']:.0f} µs",
            flush=True,
        )
    if not args.no_http:
        raw_users = [int(model.user_raw_ids[u]) for u in users]
        proc = start_server(args)
        try:
            result["http"] = asyncio.run(bench_http(args, raw_users))
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    result["wall seconds"] = time.time() - t0
    # time.time() - time.monotonic() grows by the time the machine slept (the monotonic clock stops in sleep).
    result["slept seconds (estimate)"] = max(0.0, (time.time() - time.monotonic()) - slept0)
    result["environment"] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpus": os.cpu_count(),
        "server": "one uvicorn process, sync endpoints on its thread pool; client and server on the same machine",
        "redis-py": redis.__version__,
        "server's parser": "hiredis (redis-py's default when it is installed)",
    }
    out = args.out or Path("results") / args.dataset / "serving-bench.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
