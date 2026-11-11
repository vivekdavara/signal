"""What a user with no positives costs to serve, in-process, with the popular list read every time or kept for a second.

    .venv/bin/python scripts/bench_new_users.py ml-1m --out results/ml-1m/new-users-bench.json

Three kinds of request through ``Recommender.recommend`` (k = 10, hiredis), each after an untimed lead-in:

- **known users**: the day-4 path (features, user tower, FAISS, popularity of the candidates, ranker, filter), for
  reference;
- **new users**: dense indices past every prepared user, with nothing stored under them: the store is read (one
  MULTI/EXEC that finds nothing) and the popular list answers;
- **nobody** (``user=None``): an id with no events at all, as ``GET /recommendations`` serves an unregistered id: no
  store read for the user.

The popular path runs with ``popular_ttl=0`` (the whole popularity hash read, about 3,000 fields, and sorted on every
request) and with the default 1 s, so the cost of exactness is measured, not assumed. Requests are **paced** at
``--rate`` per second (1,000 by default), lead-in included, so a run of 5,000 timed requests spans about five
expiries of a 1 s list and its tail includes the reads that refresh it; every run reports how many whole-hash
reads of popularity (``store.popularity()``, not the known users' HMGET of their candidates) happened among its timed
requests (review of day 5: unpaced, 2,200 requests took 0.3 s, and the "kept
1 s" rows measured a list that never expired). Needs the model (``signalrec export``) and the store (``signalrec
load-store``); take it on a quiet machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import redis

from signalrec import tower as tw
from signalrec.featurestore import FeatureStore
from signalrec.serving import POPULAR_TTL, STAGES, Recommender

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_serving import percentile, summary_us


def bench(rec: Recommender, users: list[int | None], lead_in: int, k: int, rate: float = 1000.0) -> dict[str, Any]:
    """Each request at its scheduled time (``rate`` per second from the first lead-in request), the first ``lead_in``
    untimed; counts the store's popularity reads among the timed ones."""
    if not rate > 0:
        raise ValueError(f"rate must be > 0, got {rate!r}")
    reads = [0]
    real_popularity = rec.store.popularity
    timing = [False]

    def counted() -> Any:
        if timing[0]:
            reads[0] += 1
        return real_popularity()

    shadowed = "popularity" in vars(rec.store)
    rec.store.popularity = counted  # type: ignore[method-assign]
    per_stage: dict[str, list[float]] = {s: [] for s in STAGES}
    totals, sources = [], set()
    start = time.perf_counter()
    try:
        for i, u in enumerate([*users[:lead_in], *users]):
            due = start + i / rate
            while (wait := due - time.perf_counter()) > 0:
                time.sleep(wait if wait > 0.002 else 0)
            timing[0] = i >= lead_in
            r = rec.recommend(u, k=k)
            if not timing[0]:
                continue
            sources.add(r.source)
            for s in STAGES:
                per_stage[s].append(r.seconds[s])
            totals.append(r.total)
    finally:
        if shadowed:
            rec.store.popularity = real_popularity  # type: ignore[method-assign]
        else:
            del rec.store.popularity
    return {
        "requests": len(users),
        "lead-in (untimed)": lead_in,
        "rate (req/s)": rate,
        "seconds": time.perf_counter() - start,
        "popularity reads while timed": reads[0],
        "source": sorted(sources),
        "stages (µs)": {s: summary_us(v) for s, v in per_stage.items()},
        "total (µs)": {**summary_us(totals), "p99.9": percentile(totals, "99.9") * 1e6},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--model", type=Path, help="default: models/<dataset>")
    parser.add_argument("--redis", default="redis://127.0.0.1:6383/0")
    parser.add_argument("--namespace", default="signal")
    parser.add_argument("--requests", type=int, default=5000, help="timed requests per kind (default 5000)")
    parser.add_argument("--lead-in", type=int, default=500, help="untimed requests first (default 500)")
    parser.add_argument("--rate", type=float, default=1000.0, help="requests per second, paced (default 1000)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, help="default: results/<dataset>/new-users-bench.json")
    args = parser.parse_args(argv)

    model = tw.load(args.model or Path("models") / args.dataset)
    store = FeatureStore(redis.Redis.from_url(args.redis), model.catalogue, args.namespace)
    n_users = len(model.user_raw_ids)
    rng = np.random.default_rng(0)
    known: list[int | None] = [int(u) for u in rng.integers(0, n_users, size=args.requests)]
    new_ids = [n_users + 10**9 + i for i in range(args.requests)]  # nothing stored under these
    for u in new_ids[:50]:
        if store.user(u).positives or len(store.serving_view(u)[1]):
            print(f"bench_new_users.py: index {u} has data in the store; not a new user", file=sys.stderr)
            return 1
    new: list[int | None] = list(new_ids)
    nobody: list[int | None] = [None] * args.requests

    t0, slept0 = time.time(), time.time() - time.monotonic()
    load_before = os.getloadavg()
    runs: dict[str, Any] = {}
    plan = [
        ("known users", known, POPULAR_TTL),
        ("new users, popular list read every time (popular_ttl 0)", new, 0.0),
        (f"new users, popular list kept {POPULAR_TTL:g} s", new, POPULAR_TTL),
        ("nobody, popular list read every time (popular_ttl 0)", nobody, 0.0),
        (f"nobody, popular list kept {POPULAR_TTL:g} s", nobody, POPULAR_TTL),
    ]
    for name, users, ttl in plan:
        row = bench(Recommender(model, store, popular_ttl=ttl), users, args.lead_in, args.k, args.rate)
        runs[name] = row
        total = row["total (µs)"]
        print(
            f"{name}: p50 {total['p50']:.0f} µs, p99 {total['p99']:.0f} µs, p99.9 {total['p99.9']:.0f} µs, "
            f"max {total['max']:.0f} µs, mean {total['mean']:.1f} µs; {row['popularity reads while timed']} reads",
            flush=True,
        )
    result = {
        "command": "python " + " ".join(["scripts/bench_new_users.py", *(sys.argv[1:] if argv is None else argv)]),
        "dataset": args.dataset,
        "k": args.k,
        "runs": runs,
        "wall seconds": time.time() - t0,
        "load average": {"before": list(load_before), "after": list(os.getloadavg())},
        "slept seconds (estimate)": max(0.0, (time.time() - time.monotonic()) - slept0),
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
            "cpus": os.cpu_count(),
            "redis-py": redis.__version__,
            "parser": "hiredis",
        },
    }
    out = args.out or Path("results") / args.dataset / "new-users-bench.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
