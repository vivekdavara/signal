"""The feature store's cost on a dataset: replaying its events into Redis, and reading one user back.

    .venv/bin/python scripts/bench_featurestore.py ml-1m --out results/ml-1m/featurestore.json

Needs Redis (``--url``, default the project's redis://127.0.0.1:6383/0); everything goes under a fresh namespace,
deleted afterwards. Measured:

- **Replay:** every training + validation positive through the Lua script, ``--batch`` scripts per round trip; events
  per second, best of ``--repeats`` (each repeat into a fresh namespace).
- **Read:** ``FeatureStore.user`` (one MULTI/EXEC of four commands) and ``FeatureStore.popularity`` (one HGETALL of
  the whole catalogue), one call at a time over every user, after an untimed warm-up; percentiles by the inverted CDF.
- **The skew check, at scale:** every user read back is compared with the offline features; the count of differences
  is recorded (it must be 0).

A local Redis on the same machine: the round trip is loopback, not a network. The load average and any sleep are
recorded, as in ``bench_pipeline.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import redis

from signalrec import features as fe
from signalrec import prepare
from signalrec.featurestore import DEFAULT_URL, FeatureStore

WARMUP = 200
PERCENTILES = (50, 95, 99)


def _percentiles(seconds: list[float]) -> dict[str, float]:
    values = np.array(seconds) * 1e6
    out = {f"p{q}_us": float(np.percentile(values, q, method="inverted_cdf")) for q in PERCENTILES}
    out["mean_us"] = float(values.mean())
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch < 1 or args.repeats < 1:
        parser.error("--batch and --repeats must be >= 1")
    p = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    catalogue = fe.Catalogue.from_prepared(p)
    events = fe.events(p, include_val=True)
    offline_users, offline_popularity = fe.offline(p, include_val=True)
    client = redis.Redis.from_url(args.url)
    client.ping()

    load = os.getloadavg()[0]
    wall0, mono0 = time.time(), time.monotonic()
    replays = []
    store: FeatureStore | None = None
    try:
        for _ in range(args.repeats):
            if store is not None:
                store.clear()
            store = FeatureStore(client, catalogue, namespace=f"bench-{uuid.uuid4().hex}")
            stats = store.replay(events, batch=args.batch)
            replays.append({"seconds": stats.seconds, "events per second": stats.events / stats.seconds})
            print(f"replay: {stats.events} events in {stats.seconds:.2f} s", file=sys.stderr, flush=True)
        assert store is not None
        for u in range(min(WARMUP, p.n_users)):
            store.user(u)
        read: list[float] = []
        differ = 0
        for u in range(p.n_users):
            t0 = time.perf_counter()
            got = store.user(u)
            read.append(time.perf_counter() - t0)
            differ += got != offline_users[u]
        popularity: list[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            online_popularity = store.popularity()
            popularity.append(time.perf_counter() - t0)
        popularity_differs = int((online_popularity != offline_popularity).sum())
    finally:
        if store is not None:
            store.clear()
    slept = max(0.0, (time.time() - wall0) - (time.monotonic() - mono0))

    best = min(replays, key=lambda r: r["seconds"])
    info = client.info("server")
    result = {
        "command": "python scripts/bench_featurestore.py " + shlex.join(argv),
        "dataset": args.dataset,
        "events": len(events),
        "users": p.n_users,
        "items": p.n_items,
        "replay": {"batch": args.batch, "runs": replays, "best events per second": best["events per second"]},
        "read one user (FeatureStore.user)": _percentiles(read),
        "read popularity (FeatureStore.popularity, 50 calls)": _percentiles(popularity),
        "users whose online features differ from offline": differ,
        "movies whose online popularity differs from offline": popularity_differs,
        "load_1m": load,
        "slept_s": round(slept, 3),
        "environment": {
            "python": platform.python_version(),
            "redis-py": redis.__version__,
            "redis server": info.get("redis_version"),
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in list(result)[5:10]}, indent=2), file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0 if differ == 0 and popularity_differs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
