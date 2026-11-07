"""Time each stage of the day-1 pipeline on a downloaded dataset, and the similarity build's peak memory by block size.

    .venv/bin/python scripts/bench_pipeline.py ml-1m --repeats 5 --out results/ml-1m/bench.json

Each case runs ``repeats`` times; the JSON keeps every run, the median and the minimum. Every run also records the
1-minute load average and how long the machine slept during it (macOS stops ``time.monotonic`` in sleep but not
``time.time``), so a disturbed run is visible rather than averaged in. Peak memory comes from ``tracemalloc``, which
sees NumPy's array allocations; it measures the Python-visible allocations of one call, not the process's RSS.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from signalrec import evaluate as ev
from signalrec import movielens, prepare
from signalrec.baselines import ItemKNN, Popularity, neighbours


def _run(fn: Callable[[], object], repeats: int) -> dict[str, Any]:
    runs = []
    for _ in range(repeats):
        load = os.getloadavg()[0]
        wall0, mono0 = time.time(), time.monotonic()
        t0 = time.perf_counter()
        fn()
        seconds = time.perf_counter() - t0
        # The two clocks disagree by microseconds without a sleep; clamp so jitter never prints as "-0.0".
        slept = max(0.0, (time.time() - wall0) - (time.monotonic() - mono0))
        runs.append({"seconds": seconds, "load_1m": load, "slept_s": round(slept, 3)})
    times = [r["seconds"] for r in runs]
    return {"median_s": statistics.median(times), "min_s": min(times), "runs": runs}


def _peak_bytes(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def bench(dataset: str, data_dir: Path, repeats: int, k: int, shrink: float) -> dict[str, Any]:
    raw = data_dir / "raw" / dataset
    parsed = movielens.load(dataset, raw)
    prepared = prepare.prepare(parsed, prepare.Params())
    history = prepared.history(include_val=True)
    knn = ItemKNN(k=k, shrink=shrink)
    knn.fit(history)
    popularity = Popularity()
    popularity.fit(history)
    cases = {
        "parse (movielens.load)": lambda: movielens.load(dataset, raw),
        "prepare (positives, k-core, split)": lambda: prepare.prepare(parsed, prepare.Params()),
        f"item-kNN fit (k={k}, shrink={shrink:g}, block 512)": lambda: ItemKNN(k=k, shrink=shrink).fit(history),
        "item-kNN evaluate (all users, full ranking)": lambda: ev.evaluate(knn, history, prepared.test_items, "test"),
        "popularity evaluate (all users, full ranking)": lambda: ev.evaluate(
            popularity, history, prepared.test_items, "test"
        ),
    }
    timings = {name: _run(fn, repeats) for name, fn in cases.items()}
    n_items = prepared.n_items
    memory = {}
    for block in sorted({b for b in (128, 512, 2048) if b < n_items} | {n_items}):
        peak = _peak_bytes(lambda block=block: neighbours(history, k, shrink, block))  # type: ignore[misc]
        memory[str(block)] = {"peak_traced_bytes": peak, "items_x_block_float64_bytes": n_items * block * 8}
    return {
        "command": "python " + " ".join([Path(sys.argv[0]).as_posix(), *sys.argv[1:]]),
        "dataset": dataset,
        "users": prepared.n_users,
        "items": n_items,
        "history_nnz": int(history.nnz),
        "repeats": repeats,
        "timings": timings,
        "similarity peak memory by block": memory,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(movielens.FORMATS))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--shrink", type=float, default=10.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = bench(args.dataset, args.data_dir, args.repeats, args.k, args.shrink)
    for name, t in result["timings"].items():
        loads = [r["load_1m"] for r in t["runs"]]
        slept = max(r["slept_s"] for r in t["runs"])
        print(
            f"{name}: median {t['median_s']:.3f} s, min {t['min_s']:.3f} s "
            f"(load {min(loads):.1f}-{max(loads):.1f}, slept {slept} s)"
        )
    for block, m in result["similarity peak memory by block"].items():
        print(f"similarity block {block}: peak {m['peak_traced_bytes'] / 2**20:.1f} MiB")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
