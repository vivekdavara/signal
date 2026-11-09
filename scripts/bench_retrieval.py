"""Approximate retrieval against exact: recall of the top n and the latency of one query (DESIGN.md, "Day 3").

    .venv/bin/python scripts/bench_retrieval.py ml-1m --out results/ml-1m/retrieval.json

1. **The real catalogue.** The reported two-tower configuration (``twotower.json``) is trained on training +
   validation, as for its test run; every user's vector comes from their latest history. The exact answer is the top
   ``n`` unseen movies by float64 dot products, ties to the smaller index (``evaluate.top_k``, the protocol's ranking).
   Every index answers the same queries with the user's movies excluded; reported: the mean share of the exact top
   ``n`` it returns (recall@n), the share of users whose test movie it returns, and the latency of one query.
2. **A SYNTHETIC large catalogue** (labelled so in the output): ``--synthetic-items`` vectors, each a real movie's
   vector plus Gaussian noise, renormalised, queried with real user vectors and no exclusion. It shows how the
   indexes scale past ml-1m's 3,125 movies, which is small enough that exact search is the obvious choice.

Latency: one query per call, on ``--threads`` FAISS threads (1 by default: the serving path is one request at a time),
after an untimed warm-up of 200 queries; percentiles by the inverted CDF (the smallest time at least that share of
queries took). The load average and any sleep during the run are recorded, as in ``bench_pipeline.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from signalrec import evaluate as ev
from signalrec import features as fe
from signalrec import prepare
from signalrec import retrieval as rt
from signalrec import twotower as tt
from signalrec.cli import _reported_config

WARMUP = 200
PERCENTILES = (50, 95, 99)

REAL_INDEXES = (
    rt.IndexSpec("flat"),
    rt.IndexSpec("hnsw", m=16, ef_search=200),
    rt.IndexSpec("hnsw", m=32, ef_search=200),
    rt.IndexSpec("hnsw", m=32, ef_search=400),
    rt.IndexSpec("ivf", nlist=56, nprobe=1),
    rt.IndexSpec("ivf", nlist=56, nprobe=4),
    rt.IndexSpec("ivf", nlist=56, nprobe=8),
    rt.IndexSpec("ivf", nlist=56, nprobe=16),
)
SYNTHETIC_INDEXES = (
    rt.IndexSpec("flat"),
    rt.IndexSpec("hnsw", m=32, ef_search=200),
    rt.IndexSpec("hnsw", m=32, ef_search=400),
    rt.IndexSpec("ivf", nlist=512, nprobe=8),
    rt.IndexSpec("ivf", nlist=512, nprobe=32),
)


def _percentiles(seconds: list[float]) -> dict[str, float]:
    values = np.array(seconds) * 1e6
    out = {f"p{q}_us": float(np.percentile(values, q, method="inverted_cdf")) for q in PERCENTILES}
    out["mean_us"] = float(values.mean())
    return out


def measure(
    retriever: rt.Retriever,
    queries: rt.Float32Array,
    n: int,
    exact: list[ev.IntArray],
    exclude: list[ev.IntArray] | None,
    targets: ev.IntArray | None,
) -> dict[str, Any]:
    """Recall of the exact top n, the targets' hit rate, and per-query latency (after the warm-up)."""
    none = np.empty(0, dtype=np.int64)
    for q in range(min(WARMUP, len(queries))):
        retriever.search(queries[q], n, none if exclude is None else exclude[q])
    seconds: list[float] = []
    recall, hits = [], []
    for q in range(len(queries)):
        excluded = none if exclude is None else exclude[q]
        t0 = time.perf_counter()
        ids, _ = retriever.search(queries[q], n, excluded)
        seconds.append(time.perf_counter() - t0)
        want = exact[q]
        recall.append(len(np.intersect1d(ids, want)) / len(want) if len(want) else 1.0)
        if targets is not None:
            hits.append(bool((ids == targets[q]).any()))
    out: dict[str, Any] = {f"recall@{n} of the exact top {n}": float(np.mean(recall))}
    out[f"queries below 0.99 recall@{n}"] = int((np.array(recall) < 0.99).sum())
    if targets is not None:
        out[f"held-out movie in the top {n} (share of users)"] = float(np.mean(hits))
    out["latency"] = _percentiles(seconds)
    return out


def _bench(
    items: rt.Float32Array,
    specs: tuple[rt.IndexSpec, ...],
    queries: rt.Float32Array,
    n: int,
    exact: list[ev.IntArray],
    exclude: list[ev.IntArray] | None,
    targets: ev.IntArray | None,
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        load = os.getloadavg()[0]
        wall0, mono0 = time.time(), time.monotonic()
        t0 = time.perf_counter()
        retriever = rt.Retriever(items, spec)
        build = time.perf_counter() - t0
        row = {"index": spec.label(), "build_s": build, **measure(retriever, queries, n, exact, exclude, targets)}
        row["load_1m"] = load
        row["slept_s"] = round(max(0.0, (time.time() - wall0) - (time.monotonic() - mono0)), 3)
        rows.append(row)
        lat = row["latency"]
        print(
            f"{spec.label()}: recall {row[f'recall@{n} of the exact top {n}']:.4f}, "
            f"p50 {lat['p50_us']:.0f} us, p99 {lat['p99_us']:.0f} us, build {build:.2f} s",
            file=sys.stderr,
            flush=True,
        )
    return rows


def synthetic_catalogue(items: rt.Float32Array, size: int, noise: float, seed: int) -> rt.Float32Array:
    """``size`` SYNTHETIC vectors: real movie vectors drawn with replacement plus Gaussian noise, renormalised."""
    rng = np.random.default_rng(seed)
    base = items[rng.integers(0, len(items), size=size)]
    x = base + noise * rng.standard_normal(size=base.shape).astype(np.float32) / np.sqrt(items.shape[1])
    out: rt.Float32Array = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--from", dest="source", type=Path, help="default: results/<dataset>/twotower.json")
    parser.add_argument("--n", type=int, default=200, help="candidates per query (default 200)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--synthetic-items", type=int, default=200_000, help="0 skips the synthetic catalogue")
    parser.add_argument("--synthetic-queries", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.5, help="the synthetic vectors' noise (default 0.5)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n < 1 or args.threads < 1 or args.synthetic_items < 0 or args.synthetic_queries < 1:
        parser.error("--n, --threads and --synthetic-queries must be >= 1, --synthetic-items >= 0")
    source = args.source or Path("results") / args.dataset / "twotower.json"
    p = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    config = _reported_config(source, args.dataset, p.manifest)

    seq = tt.sequences(p, include_val=True)
    t0 = time.perf_counter()
    net, _ = tt.train(seq, tt.genre_matrix(p.genres)[0], config)
    train_s = time.perf_counter() - t0
    embeddings = tt.embed(net, seq, config.max_history)
    history = p.history(include_val=True)
    users, _ = fe.offline(p, include_val=True)
    exclude = [u.seen for u in users]
    exact: list[ev.IntArray] = []
    for lo in range(0, p.n_users, ev.BATCH):
        batch = np.arange(lo, min(lo + ev.BATCH, p.n_users), dtype=np.int64)
        top = ev.top_k(embeddings.score(batch), history[batch], args.n)
        exact.extend(row[row >= 0] for row in top)

    threads = rt.set_threads(args.threads)
    try:
        real = _bench(embeddings.items, REAL_INDEXES, embeddings.users, args.n, exact, exclude, p.test_items)
        synthetic: dict[str, Any] | None = None
        if args.synthetic_items:
            catalogue = synthetic_catalogue(embeddings.items, args.synthetic_items, args.noise, seed=0)
            queries = embeddings.users[: args.synthetic_queries]
            synthetic_exact: list[ev.IntArray] = []
            for lo in range(0, len(queries), 100):  # 100 x 200,000 float64 scores at a time: 160 MB
                scores = queries[lo : lo + 100].astype(np.float64) @ catalogue.T.astype(np.float64)
                synthetic_exact.extend(np.argsort(-row, kind="stable")[: args.n].astype(np.int64) for row in scores)
            synthetic = {
                "SYNTHETIC": (
                    f"{args.synthetic_items} vectors = real movie vectors + Gaussian noise ({args.noise}), "
                    "renormalised; real user vectors as queries; no exclusion"
                ),
                "items": args.synthetic_items,
                "queries": len(queries),
                "indexes": _bench(catalogue, SYNTHETIC_INDEXES, queries, args.n, synthetic_exact, None, None),
            }
    finally:
        rt.set_threads(threads)

    result = {
        "command": "python scripts/bench_retrieval.py " + shlex.join(argv),
        "dataset": args.dataset,
        "two-tower configuration": config.as_dict(),
        "train_s": train_s,
        "candidates": args.n,
        "faiss threads": args.threads,
        "exact": "float64 dot products, history excluded, ties to the smaller index (evaluate.top_k)",
        "real catalogue": {"items": p.n_items, "queries": p.n_users, "indexes": real},
        "synthetic catalogue": synthetic,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": rt.FAISS_VERSION,
            "machine": platform.machine(),
            "system": platform.system(),
            "cpus": os.cpu_count(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
