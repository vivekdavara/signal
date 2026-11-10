"""Does the request path reproduce the offline test numbers? (DESIGN.md, "Day 4: serving and experiments")

Every user is served through ``serving.Recommender`` (Redis features, the NumPy user tower, FAISS, the ranker) with
the full candidate list returned (k = the candidate count), and the rank of their held-out test movie is read off it
(candidate count + 1 when it wasn't retrieved), as ``experiment_ranking`` ranks offline. Three runs:

- **protocol filter, ranked**: only the user's positives excluded, as offline. Its metrics must equal
  ``ranking.json``'s re-ranked test metrics; a difference means serving computes something else than what was
  evaluated (training-serving skew), and the script exits 1.
- **protocol filter, retrieved order** (the ranker skipped): must equal ``ranking.json``'s retrieved order.
- **serving filter** (every rated movie excluded), ranked: what the filter does to the metric. Reported, not compared.

Needs the model (``signalrec export``) and the store (``signalrec load-store``) for the dataset.

    python scripts/serving_check.py ml-1m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from signalrec import evaluate as ev
from signalrec import experiment_ranking as er
from signalrec import prepare
from signalrec import tower as tw
from signalrec.featurestore import FeatureStore
from signalrec.serving import Recommender

TOLERANCE = 1e-12  # the means of the same per-user values; any real difference is at least 1 / users


def ranks(rec: Recommender, test_items: ev.IntArray, n: int, exclude: str, rank: bool) -> ev.IntArray:
    out = np.full(len(test_items), n + 1, dtype=np.int64)
    for u, target in enumerate(test_items.tolist()):
        movies = rec.recommend(u, k=n, exclude=exclude, rank=rank).movies
        hit = np.flatnonzero(movies == target)
        if len(hit):
            out[u] = int(hit[0]) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, help="default: models/<dataset>")
    parser.add_argument("--ranking", type=Path, help="default: results/<dataset>/ranking.json")
    parser.add_argument("--redis", default="redis://127.0.0.1:6383/0")
    parser.add_argument("--namespace", default="signal")
    parser.add_argument("--out", type=Path, help="default: results/<dataset>/serving-check.json")
    args = parser.parse_args(argv)

    model = tw.load(args.model or Path("models") / args.dataset)
    prepared = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    ranking_path = args.ranking or Path("results") / args.dataset / "ranking.json"
    offline = json.loads(ranking_path.read_text())
    n = int(offline["candidates"])
    store = FeatureStore.connect(model.catalogue, args.redis, args.namespace)
    rec = Recommender(model, store, n_candidates=n)

    runs: dict[str, Any] = {}
    per_user_runs: dict[str, dict[str, ev.FloatArray]] = {}
    failures = []
    for name, exclude, rank, want in (
        ("protocol filter, ranked", "liked", True, offline["test"]["re-ranked"]),
        ("protocol filter, retrieved order", "liked", False, offline["test"]["retrieved order"]),
        ("serving filter (every rated movie), ranked", "rated", True, None),
    ):
        t0 = time.perf_counter()
        r = ranks(rec, prepared.test_items, n, exclude, rank)
        seconds = time.perf_counter() - t0
        per_user = er.metrics(r, n)
        per_user_runs[name] = per_user
        got = {metric: float(values.mean()) for metric, values in per_user.items()}
        row: dict[str, Any] = {"metrics": got, "seconds": seconds, "users": len(r)}
        if want is not None:
            diff = {metric: got[metric] - want[metric]["value"] for metric in got}
            row["minus ranking.json"] = diff
            row["equal"] = all(abs(d) <= TOLERANCE for d in diff.values())
            if not row["equal"]:
                failures.append(name)
        runs[name] = row
        print(f"{name}: " + ", ".join(f"{m} {v:.4f}" for m, v in got.items()) + f" ({seconds:.1f} s)")
        if "equal" in row:
            print(f"  equal to {ranking_path}: {row['equal']}")

    protocol = per_user_runs["protocol filter, ranked"]
    serving = per_user_runs["serving filter (every rated movie), ranked"]
    filter_effect = {
        m: {
            "value": float(serving[m].mean() - protocol[m].mean()),
            "ci95": list(ev.paired_bootstrap_ci(serving[m], protocol[m], seed=er.SEED)),
        }
        for m in protocol
    }
    ndcg = filter_effect[er.PRIMARY]
    print(f"serving filter minus protocol filter, {er.PRIMARY}: {ndcg['value']:+.4f} {ndcg['ci95']}")
    result = {
        "command": "python " + " ".join(["scripts/serving_check.py", *(sys.argv[1:] if argv is None else argv)]),
        "dataset": args.dataset,
        "model": str(args.model or Path("models") / args.dataset),
        "compared with": str(ranking_path),
        "candidates": n,
        "runs": runs,
        "serving filter minus protocol filter (paired, test)": filter_effect,
        "bootstrap": {"resamples": 1000, "seed": er.SEED},
    }
    out = args.out or Path("results") / args.dataset / "serving-check.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    if failures:
        print(f"MISMATCH with the offline numbers: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
