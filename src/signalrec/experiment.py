"""Run the baselines by the protocol: choose on validation, refit on train + validation, report on test.

Item-kNN's (k, shrink) is chosen by validation NDCG@10 over a grid; popularity has nothing to choose. Both are then
refit on the training + validation history and evaluated on the test items, with 95% bootstrap intervals over users
and a paired interval for item-kNN minus popularity. The result is one JSON document (results/<dataset>/baselines.json).
"""

from __future__ import annotations

import platform
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy

from signalrec import evaluate as ev
from signalrec.baselines import ItemKNN, Popularity
from signalrec.prepare import Prepared

PRIMARY = "ndcg@10"
DEFAULT_GRID: tuple[tuple[int, float], ...] = tuple(
    (k, shrink) for k in (1, 2, 3, 5, 10, 25, 50, 100, 200, 400) for shrink in (0.0, 10.0, 50.0)
)
SEED = 20261107


@dataclass(frozen=True)
class TestResult:
    evaluation: ev.Evaluation
    fit_seconds: float
    eval_seconds: float


def _timed_eval(model: Popularity | ItemKNN, p: Prepared, split: str) -> TestResult:
    include_val = split == "test"
    history = p.history(include_val=include_val)
    targets = p.test_items if include_val else p.val_items
    t0 = time.perf_counter()
    model.fit(history)
    t1 = time.perf_counter()
    evaluation = ev.evaluate(model, history, targets, split)
    t2 = time.perf_counter()
    return TestResult(evaluation, t1 - t0, t2 - t1)


def _with_intervals(e: ev.Evaluation) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in e.summary().items():
        out[name] = {"value": value}
        if name.startswith(("recall@", "ndcg@")):
            k = int(name.split("@")[1])
            per_user = ev.recall_at(e.ranks, k) if name.startswith("recall") else ev.ndcg_at(e.ranks, k)
            low, high = ev.bootstrap_ci(per_user, seed=SEED)
            out[name]["ci95"] = [low, high]
    return out


def run(p: Prepared, grid: Iterable[tuple[int, float]] = DEFAULT_GRID) -> dict[str, Any]:
    tuning = []
    for k, shrink in grid:
        result = _timed_eval(ItemKNN(k=k, shrink=shrink), p, "val")
        summary = result.evaluation.summary()
        tuning.append({"k": k, "shrink": shrink, **summary, "fit_seconds": result.fit_seconds})
    # Ties on the primary metric go to the first grid entry (smaller k, then smaller shrink).
    best = max(tuning, key=lambda row: row[PRIMARY])
    popularity_val = _timed_eval(Popularity(), p, "val").evaluation.summary()

    popularity = _timed_eval(Popularity(), p, "test")
    knn = _timed_eval(ItemKNN(k=int(best["k"]), shrink=float(best["shrink"])), p, "test")
    differences = {}
    for k in ev.KS:
        for metric, fn in (("recall", ev.recall_at), ("ndcg", ev.ndcg_at)):
            a = fn(knn.evaluation.ranks, k)
            b = fn(popularity.evaluation.ranks, k)
            low, high = ev.paired_bootstrap_ci(a, b, seed=SEED)
            differences[f"{metric}@{k}"] = {"value": float(a.mean() - b.mean()), "ci95": [low, high]}
    return {
        "protocol": "leave-last-out; tuned on validation by " + PRIMARY + "; refit on train+validation; test",
        "users": p.n_users,
        "items": p.n_items,
        "validation": {"popularity": popularity_val, "item-knn": tuning},
        "chosen": {"item-knn": {"k": best["k"], "shrink": best["shrink"]}},
        "test": {
            "popularity": {
                **_with_intervals(popularity.evaluation),
                "fit_seconds": popularity.fit_seconds,
                "eval_seconds": popularity.eval_seconds,
            },
            "item-knn": {
                **_with_intervals(knn.evaluation),
                "fit_seconds": knn.fit_seconds,
                "eval_seconds": knn.eval_seconds,
            },
        },
        "item-knn minus popularity (paired)": differences,
        "bootstrap": {"resamples": 1000, "seed": SEED},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }


def table(result: dict[str, Any]) -> str:
    """The test results as a Markdown table."""
    lines = [
        "| Model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Coverage@10 |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("popularity", "item-knn"):
        r = result["test"][name]

        def cell(metric: str, r: dict[str, Any] = r) -> str:
            value = r[metric]["value"]
            if "ci95" not in r[metric]:
                return f"{value:.3f}"
            low, high = r[metric]["ci95"]
            return f"{value:.4f} [{low:.4f}, {high:.4f}]"

        label = name
        if name == "item-knn":
            chosen = result["chosen"]["item-knn"]
            label = f"item-kNN (k={chosen['k']}, shrink={chosen['shrink']:g})"
        lines.append(
            f"| {label} | {cell('recall@10')} | {cell('ndcg@10')} | {cell('recall@50')} | {cell('ndcg@50')} "
            f"| {cell('coverage@10')} |"
        )
    return "\n".join(lines)
