"""How much of a model's test score comes from users who rated their test movie in the same second as the one before.

    .venv/bin/python scripts/same_second.py ml-1m --out results/ml-1m/same-second.json

The split orders each user's positives by (timestamp, movie id) (DESIGN.md, "Split"), and ml-1m users often rated
dozens of movies in one second. When the test movie shares its second with the validation movie, which of them is
"next" was decided by movie id, not by the user. A model that reads only the last few positives can learn from that
order (ids loosely follow when movies were added) rather than from taste. Post hoc, on test, nothing chosen from it:
users are split into **tied** (test second == validation second) and **untied**, and each model's NDCG@10 and the
paired differences are reported per group:

- the day-3 two-tower model (``twotower.json``'s reported configuration, retrained on training + validation),
- the day-2 one (``twotower-day2.json``: the same protocol with 50-item histories),
- item-kNN (the validation choice recorded in ``twotower.json``) and popularity, refit on training + validation.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

from signalrec import evaluate as ev
from signalrec import prepare
from signalrec import twotower as tt
from signalrec.baselines import ItemKNN, Popularity
from signalrec.cli import _reported_config

SEED = 20261107  # experiment.SEED


def _ndcg(ranks: ev.IntArray) -> ev.FloatArray:
    return ev.ndcg_at(ranks, 10)


def _cell(values: ev.FloatArray) -> dict[str, Any]:
    return {"value": float(values.mean()), "ci95": list(ev.bootstrap_ci(values, seed=SEED))}


def _diff(a: ev.FloatArray, b: ev.FloatArray) -> dict[str, Any]:
    return {"value": float(a.mean() - b.mean()), "ci95": list(ev.paired_bootstrap_ci(a, b, seed=SEED))}


def two_tower_ranks(p: prepare.Prepared, config: tt.Config) -> ev.IntArray:
    seq = tt.sequences(p, include_val=True)
    net, _ = tt.train(seq, tt.genre_matrix(p.genres)[0], config)
    history = p.history(include_val=True)
    return ev.evaluate(tt.embed(net, seq, config.max_history), history, p.test_items, "test").ranks


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--day3", type=Path, help="default: results/<dataset>/twotower.json")
    parser.add_argument("--day2", type=Path, help="default: results/<dataset>/twotower-day2.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    day3_source = args.day3 or Path("results") / args.dataset / "twotower.json"
    day2_source = args.day2 or Path("results") / args.dataset / "twotower-day2.json"
    p = prepare.load(args.data_dir / "processed" / args.dataset, args.dataset)
    day3 = _reported_config(day3_source, args.dataset, p.manifest)
    day2 = _reported_config(day2_source, args.dataset, p.manifest)
    knn_params = json.loads(day3_source.read_text())["test"]["item-knn"]

    history = p.history(include_val=True)
    knn = ItemKNN(k=knn_params["k"], shrink=float(knn_params["shrink"]))
    knn.fit(history)
    popularity = Popularity()
    popularity.fit(history)
    ndcg = {
        "two-tower (day 3)": _ndcg(two_tower_ranks(p, day3)),
        "two-tower (day 2)": _ndcg(two_tower_ranks(p, day2)),
        "item-knn": _ndcg(ev.evaluate(knn, history, p.test_items, "test").ranks),
        "popularity": _ndcg(ev.evaluate(popularity, history, p.test_items, "test").ranks),
    }
    tied = p.test_times == p.val_times
    groups = {"all": np.ones(p.n_users, dtype=bool), "tied": tied, "untied": ~tied}
    pairs = [
        ("two-tower (day 3)", "two-tower (day 2)"),
        ("two-tower (day 3)", "item-knn"),
        ("two-tower (day 2)", "item-knn"),
    ]
    by_group: dict[str, Any] = {}
    for group, members in groups.items():
        if not members.any():  # no tied (or no untied) users: no mean to report, and json would write NaN
            by_group[group] = {"users": 0, "ndcg@10": None, "paired differences, ndcg@10": None}
            continue
        by_group[group] = {
            "users": int(members.sum()),
            "ndcg@10": {name: _cell(values[members]) for name, values in ndcg.items()},
            "paired differences, ndcg@10": {
                f"{a} minus {b}": _diff(ndcg[a][members], ndcg[b][members]) for a, b in pairs
            },
        }
    result = {
        "command": "python scripts/same_second.py " + shlex.join(argv),
        "dataset": args.dataset,
        "split": "test; post hoc, nothing chosen from it",
        "tied": "the test movie's timestamp equals the validation movie's (the order between them is by movie id)",
        "share of users tied": float(tied.mean()),
        "configurations": {"two-tower (day 3)": day3.as_dict(), "two-tower (day 2)": day2.as_dict()},
        "item-knn": {"k": knn.k, "shrink": knn.shrink},
        "groups": by_group,
        "bootstrap": {"resamples": 1000, "seed": SEED},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for group, block in by_group.items():
        if block["ndcg@10"] is None:
            print(f"{group}: no users")
            continue
        cells = ", ".join(f"{name} {cell['value']:.4f}" for name, cell in block["ndcg@10"].items())
        print(f"{group} ({block['users']} users): {cells}")
        for label, cell in block["paired differences, ndcg@10"].items():
            low, high = cell["ci95"]
            print(f"  {label}: {cell['value']:+.4f} [{low:+.4f}, {high:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
