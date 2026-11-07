"""Compare a fresh baselines.json or leakage.json with the committed one: CI's regression check on the real
ml-latest-small.

The processed data must match exactly (counts, parameters, source hashes) and so must item-kNN's chosen parameters.
Each test metric must be within TOLERANCE: item-kNN sums float32 similarities, and a different CPU or SciPy build can
round a near-tie the other way, which moves one user's rank. One user is 1/601 = 0.0017 of a metric on
ml-latest-small, so the tolerance allows about three such users.

    python scripts/compare_results.py results/ml-latest-small/baselines.json fresh.json
    python scripts/compare_results.py results/ml-latest-small/leakage.json fresh-leakage.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

TOLERANCE = 0.005


LEAKAGE = "popularity, global minus as of test time (paired)"
SHARES = "training positives later than the user's test item"


def compare_leakage(committed: dict[str, Any], fresh: dict[str, Any], tolerance: float = TOLERANCE) -> list[str]:
    """A leakage.json: the users exactly; the shares and each metric's three values within the tolerance."""
    problems = []
    for key in ("dataset", "users"):
        if committed[key] != fresh[key]:
            problems.append(f"{key}: committed {committed[key]!r}, fresh {fresh[key]!r}")
    for name, value in committed[SHARES].items():
        if abs(fresh[SHARES][name] - value) > tolerance:
            problems.append(f"{name}: committed {value:.4f}, fresh {fresh[SHARES][name]:.4f}")
    for metric, cell in committed[LEAKAGE].items():
        for name in ("global (leaky)", "as of test time", "difference"):
            got = fresh[LEAKAGE][metric][name]
            if abs(got - cell[name]) > tolerance:
                problems.append(f"{metric} {name}: committed {cell[name]:.4f}, fresh {got:.4f}")
    return problems


def compare(committed: dict[str, Any], fresh: dict[str, Any], tolerance: float = TOLERANCE) -> list[str]:
    """A baselines.json, or a leakage.json (told apart by their keys)."""
    if LEAKAGE in committed:
        return compare_leakage(committed, fresh, tolerance)
    problems = []
    for key in ("dataset", "users", "items", "prepared", "chosen"):
        if committed[key] != fresh[key]:
            problems.append(f"{key}: committed {committed[key]!r}, fresh {fresh[key]!r}")
    for model, metrics in committed["test"].items():
        for metric, cell in metrics.items():
            if not isinstance(cell, dict):
                continue  # timings
            got = fresh["test"][model][metric]["value"]
            if abs(got - cell["value"]) > tolerance:
                problems.append(f"test {model} {metric}: committed {cell['value']:.4f}, fresh {got:.4f}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    committed, fresh = (json.loads(Path(p).read_text()) for p in argv)
    problems = compare(committed, fresh)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} differences beyond the tolerance ({TOLERANCE})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
