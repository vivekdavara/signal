"""Compare a fresh results document (baselines.json or leakage.json) with the committed one: CI's regression check
on the real ml-latest-small.

    python scripts/compare_results.py results/ml-latest-small/baselines.json fresh.json
    python scripts/compare_results.py results/ml-latest-small/leakage.json fresh-leakage.json

The whole document is compared, recursively: both must have the same keys at every level (a key on either side only
is a difference), lists the same length, strings and booleans equal, integers equal and of type int (601.0 is not a
count), and every other number finite and within TOLERANCE. Only what is expected to differ between machines is
skipped: timings, the command line and the library versions.

Why a tolerance: item-kNN's scores are exact integers, but the bootstrap and the metrics are float means, and another
CPU or NumPy build may round a last bit differently. One user is 1/601 = 0.0017 of a metric on ml-latest-small, so
0.005 allows about three users' worth of movement and nothing more.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

TOLERANCE = 0.005
SKIPPED = frozenset({"command", "environment", "fit_seconds", "eval_seconds"})
# Two keys of a leakage.json, named for the tests that build one.
LEAKAGE = "popularity, global minus as of test time (paired)"
SHARES = "training positives later than the user's test item"


def _show(value: Any) -> str:
    return f"{value:.4f}" if type(value) is float else repr(value)


def compare(committed: Any, fresh: Any, tolerance: float = TOLERANCE, path: str = "") -> list[str]:
    """Every difference between two JSON values, each as "path: committed X, fresh Y"."""
    where = path or "(document)"
    if isinstance(committed, dict):
        if not isinstance(fresh, dict):
            return [f"{where}: committed an object, fresh {_show(fresh)}"]
        problems = []
        for key in sorted(set(committed) | set(fresh)):
            if key in SKIPPED:
                continue
            sub = f"{path}.{key}" if path else key
            if key not in fresh:
                problems.append(f"{sub}: missing from the fresh document")
            elif key not in committed:
                problems.append(f"{sub}: only in the fresh document")
            else:
                problems += compare(committed[key], fresh[key], tolerance, sub)
        return problems
    if isinstance(committed, list):
        if not isinstance(fresh, list) or len(fresh) != len(committed):
            return [f"{where}: committed {_show(committed)}, fresh {_show(fresh)}"]
        problems = []
        for i, (a, b) in enumerate(zip(committed, fresh, strict=True)):
            problems += compare(a, b, tolerance, f"{path}[{i}]")
        return problems
    if type(committed) is float:
        if type(fresh) not in (int, float) or not (math.isfinite(committed) and math.isfinite(fresh)):
            return [f"{where}: committed {_show(committed)}, fresh {_show(fresh)}"]
        return [] if abs(fresh - committed) <= tolerance else [f"{where}: committed {committed:.4f}, fresh {fresh:.4f}"]
    # int (not bool: type() is exact), bool, str, None: equal and of the same type.
    if type(fresh) is not type(committed) or fresh != committed:
        return [f"{where}: committed {_show(committed)}, fresh {_show(fresh)}"]
    return []


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
