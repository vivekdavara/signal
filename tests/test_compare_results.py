"""scripts/compare_results.py, CI's regression check."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_results.py"
sys.path.insert(0, str(SCRIPT.parent))
import compare_results  # noqa: E402


def result() -> dict[str, Any]:
    cell = {"value": 0.03, "ci95": [0.02, 0.04]}
    return {
        "dataset": "ml-latest-small",
        "users": 601,
        "items": 1955,
        "prepared": {"counts": {"users": 601}},
        "chosen": {"item-knn": {"k": 10, "shrink": 0.0}},
        "test": {"item-knn": {"ndcg@10": dict(cell), "fit_seconds": 1.0}, "popularity": {"ndcg@10": dict(cell)}},
    }


def test_identical_results_pass() -> None:
    assert compare_results.compare(result(), result()) == []


def test_timings_are_ignored_and_small_metric_moves_pass() -> None:
    fresh = copy.deepcopy(result())
    fresh["test"]["item-knn"]["fit_seconds"] = 99.0
    fresh["test"]["item-knn"]["ndcg@10"]["value"] += 0.004
    assert compare_results.compare(result(), fresh) == []


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("test", "item-knn", "ndcg@10", "value"), 0.04, "test item-knn ndcg@10: committed 0.0300, fresh 0.0400"),
        (("chosen", "item-knn", "k"), 25, "chosen:"),
        (("prepared", "counts", "users"), 600, "prepared:"),
        (("items",), 1954, "items:"),
    ],
)
def test_differences_are_reported(path: tuple[str, ...], value: object, message: str) -> None:
    fresh = copy.deepcopy(result())
    node: Any = fresh
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    problems = compare_results.compare(result(), fresh)
    assert len(problems) == 1
    assert problems[0].startswith(message)


def test_the_script_exits_1_on_a_difference(tmp_path: Path) -> None:
    fresh = result()
    fresh["users"] = 1
    (tmp_path / "a.json").write_text(json.dumps(result()))
    (tmp_path / "b.json").write_text(json.dumps(fresh))
    same = subprocess.run([sys.executable, SCRIPT, tmp_path / "a.json", tmp_path / "a.json"], capture_output=True)
    differ = subprocess.run([sys.executable, SCRIPT, tmp_path / "a.json", tmp_path / "b.json"], capture_output=True)
    assert (same.returncode, differ.returncode) == (0, 1)
    assert b"users: committed 601, fresh 1" in differ.stdout
