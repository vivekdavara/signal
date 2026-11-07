"""Review round 4 (day 1): what compare_leakage (scripts/compare_results.py, commit 92e0737) lets through."""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_results.py"
sys.path.insert(0, str(SCRIPT.parent))
compare_results = importlib.import_module("compare_results")


def leakage() -> dict[str, Any]:
    cell = {"global (leaky)": 0.02, "as of test time": 0.019, "difference": 0.001, "difference ci95": [-0.001, 0.003]}
    return {
        "dataset": "ml-latest-small",
        "users": 601,
        compare_results.SHARES: {"mean share": 0.52, "median share": 0.49},
        compare_results.LEAKAGE: {"ndcg@10": dict(cell)},
        "bootstrap": {"resamples": 1000, "seed": 20261107},
    }


@pytest.mark.parametrize("side", ["fresh", "committed"])
@pytest.mark.parametrize(
    "path",
    [
        (compare_results.LEAKAGE, "ndcg@10", "global (leaky)"),
        (compare_results.LEAKAGE, "ndcg@10", "difference"),
        (compare_results.SHARES, "mean share"),
    ],
)
def test_nan_is_reported(side: str, path: tuple[str, ...]) -> None:
    """Defect: compare_leakage tests ``abs(got - committed) > tolerance``, which is False when either value is NaN,
    so a leakage run that produced NaN (an empty bootstrap, a 0/0 share) passes CI's comparison.
    Input: an otherwise identical leakage.json with one value set to NaN, on either side.
    Expected: one problem naming the value. Actual: [] (the CI step passes).
    """
    committed, fresh = leakage(), leakage()
    node: Any = fresh if side == "fresh" else committed
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = float("nan")
    assert len(compare_results.compare(committed, fresh)) == 1


def test_bootstrap_parameters_and_intervals_are_compared() -> None:
    """Defect: compare_leakage compares neither the bootstrap block (resamples, seed) nor any "difference ci95",
    although they are the leakage report's other numbers; baselines compares its parameters ("chosen") exactly.
    Input: fresh run with a different bootstrap seed and resample count and an interval moved by 0.5.
    Expected: problems reported. Actual: [] (the CI step passes with the published intervals wrong).
    """
    fresh = copy.deepcopy(leakage())
    fresh["bootstrap"] = {"resamples": 10, "seed": 1}
    fresh[compare_results.LEAKAGE]["ndcg@10"]["difference ci95"] = [-0.5, 0.5]
    assert compare_results.compare(leakage(), fresh) != []


def test_a_metric_only_in_the_fresh_document_is_reported() -> None:
    """Defect: compare_leakage iterates the committed document's keys only, so a metric or share the fresh run has
    and the committed file lacks (or a committed file with an empty block) is never compared.
    Input: fresh has an extra metric "recall@100" and an extra share.
    Expected: a problem naming the extra keys. Actual: [].
    """
    fresh = copy.deepcopy(leakage())
    fresh[compare_results.LEAKAGE]["recall@100"] = dict(fresh[compare_results.LEAKAGE]["ndcg@10"])
    fresh[compare_results.SHARES]["users with any"] = 0.99
    assert compare_results.compare(leakage(), fresh) != []


@pytest.mark.parametrize(
    "drop",
    [
        ("users",),
        (compare_results.SHARES, "median share"),
        (compare_results.LEAKAGE, "ndcg@10"),
        (compare_results.LEAKAGE, "ndcg@10", "as of test time"),
        (compare_results.LEAKAGE,),
    ],
)
def test_a_key_missing_from_the_fresh_document_is_reported_not_raised(drop: tuple[str, ...]) -> None:
    """Defect: compare_leakage indexes the fresh document directly, so a key it lacks (including a baselines.json
    passed as the fresh leakage document, which has no leakage block) raises KeyError: main prints a traceback
    instead of the "N differences" report.
    Input: fresh with one key removed.
    Expected: compare returns a problem naming the key. Actual: KeyError.
    """
    fresh = copy.deepcopy(leakage())
    node: Any = fresh
    for key in drop[:-1]:
        node = node[key]
    del node[drop[-1]]
    assert compare_results.compare(leakage(), fresh) != []


def test_users_must_be_an_int() -> None:
    """Defect: "users" is compared with !=, so 601.0 or (for 1 user) true passes: the type trap round 3 closed for
    the manifest's counts. Input: fresh users 601.0. Expected: a problem. Actual: [].
    """
    fresh = copy.deepcopy(leakage())
    fresh["users"] = 601.0
    assert compare_results.compare(leakage(), fresh) != []
