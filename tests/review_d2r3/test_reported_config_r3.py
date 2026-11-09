"""Review D2 round 3: ``cli._reported_config`` after round 2's fixed-settings and prepared-data rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from signalrec import cli, prepare

FIXED = {"dim": 16, "hidden": 16, "batch_size": 64, "lr": 0.01, "max_history": 50}


def _results(**fixed: Any) -> dict[str, Any]:
    return {
        "dataset": "synthetic",
        "chosen": {"two-tower": {"temperature": 0.1, "logq": True, "epochs": 2}},
        "fixed": {**FIXED, **fixed},
        "test": {"two-tower": {"seed": 3}},
    }


@pytest.mark.parametrize("where", ["fixed lr", "chosen temperature"])
def test_a_huge_integer_setting_is_a_prepare_error_not_a_traceback(tmp_path: Path, where: str) -> None:
    """A results file whose lr or temperature is an integer too large for a float makes ``Config.check`` raise
    OverflowError, which ``_reported_config`` doesn't catch, so leakage-twotower and the recency ablation crash with a
    traceback instead of "not a two-tower results file"."""
    result = _results()
    if where == "fixed lr":
        result["fixed"]["lr"] = 10**400
    else:
        result["chosen"]["two-tower"]["temperature"] = 10**400
    path = tmp_path / "twotower.json"
    path.write_text(json.dumps(result))
    with pytest.raises(prepare.PrepareError):
        cli._reported_config(path, "synthetic")
