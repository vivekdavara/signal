"""scripts/ab_simulate.py's analysis on hand-made ranks (no Redis, no model): the edge cases a small dataset hits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ab_simulate

from signalrec import abtest

USERS = np.arange(1, 401, dtype=np.int64)


def test_a_real_difference_gets_a_sample_size_and_its_check() -> None:
    rng = np.random.default_rng(0)
    control = rng.integers(0, 15, size=len(USERS))
    treatment = np.where(control > 1, control - 1, control)  # the next movie one place higher where it can be
    row = ab_simulate.analyse(abtest.ClickModel(), control, treatment, USERS, 30)
    assert row["true difference"] > 0
    assert row["users per arm for 80% power at alpha 0.05"] > 0
    assert set(row["checks"]) == {"A/A at the dataset's size", "A/B at the dataset's size", "A/B at the computed size"}


def test_identical_arms_have_no_sample_size_instead_of_crashing() -> None:
    ranks = np.random.default_rng(1).integers(0, 15, size=len(USERS))
    row = ab_simulate.analyse(abtest.ClickModel(), ranks, ranks.copy(), USERS, 30)
    assert row["true difference"] == 0
    assert row["users per arm for 80% power at alpha 0.05"] is None
    assert "A/B at the computed size" not in row["checks"]


def test_nobody_converting_has_no_relative_lift() -> None:
    none_shown = np.zeros(len(USERS), dtype=np.int64)
    row = ab_simulate.analyse(abtest.ClickModel(p_other=0.0), none_shown, none_shown, USERS, 10)
    assert row["true conversion"] == {"control": 0.0, "treatment": 0.0}
    assert row["true relative lift"] is None
