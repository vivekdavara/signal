"""Review D2: where the two-tower code disagrees with DESIGN.md, "Day 2: two-tower retriever"."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from signalrec import experiment_twotower as ext
from signalrec import movielens, prepare, synthetic
from signalrec import twotower as tt
from signalrec.prepare import Prepared

GRID = ((0.1, True), (0.5, False))
BASE = tt.Config(dim=8, hidden=8, batch_size=64, lr=1e-2)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    raw = synthetic.write(tmp_path_factory.mktemp("raw") / "synthetic", n_users=60, n_items=80)
    return prepare.prepare(movielens.load("synthetic", raw), prepare.Params())


@pytest.fixture(scope="module")
def result(prepared: Prepared) -> dict[str, Any]:
    return ext.run(prepared, GRID, 3, (0,), ((5, 0.0),), BASE)


def test_every_training_positive_but_each_users_first_is_an_example(prepared: Prepared) -> None:
    """DESIGN said "for each training positive", but ``example_positions`` drops each user's first positive (it has no
    earlier positive to form a history). Resolved in DESIGN, not the code: the first positive is never a target, and
    DESIGN now says so. The reviewer's test asserted the old wording; this one pins the documented rule."""
    seq = tt.sequences(prepared, include_val=False)
    positions, _ = tt.example_positions(seq)
    users_with_positives = int((np.diff(seq.starts) > 0).sum())
    assert len(positions) == len(prepared.train_items) - users_with_positives


def test_every_candidate_gets_a_paired_interval_against_the_chosen_one(result: dict[str, Any]) -> None:
    """DESIGN asks for the paired validation interval of every candidate against the best, and the module defines the
    candidates as (configuration, epoch) pairs, but only each configuration's best epoch gets one."""
    for row in result["validation"]["two-tower"]:
        for point in row["curve"]:
            assert ext.DIAGNOSTIC in point, (row["temperature"], row["logq"], point["epoch"])
