"""scripts/same_second.py on the synthetic dataset: the groups partition the users and recompute the reported model."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from signalrec import cli, prepare

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "same_second.py"
sys.path.insert(0, str(SCRIPT.parent))
import same_second  # noqa: E402


def test_groups_partition_the_users_and_all_is_the_reported_number(tmp_path: Path) -> None:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "80", "--items", "120"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    base = ["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", "0.1:on", "--epochs", "2", "--seeds", "0"]
    base += ["--knn-grid", "5:0", "--quiet"]
    day3, day2 = tmp_path / "day3.json", tmp_path / "day2.json"
    assert cli.main([*base, "--histories", "3", "--out", str(day3)]) == 0
    assert cli.main([*base, "--histories", "50", "--out", str(day2)]) == 0
    out = tmp_path / "same-second.json"
    argv = ["synthetic", "--data-dir", str(tmp_path), "--day3", str(day3), "--day2", str(day2), "--out", str(out)]
    assert same_second.main(argv) == 0
    result = json.loads(out.read_text())
    groups = result["groups"]
    p = prepare.load(tmp_path / "processed" / "synthetic", "synthetic")
    tied = int((p.test_times == p.val_times).sum())
    assert groups["tied"]["users"] == tied
    assert 0 < tied < p.n_users  # both groups occur (8 of 80 here)
    assert groups["tied"]["users"] + groups["untied"]["users"] == groups["all"]["users"] == p.n_users
    assert result["share of users tied"] == tied / p.n_users
    # "all" is the protocol's test number for each model, as the two-tower command reported it.
    reported3 = json.loads(day3.read_text())["test"]
    reported2 = json.loads(day2.read_text())["test"]
    cells = groups["all"]["ndcg@10"]
    assert cells["two-tower (day 3)"]["value"] == reported3["two-tower"]["ndcg@10"]["value"]
    assert cells["two-tower (day 2)"]["value"] == reported2["two-tower"]["ndcg@10"]["value"]
    assert cells["item-knn"]["value"] == reported3["item-knn"]["ndcg@10"]["value"]
    assert cells["popularity"]["value"] == reported3["popularity"]["ndcg@10"]["value"]
    # The overall mean is the users-weighted mean of the two groups.
    for name, cell in cells.items():
        parts = [
            groups[g]["ndcg@10"][name]["value"] * groups[g]["users"] for g in ("tied", "untied") if groups[g]["users"]
        ]
        assert np.isclose(sum(parts) / p.n_users, cell["value"])


def test_an_empty_group_is_reported_as_no_users_not_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Review round 2 (day 3): with no tied users the group's mean was NaN, which json.dumps writes as NaN, not JSON.
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "80", "--items", "120"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    base = ["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", "0.1:on", "--epochs", "1", "--seeds", "0"]
    base += ["--knn-grid", "5:0", "--quiet", "--histories", "3"]
    day3 = tmp_path / "day3.json"
    assert cli.main([*base, "--out", str(day3)]) == 0
    real_load = prepare.load

    def untied(*args: object, **kwargs: object) -> prepare.Prepared:
        p = real_load(*args, **kwargs)  # type: ignore[arg-type]
        return replace(p, test_times=p.val_times + 1)  # every test movie a second after the validation one

    monkeypatch.setattr(prepare, "load", untied)
    out = tmp_path / "same-second.json"
    argv = ["synthetic", "--data-dir", str(tmp_path), "--day3", str(day3), "--day2", str(day3), "--out", str(out)]
    assert same_second.main(argv) == 0
    result = json.loads(out.read_text(), parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert result["groups"]["tied"] == {"users": 0, "ndcg@10": None, "paired differences, ndcg@10": None}
    assert result["groups"]["untied"]["users"] == result["groups"]["all"]["users"]
