"""Review round 3 (day 1): baselines reads the manifest twice, and only the first read is checked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalrec import cli, prepare


def test_baselines_reports_the_manifest_it_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: round 2 made _baselines (cli.py lines 49-50) call prepare.load(processed, dataset), which checks the
    manifest, and then read the manifest a second time with read_manifest, unchecked, for the "prepared" block of
    baselines.json. A ``signalrec prepare`` that replaces the directory between the two reads (here: run from inside
    load, after it returns) makes baselines.json report the new run's params and counts beside metrics computed on
    the old arrays.
    Input: data prepared with --min-rating 4; a prepare with --min-rating 1 lands after load returns.
    Expected: "prepared" describes the arrays evaluated (min_rating 4.0). Actual: min_rating 1.0 and the new counts.
    """
    data = str(tmp_path)
    assert cli.main(["synthetic", "--data-dir", data, "--users", "60", "--items", "90"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", data, "--min-rating", "4"]) == 0
    processed = tmp_path / "processed" / "synthetic"
    before = prepare.read_manifest(processed)
    real_load = prepare.load

    def load_then_concurrent_prepare(directory: Path, dataset: str | None = None) -> prepare.Prepared:
        loaded = real_load(directory, dataset)
        assert cli.main(["prepare", "synthetic", "--data-dir", data, "--min-rating", "1"]) == 0
        return loaded

    monkeypatch.setattr(prepare, "load", load_then_concurrent_prepare)
    out = tmp_path / "baselines.json"
    assert cli.main(["baselines", "synthetic", "--data-dir", data, "--grid", "3:0", "--out", str(out)]) == 0
    reported = json.loads(out.read_text())["prepared"]
    assert reported["params"] == before["params"]
    assert reported["counts"] == before["counts"]
