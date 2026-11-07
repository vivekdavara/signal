"""The command line end to end on the synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalrec import cli, prepare


def test_synthetic_then_prepare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "80", "--items", "120"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "synthetic ratings (80 users, 120 movies, seed 7)" in out
    manifest = prepare.read_manifest(tmp_path / "processed" / "synthetic")
    assert manifest["dataset"] == "synthetic"
    assert set(manifest["sources"]) == {"ratings.csv", "movies.csv"}
    assert f"users {manifest['counts']['users']}" in out
    prepare.load(tmp_path / "processed" / "synthetic")


def test_errors_are_one_line_and_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["prepare", "ml-1m", "--data-dir", str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("signalrec: ")
    assert cli.main(["synthetic", "--data-dir", str(tmp_path)]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path), "--min-user-positives", "2"]) == 1
    assert "at least 3" in capsys.readouterr().err


def test_the_leakage_command_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "60", "--items", "90"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    out = tmp_path / "leakage.json"
    assert cli.main(["leakage", "synthetic", "--data-dir", str(tmp_path), "--out", str(out)]) == 0
    assert "popularity ndcg@10: global " in capsys.readouterr().out
    assert json.loads(out.read_text())["dataset"] == "synthetic"


@pytest.mark.parametrize("grid", ["10:nan", "10:inf", "10:-0.5", "1.5:0", ":3", "10:"])
def test_bad_grid_values_are_usage_errors(grid: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["baselines", "synthetic", "--data-dir", str(tmp_path), "--grid", grid])
    assert exc.value.code == 2
    assert "expected k:shrink" in capsys.readouterr().err
