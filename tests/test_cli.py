"""The command line end to end on the synthetic dataset."""

from __future__ import annotations

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
