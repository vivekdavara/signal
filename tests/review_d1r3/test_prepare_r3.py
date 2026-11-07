"""Review round 3 (day 1): what round 2's manifest checks and _replace_dir cleanup still let through."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalrec import cli, prepare


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "60", "--items", "90"]) == 0
    return tmp_path


def exit_code(argv: list[str]) -> int:
    """main's return value, or argparse's exit status; any other exception is a traceback and fails the test."""
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


@pytest.mark.parametrize(
    ("field", "wrong"),
    [("users", float), ("items_without_training_positives", bool)],
    ids=["users-as-float", "zero-as-false"],
)
def test_load_refuses_counts_of_the_wrong_type(data_dir: Path, field: str, wrong: type) -> None:
    """Defect: load (prepare.py line 316) compares counts with ``!=``, and 6040.0 == 6040 and False == 0 in Python,
    the same trap _check_manifest_fields avoids for "format" with ``type(fmt) is not int``. A manifest whose counts
    say "users": 60.0 or "items_without_training_positives": false passes, and _baselines (cli.py line 55) copies
    manifest["counts"] verbatim into baselines.json.
    Input: a manifest written by save, one count rewritten as float(n) or bool(n).
    Expected: PrepareError (counts must be ints equal to the arrays'). Actual: load returns the data.
    """
    assert cli.main(["prepare", "synthetic", "--data-dir", str(data_dir)]) == 0
    processed = data_dir / "processed" / "synthetic"
    manifest = prepare.read_manifest(processed)
    if wrong is bool:
        assert manifest["counts"][field] in (0, 1)
    manifest["counts"][field] = wrong(manifest["counts"][field])
    (processed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(prepare.PrepareError, match="counts"):
        prepare.load(processed, "synthetic")


@pytest.mark.parametrize("kind", ["symlink-to-dir", "file"])
def test_prepare_over_a_symlinked_or_file_out_dir_is_a_one_line_error(
    data_dir: Path, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defect: round 2 noticed that a symlinked out_dir makes _replace_dir's first rename fail (prepare.py line 241:
    rename(2) of a non-directory onto mkdtemp's empty directory is EISDIR) and only removed the empty directory. The
    IsADirectoryError still escapes save, and cli.main catches only FileNotFoundError among OSErrors, so
    ``signalrec prepare`` over data/processed/<name> that is a symlink to a directory (data on another disk) or a
    stray file ends in a traceback, every time: such a directory can never be prepared.
    Input: processed/synthetic is a symlink to an empty directory, or a regular file; signalrec prepare synthetic.
    Expected: exit 0 (data written) or exit 1 with one "signalrec: " line. Actual: IsADirectoryError traceback.
    """
    processed = data_dir / "processed"
    processed.mkdir()
    if kind == "file":
        (processed / "synthetic").write_text("not a directory\n")
    else:
        (data_dir / "elsewhere").mkdir()
        (processed / "synthetic").symlink_to(data_dir / "elsewhere", target_is_directory=True)
    code = exit_code(["prepare", "synthetic", "--data-dir", str(data_dir)])
    assert code in (0, 1)
    if code == 1:
        assert capsys.readouterr().err.startswith("signalrec: ")
    assert sorted(p.name for p in processed.iterdir()) == ["synthetic"]
