"""Review round 2 (day 1): command-line inputs and manifest fields round 1 left unchecked."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from signalrec import cli, prepare


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = tmp_path_factory.mktemp("data")
    assert cli.main(["synthetic", "--data-dir", str(data), "--users", "60", "--items", "90"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(data)]) == 0
    return data


def exit_code(argv: list[str]) -> int:
    """main's return value, or argparse's exit status; any other exception is a traceback and fails the test."""
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


@pytest.mark.parametrize(
    "bad",
    [["--neighbours", "0"], ["--shrink", "-1"], ["--shrink", "nan"], ["--k", "-1"], ["--k", "0"]],
    ids=["neighbours-0", "shrink-negative", "shrink-nan", "k-negative", "k-0"],
)
def test_recommend_refuses_bad_model_parameters_in_one_line(
    data_dir: Path, bad: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Defect: cli's recommend parser (lines 157-159) takes --neighbours, --shrink and --k as bare int/float, unlike
    --grid (round 1 moved its checks into argparse), and _recommend (line 93) catches only KeyError.
    --neighbours 0 and --shrink -1 raise ValueError from baselines.neighbours: a traceback, not "signalrec: ...".
    --shrink nan runs with an empty similarity matrix (all scores 0: the catalogue in index order), --k -1 lists
    every movie but one (evaluate.top_k slices ``[:, :-1]``) and --k 0 lists none, all with exit 0.

    Input: ``signalrec recommend synthetic --user <first prepared user> <bad>`` on a 60-user synthetic dataset.
    Expected: exit 1 or 2 with a one-line message. Actual: ValueError traceback, or exit 0 with a wrong list.
    """
    user = int(prepare.load(data_dir / "processed" / "synthetic").user_raw_ids[0])
    argv = ["recommend", "synthetic", "--data-dir", str(data_dir), "--user", str(user), *bad]
    capsys.readouterr()
    assert exit_code(argv) in (1, 2)
    assert "Traceback" not in capsys.readouterr().err


def processed_copy(data_dir: Path, tmp_path: Path, name: str) -> Path:
    """A copy of the prepared synthetic directory under ``tmp_path/data/processed/<name>``; returns the data dir."""
    data = tmp_path / "data"
    shutil.copytree(data_dir / "processed" / "synthetic", data / "processed" / name)
    return data


def test_baselines_with_a_manifest_missing_params_is_a_one_line_error(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defect: round 1 made a missing, unreadable or non-object manifest a PrepareError, and load checks "format"
    and "files", but cli._baselines (line 55) then indexes manifest["params"], ["counts"] and ["sources"], which
    nothing checks: a manifest without one of them ends in a KeyError traceback.

    Input: a prepared directory whose manifest.json has its "params" key removed (manifest.json isn't hashed).
    Expected: exit 1, "signalrec: ..." naming the manifest. Actual: KeyError: 'params'.
    """
    data = processed_copy(data_dir, tmp_path, "synthetic")
    path = data / "processed" / "synthetic" / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["params"]
    path.write_text(json.dumps(manifest))
    argv = ["baselines", "synthetic", "--data-dir", str(data), "--grid", "1:0", "--out", str(tmp_path / "r.json")]
    assert exit_code(argv) == 1
    assert capsys.readouterr().err.startswith("signalrec: ")


def test_a_processed_directory_of_another_dataset_is_refused(data_dir: Path, tmp_path: Path) -> None:
    """Defect: the manifest records which dataset it was built from ("dataset"), but neither prepare.load nor the
    CLI compares it with the dataset asked for, so processed data in the wrong directory is evaluated and the result
    is labelled with the wrong dataset (the hashes all match: they cover the arrays, not what they are).

    Input: data/processed/synthetic copied to data/processed/ml-latest-small; ``signalrec baselines
    ml-latest-small --grid 1:0``.
    Expected: exit 1 (manifest says "synthetic"). Actual: exit 0 and a results file whose "dataset" is
    "ml-latest-small" while its "sources" are the synthetic files.
    """
    data = processed_copy(data_dir, tmp_path, "ml-latest-small")
    out = tmp_path / "r.json"
    argv = ["baselines", "ml-latest-small", "--data-dir", str(data), "--grid", "1:0", "--out", str(out)]
    assert exit_code(argv) == 1, json.loads(out.read_text())["dataset"]


def test_manifest_counts_that_disagree_with_the_arrays_are_refused(data_dir: Path, tmp_path: Path) -> None:
    """Defect: manifest.json's "counts" (and "params") aren't covered by any hash or check; prepare.load verifies
    only the files, and cli._baselines copies the manifest's counts into the results as "prepared". A manifest whose
    file hashes all match but whose counts are wrong loads, and the result reports counts that contradict its own
    "users".

    Input: the prepared synthetic directory with manifest counts.users set to 999 (60 users really).
    Expected: exit 1 (counts disagree with the arrays). Actual: exit 0; result["prepared"]["counts"]["users"] is
    999 while result["users"] is 60.
    """
    data = processed_copy(data_dir, tmp_path, "synthetic")
    path = data / "processed" / "synthetic" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["users"] = 999
    path.write_text(json.dumps(manifest))
    out = tmp_path / "r.json"
    argv = ["baselines", "synthetic", "--data-dir", str(data), "--grid", "1:0", "--out", str(out)]
    code = exit_code(argv)
    if code == 0:
        result = json.loads(out.read_text())
        assert result["prepared"]["counts"]["users"] == result["users"]
    assert code == 1
