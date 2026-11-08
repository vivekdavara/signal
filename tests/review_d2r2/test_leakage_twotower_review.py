"""Review D2 round 2: ``signalrec leakage-twotower`` reads results files that don't describe the reported run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from signalrec import cli, prepare
from signalrec import leakage_twotower as lt
from signalrec import twotower as tt

FIXED = {"dim": 16, "hidden": 16, "batch_size": 64, "lr": 0.01, "max_history": 50}


def _results(dataset: str = "synthetic", **overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": dataset,
        "chosen": {"two-tower": {"temperature": 0.1, "logq": True, "epochs": 2}},
        "fixed": dict(FIXED),
        "test": {"two-tower": {"seed": 3}},
    }
    result.update(overrides)
    return result


def _refused(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result))
    with pytest.raises(prepare.PrepareError):
        cli._reported_config(path, "synthetic")


def test_a_bool_temperature_in_the_results_file_is_refused(tmp_path: Path) -> None:
    """``_reported_config`` refuses ``"logq": 1`` and ``"epochs": true`` but turns ``"temperature": true`` into a
    configuration (Config.check takes the bool as 1.0), so a corrupted file trains a model nobody chose."""
    result = _results()
    result["chosen"]["two-tower"]["temperature"] = True
    _refused(tmp_path / "twotower.json", result)


def test_missing_fixed_settings_are_refused_not_defaulted(tmp_path: Path) -> None:
    """``_reported_config`` fills every setting missing from ``fixed`` with today's ``Config`` default, so a file
    without them (``"fixed": {}``) is read as dim 64, batch 1,024 and so on, whatever the reported run used."""
    _refused(tmp_path / "twotower.json", _results(fixed={}))


class _Ran(Exception):
    pass


def test_a_results_file_from_differently_prepared_data_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the dataset's name is compared: a twotower.json whose ``prepared`` block (params, counts, sources) is not
    the data on disk is accepted, so the "leaky" side is no longer the reported test run it claims to be."""
    assert cli.main(["synthetic", "--data-dir", str(tmp_path), "--users", "60", "--items", "80"]) == 0
    assert cli.main(["prepare", "synthetic", "--data-dir", str(tmp_path)]) == 0
    manifest = prepare.load(tmp_path / "processed" / "synthetic", "synthetic").manifest
    assert manifest is not None
    other_params = {**manifest["params"], "min_rating": 1.0}
    other_counts = {key: value + 1 for key, value in manifest["counts"].items()}
    source = tmp_path / "twotower.json"
    reported = {"params": other_params, "counts": other_counts, "sources": manifest["sources"]}
    source.write_text(json.dumps(_results(prepared=reported)))

    def ran(*args: object, **kwargs: object) -> None:
        raise _Ran("leakage_twotower.run was reached")

    monkeypatch.setattr(lt, "run", ran)
    args = ["leakage-twotower", "synthetic", "--data-dir", str(tmp_path), "--from", str(source)]
    assert cli.main([*args, "--out", str(tmp_path / "out.json")]) == 1
    assert capsys.readouterr().err.startswith("signalrec: ")


def test_the_well_formed_file_still_reads(tmp_path: Path) -> None:
    """Control for the tests above (passes today): the file they corrupt is accepted as written."""
    path = tmp_path / "twotower.json"
    path.write_text(json.dumps(_results()))
    assert cli._reported_config(path, "synthetic") == tt.Config(**FIXED, temperature=0.1, logq=True, epochs=2, seed=3)  # type: ignore[arg-type]
