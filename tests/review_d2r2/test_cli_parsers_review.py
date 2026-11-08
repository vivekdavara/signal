"""Review D2 round 2: ``--grid`` still lets through a temperature that trains on NaN losses."""

from __future__ import annotations

from pathlib import Path

import pytest

from signalrec import cli


@pytest.mark.parametrize("grid", ["3e-39:on", "0.1:off,3e-39:off"])
def test_a_temperature_that_trains_on_nan_is_a_usage_error(
    grid: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``_temperature_grid`` defers to ``Config.check``, whose 1/temperature bound accepts 3e-39, a float32 denormal on
    which every loss is NaN; the run fails (evaluation refuses NaN scores) only after the grid has trained."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", grid])
    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err
