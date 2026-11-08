"""Review D2: the ``twotower`` command's argument validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from signalrec import cli


def _usage_error(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_a_seed_torch_cannot_take_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--seeds`` accepts 2**64, which ``torch.manual_seed`` refuses, so the run crashes late, not at parse time."""
    too_big = 2**64
    with pytest.raises((ValueError, RuntimeError)):
        torch.manual_seed(too_big)  # the value the parser lets through is unusable
    _usage_error(["twotower", "synthetic", "--data-dir", str(tmp_path), "--seeds", f"0,{too_big}"], capsys)


@pytest.mark.parametrize("grid", ["0.1:on,0.1:on", "0.1:off,0.10:off"])
def test_a_repeated_grid_entry_is_a_usage_error(grid: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--grid`` accepts a repeated (temperature, logQ) pair (``--seeds`` refuses repeats), which trains the same model
    twice and lists the chosen configuration as indistinguishable from itself."""
    _usage_error(["twotower", "synthetic", "--data-dir", str(tmp_path), "--grid", grid], capsys)
