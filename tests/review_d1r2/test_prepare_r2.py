"""Review round 2 (day 1): prepare's directory replacement and manifest checks after round 1's fixes."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from signalrec import prepare
from signalrec.prepare import Params, Prepared, PrepareError


def small_prepared() -> Prepared:
    def a(values: list[int]) -> np.ndarray:
        return np.array(values, dtype=np.int64)

    return Prepared(
        train_users=a([0, 1]),
        train_items=a([0, 1]),
        train_times=a([1, 1]),
        val_items=a([1, 0]),
        val_times=a([2, 2]),
        test_items=a([2, 2]),
        test_times=a([3, 3]),
        user_raw_ids=a([10, 20]),
        item_raw_ids=a([100, 200, 300]),
        titles=("A (2000)", "B (2000)", "C (2000)"),
        genres=(("Drama",), ("Comedy",), ()),
    )


def test_a_failed_first_rename_leaves_no_empty_old_directory(tmp_path: Path) -> None:
    """Defect: prepare._replace_dir (lines 239-240) makes the ``.<name>-old-*`` directory with mkdtemp and then renames
    the target onto it outside any try, so when that first rename fails the empty directory is left beside
    ``out_dir`` for good (save's cleanup removes only its own temporary directory). Round 1's version (rmtree, then
    one rename) left nothing in this case.

    Input: ``out_dir`` is a symlink to a directory holding an earlier result (as ``data/raw`` is a symlink in this
    very worktree). rename(symlink, empty directory) fails with EISDIR.
    Expected: whatever save does with the link, nothing new beside it and the link's target untouched.
    Actual: save raises IsADirectoryError and ``.out-old-XXXXXXXX`` (empty) stays in ``processed/``.
    """
    # Adapted after review round 3: save now refuses a symlinked out_dir before any rename (a one-line error), so the
    # symlink no longer reaches this path. The first rename's failure is injected instead; the assertion is kept.
    processed = tmp_path / "processed"
    out_dir = processed / "out"
    prepare.save(small_prepared(), out_dir, "synthetic", {}, Params())
    real_replace = os.replace

    def failing_first_rename(src: str | Path, dst: str | Path) -> None:
        if Path(src) == out_dir:
            raise OSError("injected: the old directory can't be moved aside")
        real_replace(src, dst)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "replace", failing_first_rename)
        with contextlib.suppress(OSError):
            prepare.save(small_prepared(), out_dir, "synthetic", {}, Params())
    assert sorted(p.name for p in processed.iterdir()) == ["out"]


def test_a_manifest_format_of_true_is_refused(tmp_path: Path) -> None:
    """Defect (minor): prepare.load (line 268) checks ``manifest.get("format") != FORMAT_VERSION``; in Python
    ``True == 1`` (and ``1.0 == 1``), so a manifest saying ``"format": true`` passes the version check.

    Input: a saved directory whose manifest.json has "format" changed from 1 to true (files untouched).
    Expected: PrepareError naming the format. Actual: loads.
    """
    out_dir = tmp_path / "out"
    prepare.save(small_prepared(), out_dir, "synthetic", {}, Params())
    path = out_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["format"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(PrepareError, match="format"):
        prepare.load(out_dir)
