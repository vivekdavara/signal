"""Review findings (day 1): what save writes, load can't always read or verify."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from signalrec import prepare
from signalrec.prepare import Params, Prepared, PrepareError


def small_prepared(genres: tuple[tuple[str, ...], ...]) -> Prepared:
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
        genres=genres,
    )


def test_a_genre_with_a_tab_is_refused_by_save_or_round_trips(tmp_path: Path) -> None:
    """Defect: prepare.save refuses a tab or line break in a title (line 193) but not in a genre, and writes it into
    the tab-separated items.tsv; prepare.load (line 247) then can't split the row.

    Input: a Prepared whose first movie's genres are ("Sci\\tFi",). parse_movies accepts such a genre from a movies file
    (it checks only for empty names and surrounding whitespace: ``1,A (2000),Sci<TAB>Fi`` parses).
    Expected: save raises PrepareError (as it does for a title), or load returns the same genres.
    Actual: save succeeds and load raises a bare ValueError ("too many values to unpack"), which the CLI doesn't catch.
    """
    p = small_prepared((("Sci\tFi",), ("Drama",), ()))
    try:
        prepare.save(p, tmp_path / "out", "hand", {}, Params())
    except PrepareError:
        return
    assert prepare.load(tmp_path / "out").genres == p.genres


def test_a_failed_replacement_keeps_the_previous_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect: prepare.save says it writes ``out_dir`` atomically, but it deletes the old directory (line 220) before
    renaming the new one into place (line 221), so a failure or crash between the two leaves neither.

    Input: an existing saved directory, then a second save whose final rename onto ``out_dir`` fails (OSError
    injected into os.replace for that destination).
    Expected: the save fails and ``out_dir`` still holds a loadable dataset (the old one), as an atomic write would.
    Actual: ``out_dir`` no longer exists; load raises FileNotFoundError.
    """
    out = tmp_path / "out"
    prepare.save(small_prepared((("Drama",), (), ())), out, "hand", {}, Params())
    real_replace = os.replace

    # Adapted from the reviewer's version, which failed every rename onto out_dir, including putting the old
    # directory back; no implementation survives that. Here only the new directory's rename in fails.
    def failing_replace(src: str | Path, dst: str | Path) -> None:
        if Path(dst) == out and ".out-old-" not in Path(src).name:
            raise OSError("injected: rename failed")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected"):
        prepare.save(small_prepared((("Comedy",), (), ())), out, "hand", {}, Params())
    monkeypatch.undo()
    assert prepare.load(out).genres[0] in (("Drama",), ("Comedy",))
    assert [p.name for p in tmp_path.iterdir()] == ["out"]  # no temporary or set-aside directory left behind


def test_load_refuses_an_array_the_manifest_does_not_cover(tmp_path: Path) -> None:
    """Defect: prepare.load (lines 238-241) hashes only the files the manifest lists, then loads every name in ARRAYS
    and items.tsv regardless, so a file missing from the manifest is read without any check.

    Input: a saved directory whose manifest no longer lists test_items.npy, and whose test_items.npy was then changed.
    Expected: PrepareError (load reads a file the manifest doesn't vouch for).
    Actual: load returns the changed test items.
    """
    p = small_prepared((("Drama",), (), ()))
    out = tmp_path / "out"
    prepare.save(p, out, "hand", {}, Params())
    manifest = json.loads((out / "manifest.json").read_text())
    del manifest["files"]["test_items.npy"]
    (out / "manifest.json").write_text(json.dumps(manifest))
    np.save(out / "test_items.npy", np.array([1, 1], dtype=np.int64))
    with pytest.raises(PrepareError):
        prepare.load(out)
