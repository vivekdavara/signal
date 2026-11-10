"""Review D4: tower.py's model-file integrity (a damaged or mismatched model must be refused on load)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from signalrec import tower as tw
from signalrec.tower import Model, ModelError


def _saved(model: Model, tmp_path: Path) -> Path:
    out = tmp_path / "model"
    tw.save(model, out)
    return out


def _edit_json(directory: Path, edit: object) -> None:
    path = directory / "model.json"
    meta = json.loads(path.read_text())
    edit(meta)  # type: ignore[operator]
    path.write_text(json.dumps(meta, indent=1) + "\n")


@pytest.mark.parametrize("field", ["max_history", "movie_genres", "genre_names", "titles"])
def test_an_edited_model_json_is_refused(model: Model, tmp_path: Path, field: str) -> None:
    """Defect (tower.py:224-247, save/load): only model.npz is hashed. model.json carries part of the model itself
    (max_history, the catalogue's genres per movie, the genre names, the titles), and a change to any of them loads
    without complaint, although the module promises "a model on disk is exactly the one exported".

    Input: save the fixture model, then change one model.json field (max_history 4 -> 3; two movies' genres swapped;
    two genre names swapped; a title changed), leaving npz_sha256 as written.
    Expected: tw.load raises ModelError. Actual: it loads a different model (other user vectors, the genre filter
    excluding the wrong movies, other titles).
    """
    directory = _saved(model, tmp_path)

    def edit(meta: dict[str, Any]) -> None:
        if field == "max_history":
            meta["max_history"] = meta["max_history"] - 1
        elif field == "movie_genres":
            genres = meta["movie_genres"]
            j = next(j for j in range(1, len(genres)) if genres[j] != genres[0])
            genres[0], genres[j] = genres[j], genres[0]
        elif field == "genre_names":
            names = meta["genre_names"]
            names[0], names[1] = names[1], names[0]
        else:
            meta["titles"][0] = meta["titles"][0] + " (edited)"

    _edit_json(directory, edit)
    with pytest.raises(ModelError):
        tw.load(directory)


@pytest.mark.parametrize("value", ["4", 4.9, True])
def test_a_non_integer_max_history_in_model_json_is_refused(model: Model, tmp_path: Path, value: object) -> None:
    """Defect (tower.py:254, ``int(meta["max_history"])``): load coerces max_history, so "4", 4.9 (read as 4) and
    true (read as 1) pass the UserTower check that demands an integer; true serves every user from one positive.

    Input: model.json max_history set to "4" / 4.9 / true.
    Expected: ModelError. Actual: loads (max_history 4, 4, 1).
    """
    directory = _saved(model, tmp_path)
    _edit_json(directory, lambda meta: meta.__setitem__("max_history", value))
    with pytest.raises(ModelError):
        tw.load(directory)


def test_the_arrays_loaded_are_the_bytes_that_were_hashed(
    model: Model, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect (tower.py:242-251): load hashes ``model.npz``'s bytes, then opens the file a second time to read the
    arrays. If the file is replaced in between (an export writing the same directory while serve starts), the hash
    check passes on the old bytes and the arrays come from the new file: a model whose npz doesn't match its json.

    Input: the npz on disk replaced by one with other item vectors, while the read the hash is computed from returns
    the original bytes (the file changing between the two reads).
    Expected: either ModelError or the original (hashed) arrays. Actual: the replacement's item vectors are served.
    """
    directory = _saved(model, tmp_path)
    npz = directory / "model.npz"
    original = npz.read_bytes()
    other = np.ascontiguousarray(model.items[::-1])
    with np.load(npz, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    arrays["items"] = other
    with open(npz, "wb") as f:
        np.savez(f, allow_pickle=False, **arrays)

    real_read_bytes = Path.read_bytes

    def read_bytes(self: Path) -> bytes:
        return original if self == npz else real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    try:
        loaded = tw.load(directory)
    except ModelError:
        return
    np.testing.assert_array_equal(loaded.items, model.items, err_msg="arrays loaded are not the hashed bytes")
