"""The trained model in NumPy, for serving (DESIGN.md, "Day 4: serving and experiments").

Serving computes one user vector per request from the user's latest positives, then retrieves and ranks. PyTorch is
not on that path: the user tower's weights are exported to NumPy (``UserTower``) next to the item vectors, which the
item tower computed once at export time. ``UserTower.vector`` is ``TwoTowerNet.users`` written out:

    mean of the history table's rows for the last ``max_history`` positives (zero for an empty history)
    → x + W2 · relu(W1 · x + b1) + b2
    → divided by its L2 norm (at least 1e-12, as ``F.normalize``)

in float32, as PyTorch computes it. The two can still differ in the last bits (the order of a float32 sum differs
between NumPy and PyTorch's kernels), so the tests bound the difference and check that retrieval returns the same
movies from either vector.

``Model`` bundles what serving reads: the user tower, the item vectors, the ranker's weights and the catalogue's raw
ids and titles. It is saved as one ``.npz`` plus a ``model.json`` holding the configuration, provenance and the
SHA-256 of the ``.npz``; loading checks the hash, so a model on disk is exactly the one exported.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from signalrec.features import Catalogue
from signalrec.ranking import FEATURES, RankerWeights

IntArray = npt.NDArray[np.int64]
Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]

FORMAT_VERSION = 1
_EPS = np.float32(1e-12)  # F.normalize's default eps


class ModelError(ValueError):
    """A model directory is missing, damaged, or doesn't match the data it is served with."""


@dataclass(frozen=True)
class UserTower:
    """``TwoTowerNet``'s user side: the history table (row ``pad`` is the zero padding row) and the residual MLP."""

    history_table: Float32Array  # (n_items + 1, dim)
    w1: Float32Array  # (hidden, dim)
    b1: Float32Array  # (hidden,)
    w2: Float32Array  # (dim, hidden)
    b2: Float32Array  # (dim,)
    max_history: int

    @property
    def n_items(self) -> int:
        return int(self.history_table.shape[0]) - 1

    @property
    def dim(self) -> int:
        return int(self.history_table.shape[1])

    def check(self) -> None:
        n, d = self.history_table.shape
        h = self.w1.shape[0]
        shapes = {"w1": (h, d), "b1": (h,), "w2": (d, h), "b2": (d,)}
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ModelError(f"user tower: {name} has shape {getattr(self, name).shape}, expected {shape}")
        for name in ("history_table", "w1", "b1", "w2", "b2"):
            array = getattr(self, name)
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ModelError(f"user tower: {name} must be finite float32")
        if n < 2:
            raise ModelError("user tower: the history table needs at least one movie and the padding row")
        if self.history_table[n - 1].any():
            raise ModelError("user tower: the padding row is not zero")
        if type(self.max_history) is not int or self.max_history < 1:
            raise ModelError(f"user tower: max_history must be an integer >= 1, got {self.max_history!r}")

    def vector(self, recent: IntArray) -> Float32Array:
        """The user vector from their positives, oldest first; only the last ``max_history`` are read."""
        history = np.asarray(recent, dtype=np.int64)[-self.max_history :] if len(recent) else np.empty(0, np.int64)
        if len(history) and (int(history.min()) < 0 or int(history.max()) >= self.n_items):
            raise ValueError(f"a history movie is outside the catalogue of {self.n_items}")
        if len(history):
            # PyTorch sums the (padded) rows, then divides by the count of real ones; padding rows are zero.
            mean = self.history_table[history].sum(axis=0, dtype=np.float32) / np.float32(len(history))
        else:
            mean = np.zeros(self.dim, dtype=np.float32)
        hidden = np.maximum(self.w1 @ mean + self.b1, np.float32(0))
        x = mean + (self.w2 @ hidden + self.b2)
        norm = np.sqrt(np.dot(x, x), dtype=np.float32)
        out: Float32Array = (x / max(norm, _EPS)).astype(np.float32)
        return out


def export_user_tower(net: Any, max_history: int) -> UserTower:
    """The user tower of a trained ``twotower.TwoTowerNet`` (imported lazily: serving never needs torch)."""
    mlp = net.user_mlp.mlp
    first, second = mlp[0], mlp[2]

    def arr(t: Any) -> Float32Array:
        out: Float32Array = t.detach().cpu().numpy().astype(np.float32).copy()
        return out

    tower = UserTower(
        history_table=arr(net.history_table.weight),
        w1=arr(first.weight),
        b1=arr(first.bias),
        w2=arr(second.weight),
        b2=arr(second.bias),
        max_history=max_history,
    )
    tower.check()
    return tower


@dataclass(frozen=True)
class Model:
    """Everything serving reads besides the feature store."""

    tower: UserTower
    items: Float32Array  # (n_items, dim) unit vectors from the item tower
    ranker: RankerWeights
    item_raw_ids: IntArray  # MovieLens ids, ascending: dense index i is item_raw_ids[i]
    user_raw_ids: IntArray
    titles: tuple[str, ...]
    catalogue: Catalogue  # genres and years: the ranker's static inputs and the feature store's checks
    info: dict[str, Any]  # configuration and provenance, saved in model.json

    @property
    def n_items(self) -> int:
        return len(self.items)

    def check(self) -> None:
        self.tower.check()
        n = self.n_items
        if self.items.ndim != 2 or self.items.shape[1] != self.tower.dim:
            raise ModelError(f"item vectors of shape {self.items.shape} for a {self.tower.dim}-dimensional tower")
        if self.items.dtype != np.float32 or not np.isfinite(self.items).all():
            raise ModelError("item vectors must be finite float32")
        sizes = (self.tower.n_items, len(self.item_raw_ids), len(self.titles), self.catalogue.n_items)
        if sizes != (n, n, n, n):
            raise ModelError(
                f"{n} item vectors, {sizes[0]} history rows, {sizes[1]} movie ids, {sizes[2]} titles, "
                f"{sizes[3]} catalogue movies"
            )
        if self.catalogue.years.shape != (n,):
            raise ModelError(f"{len(self.catalogue.years)} release years for {n} movies")
        n_genres = len(self.catalogue.genre_names)
        if any(g < 0 or g >= n_genres for movie in self.catalogue.movie_genres for g in movie):
            raise ModelError("a movie's genre index is outside the catalogue's genres")
        for name in ("item_raw_ids", "user_raw_ids"):
            ids = getattr(self, name)
            if len(ids) and (np.diff(ids) <= 0).any():
                raise ModelError(f"{name} must be strictly ascending")
        r = self.ranker
        if r.mean.shape != (len(FEATURES),) or r.scale.shape != (len(FEATURES),):
            raise ModelError(f"the ranker standardises {r.mean.shape} features; serving computes {len(FEATURES)}")
        width = len(FEATURES)
        for weight, bias in r.layers:
            if weight.ndim != 2 or weight.shape[1] != width or bias.shape != (weight.shape[0],):
                raise ModelError("the ranker's layers don't chain")
            width = weight.shape[0]
        if not r.layers or width != 1:
            raise ModelError("the ranker must end in one score")
        arrays = [r.mean, r.scale, *(a for layer in r.layers for a in layer)]
        if not all(np.isfinite(a).all() for a in arrays) or not (r.scale > 0).all():
            raise ModelError("the ranker's weights must be finite and its scales positive")

    def movie_index(self, raw: int) -> int:
        at = int(np.searchsorted(self.item_raw_ids, raw))
        if at == len(self.item_raw_ids) or self.item_raw_ids[at] != raw:
            raise KeyError(f"movie {raw} is not in the catalogue")
        return at

    def user_index(self, raw: int) -> int:
        at = int(np.searchsorted(self.user_raw_ids, raw))
        if at == len(self.user_raw_ids) or self.user_raw_ids[at] != raw:
            raise KeyError(f"user {raw} is not in the prepared data")
        return at


def _arrays(model: Model) -> dict[str, npt.NDArray[Any]]:
    out: dict[str, npt.NDArray[Any]] = {
        "history_table": model.tower.history_table,
        "w1": model.tower.w1,
        "b1": model.tower.b1,
        "w2": model.tower.w2,
        "b2": model.tower.b2,
        "items": model.items,
        "item_raw_ids": model.item_raw_ids,
        "user_raw_ids": model.user_raw_ids,
        "years": model.catalogue.years,
        "ranker_mean": model.ranker.mean,
        "ranker_scale": model.ranker.scale,
    }
    for i, (weight, bias) in enumerate(model.ranker.layers):
        out[f"ranker_w{i}"] = weight
        out[f"ranker_b{i}"] = bias
    return out


def save(model: Model, directory: Path) -> dict[str, Any]:
    """Write ``model.npz`` and ``model.json`` into ``directory`` (created); returns the JSON written."""
    model.check()
    directory.mkdir(parents=True, exist_ok=True)
    npz = directory / "model.npz"
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".npz")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez(f, allow_pickle=False, **_arrays(model))
        os.chmod(tmp, 0o644)  # mkstemp makes it 0600
        os.replace(tmp, npz)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    meta = {
        "format_version": FORMAT_VERSION,
        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "max_history": model.tower.max_history,
        "ranker_layers": len(model.ranker.layers),
        "titles": list(model.titles),
        "genre_names": list(model.catalogue.genre_names),
        "movie_genres": [list(g) for g in model.catalogue.movie_genres],
        "info": model.info,
    }
    (directory / "model.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def load(directory: Path) -> Model:
    """The model saved in ``directory``, checked against its hash and shapes."""
    try:
        meta = json.loads((directory / "model.json").read_text())
        raw = (directory / "model.npz").read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelError(f"{directory}: no readable model ({exc}); run signalrec export") from None
    if not isinstance(meta, dict) or meta.get("format_version") != FORMAT_VERSION:
        raise ModelError(f"{directory}: model.json is not format version {FORMAT_VERSION}")
    if hashlib.sha256(raw).hexdigest() != meta.get("npz_sha256"):
        raise ModelError(f"{directory}: model.npz doesn't match the SHA-256 in model.json")
    try:
        with np.load(directory / "model.npz", allow_pickle=False) as z:
            a = {name: z[name] for name in z.files}
        layers = tuple((a[f"ranker_w{i}"], a[f"ranker_b{i}"]) for i in range(int(meta["ranker_layers"])))
        model = Model(
            tower=UserTower(a["history_table"], a["w1"], a["b1"], a["w2"], a["b2"], int(meta["max_history"])),
            items=a["items"],
            ranker=RankerWeights(a["ranker_mean"], a["ranker_scale"], layers),
            item_raw_ids=a["item_raw_ids"].astype(np.int64),
            user_raw_ids=a["user_raw_ids"].astype(np.int64),
            titles=tuple(str(t) for t in meta["titles"]),
            catalogue=Catalogue(
                tuple(str(g) for g in meta["genre_names"]),
                tuple(tuple(int(g) for g in movie) for movie in meta["movie_genres"]),
                a["years"].astype(np.int64),
            ),
            info=dict(meta["info"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelError(f"{directory}: model.npz or model.json is incomplete ({exc!r})") from None
    model.check()
    return model
